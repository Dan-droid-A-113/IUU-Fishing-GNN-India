"""Milestone 2 experiment orchestration (cached, restart-safe)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd

from .config import (DATA_DIR, OUTPUT_DIR, PreprocessConfig, TrainConfig, WindowConfig,
                     m2_synthetic_config)

M2_SYN_DIR = DATA_DIR / "synthetic_m2"
M2_PROC_DIR = DATA_DIR / "processed_m2"
CKPT_DIR = OUTPUT_DIR / "checkpoints"


def prepare_m2_data(force: bool = False, verbose: bool = True) -> Dict:
    """Generate (60-day synthetic) -> preprocess -> windows -> labels, caching each stage."""
    from .data_io import anonymize_ids, normalize_schema
    from .graph_builder import gap_prism_pairs, proximity_pairs
    from .india_layers import get_layers
    from .preprocessing import run_preprocessing
    from .synthetic import generate_synthetic_dataset
    from .windows import attach_synthetic_labels, build_windows

    say = print if verbose else (lambda *a, **k: None)
    M2_PROC_DIR.mkdir(parents=True, exist_ok=True)
    raw_p = M2_SYN_DIR / "ais_raw_synthetic.parquet"
    if force or not raw_p.exists():
        say("Generating 60-day synthetic dataset (~1.5-3 min on CPU) ...")
        generate_synthetic_dataset(m2_synthetic_config(), out_dir=M2_SYN_DIR)
    hid = M2_SYN_DIR / "hidden_truth"
    tables = {"raw": pd.read_parquet(raw_p),
              "registry": pd.read_parquet(M2_SYN_DIR / "vessel_registry_synthetic.parquet"),
              "truth": pd.read_parquet(hid / "truth_reports.parquet"),
              "events": pd.read_parquet(hid / "events.parquet")}
    layers = get_layers("IN-GOM")
    ais_p, gaps_p, id_p = M2_PROC_DIR / "ais.parquet", M2_PROC_DIR / "gaps.parquet", M2_PROC_DIR / "_private_id_map.parquet"
    if force or not ais_p.exists():
        say("Preprocessing (clean, features, gaps) ...")
        anon, id_map = anonymize_ids(normalize_schema(tables["raw"], "synthetic_demo", registry=tables["registry"]))
        res = run_preprocessing(anon, layers, PreprocessConfig())
        res["ais"].to_parquet(ais_p)
        res["gaps"].to_parquet(gaps_p)
        id_map.to_parquet(id_p)
        (M2_PROC_DIR / "preprocess_report.json").write_text(json.dumps(res["report"], indent=2))
    ais, gaps, id_map = pd.read_parquet(ais_p), pd.read_parquet(gaps_p), pd.read_parquet(id_p)
    win_p = M2_PROC_DIR / "windows.npz"
    lab_p = M2_PROC_DIR / "window_labels.parquet"
    stale = lab_p.exists() and "fishing_frac" not in pd.read_parquet(lab_p).columns  # cache from older version
    if force or stale or not win_p.exists():
        say("Building 6-h vessel windows + neighbour context ...")
        prism = gap_prism_pairs(ais, gaps)
        pairs = proximity_pairs(ais, radius_m=500, min_bins=4)
        W = build_windows(ais, gaps, layers, prism, WindowConfig())
        lab = attach_synthetic_labels(W, tables["truth"], id_map)
        np.savez_compressed(win_p, X=W["X"], S=W["S"])
        W["agg"].to_parquet(M2_PROC_DIR / "window_agg.parquet")
        lab.to_parquet(M2_PROC_DIR / "window_labels.parquet")
        prism.to_parquet(M2_PROC_DIR / "dark_encounter_links.parquet")
        pairs.to_parquet(M2_PROC_DIR / "proximity_pairs.parquet")
    z = np.load(win_p)
    lab = pd.read_parquet(M2_PROC_DIR / "window_labels.parquet")
    out = {"X": z["X"], "S": z["S"], "agg": pd.read_parquet(M2_PROC_DIR / "window_agg.parquet"),
           "labels": lab, "ais": ais, "gaps": gaps, "id_map": id_map, "events": tables["events"],
           "truth": tables["truth"], "prism": pd.read_parquet(M2_PROC_DIR / "dark_encounter_links.parquet"),
           "pairs": pd.read_parquet(M2_PROC_DIR / "proximity_pairs.parquet"),
           "T0": lab.t0.min().floor("D"), "layers": layers}
    say(f"windows: {len(lab):,} | suspicious (synthetic_demo): {int(lab.y.sum())} ({lab.y.mean():.1%})")
    return out


def make_splits(d: Dict, seed: int = 42, train_days: int = 40, val_days: int = 10) -> Dict[str, np.ndarray]:
    from .splits import (labelled_interaction_pairs, vessel_disjoint_split, window_time_split,
                         window_vessel_split)
    lab, ais = d["labels"], d["ais"]
    vt = ais.groupby("vessel_id").vessel_type.first().to_dict()
    pairs = labelled_interaction_pairs(d["events"], d["id_map"], vt)
    strata = lab.groupby("vessel_id").y.max().to_dict()
    vs = vessel_disjoint_split(list(vt), pairs, strata, seed=seed)
    T0 = d["T0"]
    return {
        "vessel_disjoint": window_vessel_split(lab, vs),
        "time_disjoint": window_time_split(lab, ais, [T0 + pd.Timedelta(days=train_days),
                                                      T0 + pd.Timedelta(days=train_days + val_days)]),
    }


def run_baselines(d: Dict, splits: Dict[str, np.ndarray], cfg: TrainConfig = TrainConfig(),
                  models: Iterable[str] = ("MLP", "GRU", "TCN"), device=None, verbose: bool = True) -> Dict:
    """Rule reference + neural baselines on every split and seed. Returns predictions,
    histories and per-run metrics (thresholds are always chosen on validation)."""
    from .baselines import WindowTensors, get_device, predict, rule_score, train_model
    from .evaluation import binary_metrics, event_level, threshold_max_f1

    device = device or get_device()
    lab = d["labels"]
    y = lab.y.values
    hours = lab.active_hours.values
    rule = rule_score(d["agg"])
    runs, preds, hists = [], {}, {}
    for sname, sp in splits.items():
        tr, va, te = [np.flatnonzero(sp == k) for k in ("train", "val", "test")]
        D = WindowTensors(d["X"], d["S"], d["agg"], y, tr)
        cands = [("Rule (reference)", None)] + [(m, s) for m in models for s in cfg.seeds]
        for mname, seed in cands:
            if mname.startswith("Rule"):
                pv, pt, info = rule[va], rule[te], {"seconds": 0.0, "n_params": 0, "epochs": 0}
            else:
                res = train_model(mname, D, tr, va, cfg, seed, device,
                                  ckpt=CKPT_DIR / f"m2_{mname}_{sname}_seed{seed}.pt")
                pv, pt = predict(res["model"], D, va, device), predict(res["model"], D, te, device)
                hists[(sname, mname, seed)] = res["history"]
                info = {"seconds": res["seconds"], "n_params": res["n_params"], "epochs": len(res["history"])}
            thr = threshold_max_f1(y[va], pv)
            met = binary_metrics(y[te], pt, thr, hours[te])
            ev = event_level(lab.iloc[te], pt >= thr, d["events"], d["id_map"])
            met.update(event_recall=float(ev.detected.mean()) if len(ev) else float("nan"),
                       n_events=int(len(ev)), split=sname, model=mname, seed=seed, **info)
            runs.append(met)
            preds[(sname, mname, seed)] = {"val": pv, "test": pt, "thr": thr, "test_idx": te, "events": ev}
            if verbose:
                print(f"[{sname:15s}] {mname:17s} seed={seed}  PR-AUC {met['pr_auc']:.3f}  F1 {met['f1']:.3f}  "
                      f"FA/1000h {met['false_alerts_per_1000_vessel_h']:.1f}  event recall {met['event_recall']:.2f}")
    return {"runs": pd.DataFrame(runs), "preds": preds, "histories": hists}


def comparison_table(runs: pd.DataFrame) -> pd.DataFrame:
    cols = ["pr_auc", "f1", "precision", "recall", "false_alerts_per_1000_vessel_h", "event_recall"]
    g = runs.groupby(["split", "model"], sort=False)[cols]
    mean, sd = g.mean(), g.std().fillna(0)
    out = mean.round(3).astype(str) + " ± " + sd.round(3).astype(str)
    out.loc[mean.index.get_level_values("model").str.startswith("Rule")] = mean.round(3).astype(str)
    out["n_seeds"] = g.size()
    return out


def best_run(res: Dict, split: str, model: str):
    r = res["runs"]
    r = r[(r.split == split) & (r.model == model)]
    row = r.loc[r.pr_auc.idxmax()]
    seed = None if pd.isna(row.seed) else int(row.seed)
    return res["preds"][(split, model, seed)]


# ==========================================================================
# Milestone 3 - temporal graph model
# ==========================================================================
def run_tgat(d: Dict, G: Dict, splits: Dict[str, np.ndarray], cfg=None, variants=None, device=None,
             verbose: bool = True) -> Dict:
    """Train the temporal GAT (and ablations) on every split/seed; same metrics as run_baselines,
    plus per-task metrics (fishing, gap anomaly, interaction edges, trajectory)."""
    from sklearn.metrics import average_precision_score
    from .baselines import WindowTensors, get_device
    from .evaluation import binary_metrics, event_level, threshold_max_f1
    from .temporal_gnn import GNNConfig, train_tgat
    from .trajectory import endpoint_coverage, path_errors

    cfg = cfg or GNNConfig()
    variants = variants or {"TGAT (temporal graph)": True, "TGAT without graph (ablation)": False}
    device = device or get_device()
    lab = d["labels"]
    y, hours = lab.y.values, lab.active_hours.values
    gd = G["gap_ds"]
    et = G["edge_type"]
    inter = et < 2
    runs, preds, hists, tasks = [], {}, {}, []
    for sname, sp in splits.items():
        tr, va, te = [np.flatnonzero(sp == k) for k in ("train", "val", "test")]
        T = WindowTensors(d["X"], d["S"], d["agg"], y, tr)
        te_mask = sp == "test"
        gap_te = (gd["end_node"] >= 0) & te_mask[np.clip(gd["end_node"], 0, None)]
        for vname, use_graph in variants.items():
            for seed in cfg.seeds:
                r = train_tgat(G, T, sp, cfg, seed, device, use_graph,
                               ckpt=CKPT_DIR / f"m3_{'tgat' if use_graph else 'nograph'}_{sname}_seed{seed}.pt")
                pv, pt = r["risk"][va], r["risk"][te]
                thr = threshold_max_f1(y[va], pv)
                met = binary_metrics(y[te], pt, thr, hours[te])
                ev = event_level(lab.iloc[te], pt >= thr, d["events"], d["id_map"])
                met.update(event_recall=float(ev.detected.mean()) if len(ev) else float("nan"), n_events=int(len(ev)),
                           split=sname, model=vname, seed=seed, seconds=r["seconds"], n_params=r["n_params"],
                           epochs=len(r["history"]))
                runs.append(met)
                preds[(sname, vname, seed)] = {"val": pv, "test": pt, "thr": thr, "test_idx": te, "events": ev,
                                               "raw": r}
                hists[(sname, vname, seed)] = r["history"]
                # --- auxiliary task metrics on the test split
                t = {"split": sname, "model": vname, "seed": seed}
                t["fishing_pr_auc"] = average_precision_score(G["y_fishing"][te], r["fish"][te])
                gm = te_mask & G["gap_mask"]
                t["gap_anomaly_pr_auc"] = average_precision_score(G["y_gap"][gm], r["gap"][gm]) if G["y_gap"][gm].any() else np.nan
                if use_graph and len(r["edge"]):
                    ei = r["edge_index_inter"]
                    em = te_mask[ei[0]] & te_mask[ei[1]]
                    ey = G["edge_y"][inter][em]
                    t["interaction_edge_pr_auc"] = average_precision_score(ey, r["edge"][em]) if ey.any() else np.nan
                    t["test_interaction_edges"] = int(em.sum())
                P = r["gap_path"][gap_te]
                t.update({f"traj_{k}": v for k, v in path_errors(P, gd["truth_path"][gap_te]).items()})
                t["traj_endpoint_90pct_coverage"] = endpoint_coverage(P[:, -1], r["gap_logsig"][gap_te], gd["end_km"][gap_te])
                tasks.append(t)
                if verbose:
                    print(f"[{sname:15s}] {vname:30s} seed={seed}  PR-AUC {met['pr_auc']:.3f}  F1 {met['f1']:.3f}  "
                          f"FA/1000h {met['false_alerts_per_1000_vessel_h']:.1f}  event recall {met['event_recall']:.2f}  "
                          f"({r['seconds']:.0f}s, {len(r['history'])} ep)")
        # non-learning trajectory references on the same test gaps
        for ref, P in {"dead reckoning": gd["dead_reckoning"][gap_te],
                       "stay at last fix": np.zeros_like(gd["dead_reckoning"][gap_te])}.items():
            t = {"split": sname, "model": f"{ref} (reference)", "seed": None}
            t.update({f"traj_{k}": v for k, v in path_errors(P, gd["truth_path"][gap_te]).items()})
            tasks.append(t)
    return {"runs": pd.DataFrame(runs), "preds": preds, "histories": hists, "tasks": pd.DataFrame(tasks)}


def merge_results(*results: Dict) -> Dict:
    return {"runs": pd.concat([r["runs"] for r in results], ignore_index=True),
            "preds": {k: v for r in results for k, v in r["preds"].items()},
            "histories": {k: v for r in results for k, v in r["histories"].items()}}
