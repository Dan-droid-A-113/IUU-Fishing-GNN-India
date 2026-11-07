"""Generate the project notebooks from source (keeps them reviewable and reproducible).

    python scripts/build_notebooks.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

SETUP = r'''# ---- Setup: works in Google Colab (CPU is enough) and locally --------------------
import os, sys, subprocess, pathlib, zipfile, glob
IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/Dan-droid-A-113/IUU-Fishing-GNN-India"   # clones the code; set to "" to upload the zip instead
USE_DRIVE = False      # True = keep project + outputs on Google Drive (restart-safe)

if IN_COLAB:
    PROJECT_DIR = pathlib.Path("/content/india-fishing-gnn")
    if USE_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")
        PROJECT_DIR = pathlib.Path("/content/drive/MyDrive/india-fishing-gnn")
    if not (PROJECT_DIR / "src").exists():
        if REPO_URL:
            subprocess.run(["git", "clone", "-q", REPO_URL, str(PROJECT_DIR)], check=True)
        else:
            # accepts renamed downloads such as "india-fishing-gnn (1).zip"
            zips = sorted(glob.glob("/content/india-fishing-gnn*.zip"), key=os.path.getmtime)
            if not zips:
                from google.colab import files
                print("Project zip not found - choose india-fishing-gnn.zip from your computer:")
                uploaded = files.upload()   # file picker appears below this cell
                zips = [os.path.abspath(n) for n in uploaded if n.endswith(".zip")]
            if not zips:
                raise FileNotFoundError("No project zip uploaded.")
            print("Extracting", zips[-1])
            zipfile.ZipFile(zips[-1]).extractall(PROJECT_DIR.parent)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r",
                    str(PROJECT_DIR / "requirements-colab.txt")], check=True)
else:
    PROJECT_DIR = pathlib.Path.cwd()
    while not (PROJECT_DIR / "src").exists() and PROJECT_DIR != PROJECT_DIR.parent:
        PROJECT_DIR = PROJECT_DIR.parent

os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))
print("Project directory:", PROJECT_DIR, "| Colab:", IN_COLAB)'''


def nb01():
    cells = [
        md("""# 01 · India-focused synthetic AIS demo (Milestone 1)

**Pilot region:** Gulf of Mannar – Palk Bay, south Tamil Nadu coast (`IN-GOM`).

This notebook runs from a clean runtime on **CPU only** (no GPU cell in this milestone; ~1–2 min):

1. generates a clearly labelled **synthetic** India-coast AIS dataset (Mode A);
2. normalises it to the canonical schema and **anonymises** vessel identifiers;
3. cleans it (duplicates, invalid coordinates, impossible speeds, AIS sentinels);
4. computes movement and India-specific geospatial features;
5. detects **explicit AIS-gap events** with reception and boundary context;
6. validates gap detection against hidden simulation truth;
7. plots trajectories on an India-focused map (static + interactive);
8. builds leakage-safe train/validation/test splits;
9. saves processed Parquet files, figures and metrics.

> ⚠️ **Synthetic demo data.** Every vessel, track and label here is simulated (`synthetic_demo_*`). Nothing in this notebook is evidence about real vessels or real illegal fishing. Boundaries and protected-area outlines are approximate placeholders. Outputs describe *suspicious behaviour / investigation priority*, never illegality."""),
        code(SETUP),
        md("## 0 · Configuration"),
        code('''import json, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
warnings.filterwarnings("ignore", category=FutureWarning)

from src.config import (SyntheticConfig, PreprocessConfig, SYNTHETIC_DIR, PROCESSED_DIR,
                        FIGURE_DIR, METRIC_DIR, REGIONS, SYNTHETIC_CAVEAT, SEED)
from src.india_layers import get_layers
from src.synthetic import generate_synthetic_dataset
from src.data_io import normalize_schema, anonymize_ids, save_table
from src.preprocessing import run_preprocessing, resample_tracks
from src.graph_builder import proximity_pairs, gap_prism_pairs
from src.splits import (vessel_disjoint_split, time_disjoint_split, region_disjoint_split,
                        segment_table, check_no_leakage)
from src.evaluation import attach_gap_truth, planted_dark_windows, gap_detection_report
from src import visualization as viz

np.random.seed(SEED)
for d in (SYNTHETIC_DIR, PROCESSED_DIR, FIGURE_DIR, METRIC_DIR):
    d.mkdir(parents=True, exist_ok=True)

FORCE_REGENERATE = False          # True = rebuild synthetic data even if cached
SYN_CFG = SyntheticConfig()       # 7 days, 50 vessels, seed 42
PRE_CFG = PreprocessConfig()      # gap threshold 60 min, max speed 40 kn, ...
print(SYNTHETIC_CAVEAT)
pd.DataFrame([vars(r) for r in REGIONS.values()])'''),
        md("""## 1 · Generate (or load cached) synthetic AIS — Mode A

The simulator moves vessels minute by minute around real coastline geometry (offline land mask) and then *observes* them through a coverage model: synthetic coastal receivers, weak satellite pick-up for Class B fishing transponders, one scheduled receiver-cluster outage, transmitter glitches, intentional blackouts, and AIS switched off in port. Observable AIS and **hidden truth** are written to separate files; hidden truth is used only for evaluation."""),
        code('''raw_path = SYNTHETIC_DIR / "ais_raw_synthetic.parquet"
if raw_path.exists() and not FORCE_REGENERATE:
    tables = {"ais_raw": pd.read_parquet(raw_path),
              "registry": pd.read_parquet(SYNTHETIC_DIR / "vessel_registry_synthetic.parquet")}
    for k in ("truth_reports", "events", "vessel_truth", "injected_errors"):
        tables[k] = pd.read_parquet(SYNTHETIC_DIR / "hidden_truth" / f"{k}.parquet")
    print("Loaded cached synthetic dataset")
else:
    tables = generate_synthetic_dataset(SYN_CFG)
    print("Generated and saved synthetic dataset")

raw, registry, events = tables["ais_raw"], tables["registry"], tables["events"]
print(f"raw AIS rows: {len(raw):,} | vessels: {raw.mmsi.nunique()} | "
      f"time span: {raw.timestamp.min()} -> {raw.timestamp.max()}")
display(registry.vessel_class.value_counts().rename("vessels").to_frame())
display(raw.head())'''),
        code('''# Planted scenarios (hidden truth - evaluation only)
display(events.groupby(["event_type", "synthetic_demo_label"]).size().rename("events").to_frame())
display(tables["truth_reports"].miss_reason.value_counts(normalize=True).round(3)
        .rename("share of scheduled AIS reports").to_frame())'''),
        md("## 2 · Canonical schema and anonymisation\n\nRaw identifiers are replaced by keyed-hash IDs (`vessel_xxxxxxxx`). The mapping is kept in `data/processed/_private/` (git-ignored). For real data set a private salt in the `AIS_ID_SALT` environment variable."),
        code('''canon = normalize_schema(raw, source="synthetic_demo", registry=registry)
ais_anon, id_map = anonymize_ids(canon)
save_table(id_map, PROCESSED_DIR / "_private" / "id_map_synthetic")
assert not ais_anon.astype(str).apply(lambda c: c.str.contains("SYNTH")).any().any()
display(ais_anon.head())'''),
        md("## 3 · Cleaning, kinematics, geospatial features, AIS-gap events"),
        code('''layers = get_layers("IN-GOM")
display(layers.metadata_table()[["name", "kind", "crs", "source", "uncertainty"]])

res = run_preprocessing(ais_anon, layers, PRE_CFG)
ais, gaps, report = res["ais"], res["gaps"], res["report"]
display(pd.Series(report, name="value").to_frame())'''),
        code('''# Did cleaning catch the injected data-quality errors?
inj = tables["injected_errors"].error_type.value_counts()
check = pd.DataFrame({
    "injected": inj,
    "handled": pd.Series({"duplicate": report["duplicates_removed"],
                          "invalid_coordinate": report["invalid_coordinates_removed"],
                          "position_jump": report["impossible_speed_flagged"] + report["outside_regions_removed"],
                          "sog_not_available": report["sog_not_available"]})})
display(check)'''),
        code('''cols = ["vessel_id", "timestamp", "latitude", "longitude", "sog", "dt_minutes", "distance_m",
        "turn_rate", "fishing_motion_score", "mpa_distance_m", "eez_distance_m", "port_distance_m",
        "ais_gap_id", "gap_duration_min", "sar_detection_flag", "segment_id"]
display(ais[cols].sample(5, random_state=1))
gap_cols = ["ais_gap_id", "vessel_id", "vessel_type", "gap_start", "gap_duration_min", "displacement_m",
            "gap_context", "concurrent_silence_frac", "n_neighbours_before", "crossed_boundary",
            "min_mpa_distance_m", "pre_gap_sog_mean"]
display(gaps[gaps.gap_context == "at_sea"][gap_cols].head(10))'''),
        md("""## 4 · Validate gap detection against hidden truth (synthetic only)

Two things must hold: every planted blackout longer than the threshold is detected, **and** most detected gaps are benign (port stays, coverage, receiver outage, transmitter faults). The second point is why an AIS gap alone must never be treated as a suspicion label."""),
        code('''gaps_t = attach_gap_truth(gaps, tables["truth_reports"], id_map)
dark_w = planted_dark_windows(tables["truth_reports"], id_map)
gap_metrics = gap_detection_report(gaps_t, dark_w, PRE_CFG.gap_threshold_min)
print(json.dumps(gap_metrics, indent=2))
display(pd.crosstab(gaps_t.truth_cause, gaps_t.gap_context, margins=True))
within = gaps_t[gaps_t.truth_cause != "between_voyages"]
display(within.groupby("truth_cause").agg(
    n=("gap_duration_min", "size"), median_duration_min=("gap_duration_min", "median"),
    mean_concurrent_silence=("concurrent_silence_frac", "mean"),
    median_displacement_km=("displacement_m", lambda s: s.median() / 1000)).round(2))
fig = viz.plot_gap_diagnostics(gaps_t, PRE_CFG.gap_threshold_min, FIGURE_DIR / "m1_gap_diagnostics.png")
plt.show()'''),
        md("## 5 · Maps"),
        code('''fig = viz.plot_tracks_map(ais, layers, gaps,
                          title="Synthetic AIS tracks - Gulf of Mannar / Palk Bay pilot (7 days)",
                          path=FIGURE_DIR / "m1_tracks_map.png")
plt.show()'''),
        code('''# Zoom on one planted blackout: what the pipeline sees vs. hidden truth
ex = gaps_t[gaps_t.truth_cause == "dark_intentional"].sort_values("gap_duration_min").iloc[-1]
raw_id = id_map.set_index("vessel_id").raw_id[ex.vessel_id]
tr = tables["truth_reports"]
true_path = tr[(tr.raw_id == raw_id) & (tr.timestamp >= ex.gap_start) & (tr.timestamp <= ex.gap_end)]
fig = viz.plot_gap_example(ais, ex, layers, true_path, path=FIGURE_DIR / "m1_planted_blackout_example.png",
                           title=f"Planted synthetic blackout - {ex.vessel_id} ({ex.vessel_type})")
plt.show()'''),
        code('''m = viz.folium_map(ais, layers, gaps, path=FIGURE_DIR / "m1_tracks_map.html")
m  # interactive; background tiles need internet'''),
        code('''# Feature sanity check: do kinematic features behave as expected per simulated behaviour?
obs_truth = tr[tr.observed].merge(id_map, on="raw_id")[["vessel_id", "timestamp", "behavior"]]
# align timestamp precision (pandas 2 vs 3 may differ between the two tables)
ais_ts = ais.assign(timestamp=ais.timestamp.astype("datetime64[ns, UTC]"))
obs_truth = obs_truth.assign(timestamp=obs_truth.timestamp.astype("datetime64[ns, UTC]"))
joined = pd.merge_asof(ais_ts.sort_values("timestamp"), obs_truth.sort_values("timestamp"),
                       on="timestamp", by="vessel_id", tolerance=pd.Timedelta("2s"), direction="nearest")
fig = viz.plot_behaviour_features(joined, FIGURE_DIR / "m1_behaviour_features.png")
plt.show()
display(joined.groupby("behavior")[["sog", "fishing_motion_score"]].median().round(2))'''),
        md("## 6 · Optional resampling (never across long gaps)"),
        code('''rs = resample_tracks(ais, PRE_CFG)
print(f"{len(rs):,} resampled rows at {PRE_CFG.resample_min}-min; "
      f"max bracket span = {rs.bracket_span_min.max():.1f} min (limit {PRE_CFG.max_interp_gap_min})")
assert (rs.bracket_span_min <= PRE_CFG.max_interp_gap_min).all()'''),
        md("""## 7 · Leakage-safe splits

* **Vessel-disjoint**: vessels are grouped so an interaction never straddles splits. Two link types: (a) ≥ 40 min within 500 m of each other outside ports (BallTree search, not all-pairs); (b) *possible dark encounters* — a slow vessel that the silent vessel could have reached during its AIS gap (space-time prism test). Link (b) is what keeps blackout rendezvous partners together.
* **Time-disjoint**: days 1–4 train, day 5 validation, days 6–7 test, with a 6 h embargo; voyage segments that touch a boundary are dropped, never cut.
* **Region-disjoint**: only `IN-GOM` exists so far; the function is ready for when the Gujarat–Maharashtra or Odisha–West Bengal regions are added."""),
        code('''pairs = proximity_pairs(ais, radius_m=500, bin_min=10, min_bins=4, exclude_port_m=3000)
prism = gap_prism_pairs(ais, gaps, vmax_kn=10, max_partner_sog_kn=2, min_bins=3)
print(f"{len(pairs)} sustained close-proximity pairs | {len(prism)} possible-dark-encounter links")
links = list(zip(pairs.vessel_a, pairs.vessel_b)) + list(zip(prism.vessel_a, prism.vessel_b))

# stratify by whether a vessel has any planted suspicious report (labels used ONLY to balance splits)
sus_raw = set(tr.loc[tr.suspicious, "raw_id"])
strata = {v: int(r in sus_raw) for r, v in zip(id_map.raw_id, id_map.vessel_id)}
vsplit = vessel_disjoint_split(ais.vessel_id.unique(), links, strata, seed=SEED)
ais["split_vessel"] = ais.vessel_id.map(vsplit)

seg = segment_table(ais)
t0 = ais.timestamp.min().floor("D")
tsplit = time_disjoint_split(seg, [t0 + pd.Timedelta(days=4), t0 + pd.Timedelta(days=5)], pd.Timedelta("6h"))
ais["split_time"] = ais.segment_id.map(tsplit)
rsplit = region_disjoint_split(ais.region_id.unique(), test_regions=[])

# planted rendezvous partners must end up in the same vessel split
rdv = events[events.event_type == "low_speed_rendezvous"]
to_v = dict(zip(id_map.raw_id, id_map.vessel_id))
same = [len({vsplit.get(to_v.get(x)) for x in e.split(",")}) == 1 for e in rdv.vessels]
split_summary = {
    "vessel_split_counts": pd.Series(vsplit).value_counts().to_dict(),
    "suspicious_vessels_per_split": pd.Series({v: vsplit[v] for v, s in strata.items() if s}).value_counts().to_dict(),
    "time_split_segments": tsplit.value_counts().to_dict(),
    "region_split": rsplit,
    "rendezvous_pairs_kept_together": f"{sum(same)}/{len(same)}",
    "vessel_split_leakage_check": check_no_leakage(ais, "split_vessel"),
    "time_split_segment_check": check_no_leakage(ais, "split_time")["segments_unique"],
}
print(json.dumps(split_summary, indent=2, default=str))'''),
        md("## 8 · Save processed data and metrics"),
        code('''p1 = save_table(ais, PROCESSED_DIR / "ais_processed_synthetic")
p2 = save_table(gaps, PROCESSED_DIR / "ais_gaps_synthetic")
p3 = save_table(pairs, PROCESSED_DIR / "proximity_pairs_synthetic")
save_table(prism, PROCESSED_DIR / "dark_encounter_links_synthetic")
summary = {
    "milestone": 1, "data_mode": "A_synthetic", "caveat": SYNTHETIC_CAVEAT, "region": "IN-GOM",
    "n_raw_rows": int(len(raw)), "n_clean_rows": int(len(ais)), "n_vessels": int(ais.vessel_id.nunique()),
    "vessel_hours_observed": float(ais.groupby("segment_id").timestamp.agg(lambda s: (s.max() - s.min()).total_seconds() / 3600).sum()),
    "cleaning": report, "gap_detection": gap_metrics, "splits": split_summary,
    "preprocess_config": PRE_CFG.to_dict(), "synthetic_config": SYN_CFG.to_dict(),
}
(METRIC_DIR / "m1_synthetic_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print("saved:", p1, p2, p3, METRIC_DIR / "m1_synthetic_summary.json", sep="\\n  ")
print("figures:", *sorted(p.name for p in FIGURE_DIR.glob("m1_*")), sep="\\n  ")'''),
        md("""## What Milestone 1 can and cannot conclude

**Can conclude**
* The pipeline runs end-to-end from a clean runtime: synthetic generation → schema → anonymisation → cleaning → kinematic and India-specific geospatial features → explicit AIS-gap events → maps → leakage-safe splits → saved artefacts.
* Cleaning catches the injected data errors, and the gap detector recovers **every planted blackout** longer than the threshold.
* Most detected gaps are **benign** (port stays, coverage, receiver outage, transmitter faults), so gap duration alone is not a usable suspicion signal. Context features such as concurrent silence, boundary crossing and displacement are needed.

**Cannot conclude**
* Nothing about real vessels, real fishing, or real illegal activity in Indian waters. All data are synthetic.
* Nothing about detection performance on real AIS. The coverage model, behaviours and label rates are assumptions.
* The MPA, maritime-boundary and fishing-ground layers are approximate placeholders and must be replaced by authoritative layers (WDPA, Marine Regions) before any real-data run.
* No model has been trained yet. Risk scores, graph learning and explanations start in Milestones 2–4.

**Next:** Milestone 2, deep-learning baselines (rule reference, MLP, GRU, TCN) trained on these splits."""),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}, "colab": {"provenance": []}}
    return nb


def nb03():
    cells = [
        md("""# 03 · Deep-learning baselines (Milestone 2)

**Pilot region:** Gulf of Mannar – Palk Bay (`IN-GOM`). **Data:** 60-day synthetic simulation (Mode A).

What this notebook does:
1. builds (or loads cached) a 60-day synthetic dataset — 62 vessels, four suspicious carrier networks;
2. turns tracks into **6-hour vessel windows** (36 × 10-min steps with an explicit *observed* mask; silent steps are never interpolated);
3. makes **vessel-disjoint** and **time-disjoint** splits (no vessel / voyage leakage);
4. evaluates a transparent **rule-based reference** and trains three **neural baselines** — MLP (window aggregates), bidirectional GRU and dilated TCN (step sequences) — with 3 seeds each;
5. reports PR-AUC, precision/recall/F1 at a validation-calibrated threshold, **false alerts per 1,000 vessel-hours**, **event-level recall**, confusion matrices, learning curves, and a **false-alert gallery** (anchorage, MPA transit, receiver outage, loitering carriers, …).

**Runtime:** CPU only is fine — about 3 min for data + about 8 min for training (3 seeds × 3 models × 2 splits). A GPU (Runtime → Change runtime type → T4) speeds up the GRU/TCN cells; nothing requires it. Set `FAST_MODE = True` for a 1-seed run (~3 min).

> ⚠️ **Synthetic demo data.** Labels are planted simulation events (`synthetic_demo_suspicious`). Scores mean *investigation priority*, never illegality. The rule reference is **not** the contribution; the neural baselines exist to be beaten by the temporal graph model in Milestone 3."""),
        code(SETUP),
        md("## 0 · Configuration"),
        code("""import json, warnings
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from IPython.display import display
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src.config import TrainConfig, WindowConfig, FIGURE_DIR, METRIC_DIR, SYNTHETIC_CAVEAT
from src.experiments import prepare_m2_data, make_splits, run_baselines, comparison_table, best_run, M2_PROC_DIR
from src.baselines import get_device, RULES
from src.windows import STEP_FEATURES, STATIC_FEATURES
from src import visualization as viz

FAST_MODE = False                  # True = 1 seed per model (quick look)
FORCE_REBUILD = False              # True = regenerate synthetic data and windows
CFG = TrainConfig(seeds=(0,) if FAST_MODE else (0, 1, 2))
DEVICE = get_device()              # GPU if available, otherwise CPU
torch.set_num_threads(max(1, torch.get_num_threads()))
for p in (FIGURE_DIR, METRIC_DIR):
    p.mkdir(parents=True, exist_ok=True)
print("device:", DEVICE, "| seeds:", CFG.seeds)
print(SYNTHETIC_CAVEAT)"""),
        md("""## 1 · Data: 60-day simulation → preprocessing → 6-h windows (cached)

Each stage is cached under `data/synthetic_m2/` and `data/processed_m2/`, so a restarted runtime resumes quickly."""),
        code("""d = prepare_m2_data(force=FORCE_REBUILD)
lab = d["labels"]
display(pd.Series(json.loads((M2_PROC_DIR / "preprocess_report.json").read_text()), name="value").to_frame().T)
ev = d["events"]
display(ev.groupby(["event_type", "synthetic_demo_label"]).size().rename("planted events").to_frame())
display(lab.context.value_counts().rename("windows").to_frame())"""),
        code("""print(f"Sequence tensor: {d['X'].shape}  (windows x 10-min steps x features)")
print("Step features :", ", ".join(STEP_FEATURES))
print("Static        :", ", ".join(STATIC_FEATURES))
print(f"MLP aggregates: {d['agg'].shape[1]} columns ->", ", ".join(d["agg"].columns[:12]), "...")"""),
        md("""## 2 · Leakage-safe splits

* **Vessel-disjoint** — whole vessels per split. Vessels in a *labelled* interaction (planted rendezvous, fleet pairs, tug assists — tugs excluded because they meet almost every ship) are grouped, so each suspicious carrier network sits entirely in one split (2 train / 1 validation / 1 test). Over 60 days, *unlabelled* proximity links chain almost all vessels together, so they cannot be used for grouping; the number of such cross-split contacts is reported instead.
* **Time-disjoint** — days 1–40 train, 41–50 validation, 51–60 test, 6-h embargo; windows touching a voyage that straddles a boundary are dropped.
* **Region-disjoint** — not possible yet (one pilot region)."""),
        code("""from src.splits import cross_split_contacts
splits = make_splits(d)
for name, sp in splits.items():
    display(pd.crosstab(pd.Series(sp, name=name), lab.y.rename("synthetic_demo_suspicious")))
vsplit = splits["vessel_disjoint"]
print("cross-split sustained-proximity pairs touching the test vessels:",
      cross_split_contacts(lab, vsplit, d["pairs"]))"""),
        md("## 3 · Rule-based reference (transparent, not learned)"),
        code("""display(pd.Series(RULES, name="rule").to_frame())"""),
        md("""## 4 · Train and evaluate (rule + MLP + GRU + TCN, all seeds, both splits)

Loss: class-weighted binary cross-entropy. Optimiser: AdamW + ReduceLROnPlateau. Early stopping on **validation PR-AUC**; the alert threshold is also chosen on validation (max F1) and then applied unchanged to the test set. Checkpoints go to `outputs/checkpoints/`. *GPU optional — CPU ~8 min.*"""),
        code("""res = run_baselines(d, splits, CFG, device=DEVICE)
res["runs"].to_csv(METRIC_DIR / "m2_runs_all_seeds.csv", index=False)
table = comparison_table(res["runs"])
table.to_csv(METRIC_DIR / "m2_model_comparison.csv")
display(table)"""),
        md("## 5 · Curves and confusion matrices"),
        code("""fig = viz.plot_pr_curves(res, lab, FIGURE_DIR / "m2_pr_curves.png"); plt.show()
for sp in splits:
    fig = viz.plot_confusion_matrices(res, lab, sp, FIGURE_DIR / f"m2_confusion_{sp}.png"); plt.show()"""),
        code("""for sp in splits:
    fig = viz.plot_learning_curves(res, sp, FIGURE_DIR / f"m2_learning_curves_{sp}.png"); plt.show()"""),
        md("## 6 · Event-level detection and performance by vessel type"),
        code("""rows = []
for (sp, m, seed), p in res["preds"].items():
    e = p["events"]
    if len(e):
        for t, g in e.groupby("event_type"):
            rows.append({"split": sp, "model": m, "seed": seed, "event_type": t, "events": len(g),
                         "recall": g.detected.mean()})
ev_tab = (pd.DataFrame(rows).groupby(["split", "event_type", "model"]).recall.mean()
          .unstack("model").round(2))
ev_tab.to_csv(METRIC_DIR / "m2_event_recall_by_type.csv")
display(ev_tab)

from src.evaluation import binary_metrics
rows = []
for sp in splits:
    for m in res["runs"][res["runs"].split == sp].model.unique():
        p = best_run(res, sp, m)
        L = lab.iloc[p["test_idx"]]
        for vt, idx in L.groupby("vessel_type").indices.items():
            mm = binary_metrics(L.y.values[idx], p["test"][idx], p["thr"], L.active_hours.values[idx])
            rows.append({"split": sp, "model": m, "vessel_type": vt, "windows": len(idx), "positives": mm["positives"],
                         "recall": mm["recall"], "false_alerts_per_1000_vessel_h": mm["false_alerts_per_1000_vessel_h"]})
vt_tab = pd.DataFrame(rows)
vt_tab.to_csv(METRIC_DIR / "m2_by_vessel_type.csv", index=False)
display(vt_tab[vt_tab.split == "vessel_disjoint"].pivot(index="vessel_type", columns="model",
        values="false_alerts_per_1000_vessel_h").round(1))"""),
        md("## 7 · Where do false alerts come from? (hard negatives)"),
        code("""for sp in splits:
    t = viz.fp_by_context(res, lab, sp)
    t.to_csv(METRIC_DIR / f"m2_false_alerts_by_context_{sp}.csv", index=False)
    fig = viz.plot_fp_by_context(t, sp, FIGURE_DIR / f"m2_false_alerts_by_context_{sp}.png"); plt.show()"""),
        code("""r = res["runs"]
best_nn = (r[(r.split == "vessel_disjoint") & (r.model != "Rule (reference)")]
           .groupby("model").pr_auc.mean().idxmax())
for m in dict.fromkeys([best_nn, "TCN"]):
    fig = viz.plot_fp_gallery(d, best_run(res, "vessel_disjoint", m), "vessel_disjoint", m,
                              FIGURE_DIR / f"m2_false_alert_gallery_{m}.png")
    if fig is not None:
        plt.show()"""),
        md("## 8 · Save summary"),
        code("""summary = {
    "milestone": 2, "data_mode": "A_synthetic (60 days)", "caveat": SYNTHETIC_CAVEAT,
    "n_windows": int(len(lab)), "positives": int(lab.y.sum()), "device": str(DEVICE),
    "train_config": vars(CFG), "window_config": vars(WindowConfig()),
    "comparison_mean": res["runs"].groupby(["split", "model"])[["pr_auc", "f1", "precision", "recall",
        "false_alerts_per_1000_vessel_h", "event_recall"]].mean().round(4).reset_index().to_dict("records"),
}
(METRIC_DIR / "m2_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print("metrics:", *sorted(p.name for p in METRIC_DIR.glob("m2_*")), sep="\\n  ")
print("figures:", *sorted(p.name for p in FIGURE_DIR.glob("m2_*")), sep="\\n  ")"""),
        md("""## What Milestone 2 can and cannot conclude

**Can conclude (synthetic pilot)**
* The full learning pipeline works end to end — windowing, leakage-safe splits, class-weighted training with early stopping, validation-calibrated thresholds, window- and event-level metrics — on CPU in Colab.
* Planted suspicious episodes are recoverable from AIS kinematics, explicit gap events, boundary/MPA context and neighbour summaries. Compare models in the table above; the MLP on hand-built window aggregates (including gap-event features) is the strongest baseline, and the TCN beats the GRU among sequence models.
* **Hard negatives dominate false alerts:** benign AIS gaps (transmitter faults, coverage holes, receiver outages) and loitering carriers are the main confusions, which is exactly the context a graph model with neighbour messages and reception evidence should help with.

**Cannot conclude**
* Anything about real vessels or real Indian waters — labels are planted by the simulator, and the synthetic suspicious behaviours are partly "rule-shaped", which flatters the rule reference.
* How the models would rank on real AIS, with real class imbalance and label noise.
* Region transfer (only one pilot region exists).

**Next — Milestone 3:** a temporal graph-attention network (vessels = nodes, encounters and possible dark encounters = edges) with multi-task heads and a neural gap-trajectory decoder, compared against these baselines on the same splits."""),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}, "colab": {"provenance": []}}
    return nb


def nb04():
    cells = [
        md("""# 04 · Temporal graph-attention network (Milestone 3)

**Pilot region:** Gulf of Mannar – Palk Bay (`IN-GOM`). **Data:** the same 60-day synthetic simulation and splits as notebook 03.

**The model**
- **Nodes:** each vessel in each 6-hour window.
- **Edges:** (a) proximity encounters, (b) *possible dark encounters*, meaning meetings that could have happened while one vessel was silent, and (c) links to the same vessel 6 h and 12 h earlier (past → present only).
- **Layers:** a TCN + MLP node encoder, then 2 GATv2 layers with edge features and a learned time encoding (PyTorch Geometric).
- **Heads:**
  - fishing activity;
  - suspicious interaction, scored per edge;
  - AIS-gap anomaly;
  - a neural decoder that predicts where a silent vessel went, with a 90% uncertainty region;
  - the final **investigation score**, learned from the node embedding plus all the other heads.

The notebook trains the model and a **no-graph ablation** (same model, all edges removed), retrains the MLP and TCN baselines on the same splits, and compares them. Every loss term is reported separately.

**Runtime:**
- **GPU recommended** (Runtime → Change runtime type → T4 GPU): about 10–15 min in total.
- **CPU only:** about 35–40 min.
- **`FAST_MODE = True`:** 1 seed, about 12 min on CPU.

> ⚠️ **Synthetic demo data.** Scores mean *investigation priority*, never illegality. Graph-attention weights are **not** explanations; proper explanation methods (GNNExplainer/PGExplainer, SHAP, counterfactuals) come in Milestone 4."""),
        code(SETUP),
        md("## 0 · Configuration"),
        code("""import json, warnings
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from IPython.display import display
warnings.filterwarnings("ignore")

from src.config import TrainConfig, WindowConfig, FIGURE_DIR, METRIC_DIR, SYNTHETIC_CAVEAT
from src.experiments import (prepare_m2_data, make_splits, run_baselines, run_tgat, merge_results,
                             comparison_table, best_run)
from src.temporal_gnn import GNNConfig, build_graph, graph_summary
from src.graph_builder import EDGE_FEATURES
from src.baselines import get_device
from src import visualization as viz

FAST_MODE = False           # True = 1 seed per model
INCLUDE_BASELINES = True    # retrain MLP + TCN on the same splits for a like-for-like comparison
SEEDS = (0,) if FAST_MODE else (0, 1, 2)
GCFG = GNNConfig(seeds=SEEDS)
DEVICE = get_device()
for p in (FIGURE_DIR, METRIC_DIR):
    p.mkdir(parents=True, exist_ok=True)
print("device:", DEVICE, "| seeds:", SEEDS, "| GPU available:", torch.cuda.is_available())
print(SYNTHETIC_CAVEAT)"""),
        md("## 1 · Data (same 60-day simulation and windows as notebook 03; cached after the first build, ~3 min)"),
        code("""d = prepare_m2_data()
lab = d["labels"]
display(lab.context.value_counts().rename("windows").to_frame())"""),
        md("## 2 · Build the temporal interaction graph"),
        code("""G = build_graph(d, WindowConfig(), lags=GCFG.temporal_lags)
display(graph_summary(G).rename("count").to_frame())
print("edge features:", ", ".join(EDGE_FEATURES), "+ 8-d learned time encoding")"""),
        md("## 3 · Splits (identical to notebook 03)"),
        code("""splits = make_splits(d)
for name, sp in splits.items():
    display(pd.crosstab(pd.Series(sp, name=name), lab.y.rename("synthetic_demo_suspicious")))"""),
        md("""## 4 · Train the temporal GAT and the no-graph ablation

- **Training setup:** full-graph training, **inductive**. While fitting, only edges between training nodes are used; validation and test nodes are scored on the full graph without their labels.
- **Early stopping and threshold:** both use validation PR-AUC for the investigation score.
- **Loss:** investigation + fishing + gap-anomaly + interaction-edge BCE, plus 0.3 × the trajectory Gaussian NLL.
- **Speed:** GPU is much faster; CPU works."""),
        code("""res_g = run_tgat(d, G, splits, GCFG, device=DEVICE)
res_g["runs"].to_csv(METRIC_DIR / "m3_tgat_runs.csv", index=False)
res_g["tasks"].to_csv(METRIC_DIR / "m3_task_metrics.csv", index=False)"""),
        md("## 5 · Baselines on the same splits (MLP and TCN from Milestone 2)"),
        code("""if INCLUDE_BASELINES:
    res_b = run_baselines(d, splits, TrainConfig(seeds=SEEDS), models=("MLP", "TCN"), device=DEVICE)
    res = merge_results(res_b, res_g)
else:
    res = merge_results(res_g)
table = comparison_table(res["runs"])
table.to_csv(METRIC_DIR / "m3_model_comparison.csv")
display(table)
fig = viz.plot_model_comparison(res["runs"], "pr_auc", FIGURE_DIR / "m3_model_comparison_pr_auc.png"); plt.show()
fig = viz.plot_model_comparison(res["runs"], "f1", FIGURE_DIR / "m3_model_comparison_f1.png", label="Test F1"); plt.show()
fig = viz.plot_pr_curves(res, lab, FIGURE_DIR / "m3_pr_curves.png"); plt.show()"""),
        md("## 6 · Multi-task losses (each term reported separately)"),
        code("""for sp in splits:
    fig = viz.plot_task_losses(res_g["histories"], sp, path=FIGURE_DIR / f"m3_task_losses_{sp}.png"); plt.show()"""),
        md("""## 7 · Auxiliary heads: fishing, gap anomaly, interaction edges, trajectory

- **Fishing, gap-anomaly and interaction-edge heads:** scored by test PR-AUC.
- **Trajectory decoder:** scored on test gaps against the hidden simulated path, which is never used for training.
  - ADE: mean distance over the 8 predicted points.
  - FDE: distance at reappearance.
  - Coverage: share of reappearances inside the predicted 90% region.
- **References:** dead reckoning (last speed and course held) and "stay at the last fix"."""),
        code("""tasks = res_g["tasks"]
cols = ["fishing_pr_auc", "gap_anomaly_pr_auc", "interaction_edge_pr_auc", "traj_ade_km", "traj_fde_km",
        "traj_median_fde_km", "traj_endpoint_90pct_coverage", "traj_n"]
display(tasks.groupby(["split", "model"], sort=False)[[c for c in cols if c in tasks]].mean().round(3))"""),
        code("""# one planted blackout from the test vessels: what the decoder predicts vs the hidden truth
gd = G["gap_ds"]
sp = splits["vessel_disjoint"]
en = gd["end_node"]
cand = np.flatnonzero((en >= 0) & (sp[np.clip(en, 0, None)] == "test") & (G["y"][np.clip(en, 0, None)] == 1)
                      & ~np.isnan(gd["truth_path"]).any(axis=(1, 2)))
best = best_run(res, "vessel_disjoint", "TGAT (temporal graph)")["raw"]
if len(cand):
    k = cand[np.argmax(gd["gaps"].gap_duration_min.values[cand])]
    fig = viz.plot_gap_reconstruction(d, G, best, k, FIGURE_DIR / "m3_gap_reconstruction_example.png",
                                      title="Planted synthetic blackout (test vessel) - neural gap reconstruction")
    plt.show()"""),
        md("## 8 · Does the graph help? Hard negatives and event types"),
        code("""from src.evaluation import binary_metrics
rows = []
for sp_name in splits:
    t = viz.fp_by_context(res, lab, sp_name)
    t.to_csv(METRIC_DIR / f"m3_false_alerts_by_context_{sp_name}.csv", index=False)
    fig = viz.plot_fp_by_context(t, sp_name, FIGURE_DIR / f"m3_false_alerts_by_context_{sp_name}.png"); plt.show()
    display(t.pivot(index="context", columns="model", values="fp_rate").round(3))

ev_rows = []
for (sp_name, m, seed), p in res["preds"].items():
    for et, g in p["events"].groupby("event_type"):
        ev_rows.append({"split": sp_name, "model": m, "event_type": et, "recall": g.detected.mean()})
ev_tab = pd.DataFrame(ev_rows).groupby(["split", "event_type", "model"]).recall.mean().unstack("model").round(2)
ev_tab.to_csv(METRIC_DIR / "m3_event_recall_by_type.csv")
display(ev_tab)"""),
        code("""fig = viz.plot_fp_gallery(d, best_run(res, "vessel_disjoint", "TGAT (temporal graph)"), "vessel_disjoint",
                          "TGAT (temporal graph)", FIGURE_DIR / "m3_false_alert_gallery_TGAT.png")
if fig is not None:
    plt.show()"""),
        md("## 9 · Save summary"),
        code("""summary = {
    "milestone": 3, "data_mode": "A_synthetic (60 days)", "caveat": SYNTHETIC_CAVEAT, "device": str(DEVICE),
    "graph": graph_summary(G).to_dict(), "gnn_config": vars(GCFG),
    "comparison_mean": res["runs"].groupby(["split", "model"])[["pr_auc", "f1", "precision", "recall",
        "false_alerts_per_1000_vessel_h", "event_recall"]].mean().round(4).reset_index().to_dict("records"),
    "task_metrics_mean": tasks.groupby(["split", "model"]).mean(numeric_only=True).round(4).reset_index().to_dict("records"),
}
(METRIC_DIR / "m3_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print("metrics:", *sorted(p.name for p in METRIC_DIR.glob("m3_*")), sep="\\n  ")
print("figures:", *sorted(p.name for p in FIGURE_DIR.glob("m3_*")), sep="\\n  ")"""),
        md("""## How to read the results, and what Milestone 3 can and cannot conclude

**Questions to answer from the tables above**

1. **Does the graph add information?** Compare *TGAT (temporal graph)* with *TGAT without graph (ablation)*. They are the same model trained the same way, and only the edges differ. A higher PR-AUC and fewer false alerts, especially on *carrier loitering* and *benign_gap* windows and on *low_speed_rendezvous* event recall, would mean neighbour messages help. If the two are tied, the graph adds nothing yet on this data, and that should be reported honestly.
2. **Is it better than the strongest non-graph baseline (MLP)?** The MLP is strong on this synthetic data because its hand-built gap and encounter aggregates closely match how suspicious behaviour was simulated.
3. **Does the trajectory decoder beat dead reckoning?** Compare ADE and FDE. A 90% coverage close to 0.90 means the uncertainty regions are well calibrated.

**Can conclude:** whether, on this synthetic pilot, the temporal interaction graph and the multi-task heads improve the investigation score over non-graph models, with all runs using leakage-safe splits.

**Cannot conclude:**
- anything about real vessels or real illegal fishing;
- anything from raw attention weights, which are not explanations;
- performance on real AIS, or transfer to other Indian regions.

**Next, Milestone 4:** the explanation layer. It covers GNNExplainer/PGExplainer for important nodes and edges, SHAP for features, temporal masking, counterfactual tests, and fidelity/sparsity/stability checks. It also produces structured JSON alerts with a caveat."""),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}, "colab": {"provenance": []}}
    return nb


def nb02():
    cells = [
        md("""# 02 · REAL data for Indian waters — Global Fishing Watch — 5 YEARS (2021–2025)

This notebook downloads **real** data for the Gulf of Mannar – Palk Bay pilot (`IN-GOM`) from the Global Fishing Watch (GFW) API and puts it into the project pipeline. No synthetic data is used anywhere here.

| What | Source (real) |
|---|---|
| Apparent fishing events, encounters (vessel meetings), loitering, port visits, **AIS-off gaps** (with off/on positions and satellite-reception quality) | GFW Events API, derived from real AIS |
| Vessel detections from **Sentinel-1 radar** that match **no AIS vessel** (candidate "dark" vessels) | GFW 4Wings SAR presence |
| Apparent fishing hours per vessel | GFW 4Wings fishing effort |

**Important limits**
- **No raw ping-by-ping AIS tracks:** GFW's public API gives processed events instead.
- **Incomplete coverage:** many small Indian fishing boats carry no AIS, so real coverage is partial.
- **No labels:** there are **no public "confirmed illegal fishing" labels** for these waters. Everything here describes *activity patterns and investigation evidence*, never illegality.

**Terms:** non-commercial use only; attribute *Global Fishing Watch*. Vessel identities are anonymised in every output.

**Runtime:** use a CPU runtime. The download for the default wide area (southern Indian EEZ, 2021–2025, 5 years) mostly waits on the API. Months already cached from an earlier shorter run (for example 2024–2025) are reused, so only the new months download. Months download 4 at a time and each finished month is cached, so if the session stops, **Run all** resumes where it left off. Downloads are cached month by month, so a re-run resumes where it stopped."""),
        code(SETUP),
        md("""## 0 · Your API token (kept private)

1. In Colab, click the **key icon 🔑** in the left sidebar (*Secrets*).
2. Click **Add new secret**. Name: `GFW_API_TOKEN`. Value: paste your token.
3. Switch **Notebook access** on.

The token is never printed or saved to any file. If no secret is found, the cell asks for it in a hidden input box."""),
        code("""from src.gfw_real import get_token, make_client
client = make_client(get_token())
print("GFW client ready (token loaded, not displayed).")"""),
        md("## 1 · Settings"),
        code("""import json, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
warnings.filterwarnings("ignore")
from src.config import REGIONS, PROCESSED_DIR, FIGURE_DIR, METRIC_DIR
from src.gfw_real import (GFW_DIR, EVENT_DATASETS, REAL_CAVEAT, bbox_polygon, fetch_events, fetch_sar,
                          fetch_fishing_effort, fetch_region_names, anonymize_real, sar_near_gaps,
                          vessel_week_tables, write_provenance)

# Wider real-data area: southern Indian EEZ (Kerala, Lakshadweep Sea, Tamil Nadu, Gulf of Mannar - Palk Bay).
# For the small pilot box only, use: REGION_ID = "IN-GOM"; BBOX = REGIONS["IN-GOM"].bbox
REGION_ID = "IN-SOUTH"
REGION_NAME = "Southern Indian EEZ (Kerala, Lakshadweep Sea, Tamil Nadu, Gulf of Mannar)"
BBOX = (5.0, 16.0, 70.0, 82.0)       # (lat_min, lat_max, lon_min, lon_max)
START, END = "2021-01-01", "2025-12-31"   # 5 years (was 2024-2025); GFW data run up to ~3 days ago
# Order matters: rare, important events first; bulky fishing / port-visit events last.
KINDS = ["gap", "encounter", "loitering", "fishing", "port_visit"]
GEOM = bbox_polygon(BBOX)
REAL_DIR = PROCESSED_DIR / "real_gfw"
# In Colab, keep the month-by-month download cache on Google Drive, so months already
# downloaded in an earlier run (e.g. 2024-2025) are reused and only new months download.
DRIVE_DATA = pathlib.Path("/content/drive/MyDrive/india-fishing-gnn-data")
CACHE = GFW_DIR
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    CACHE = DRIVE_DATA / "gfw_raw_cache"
for p in (CACHE, REAL_DIR, FIGURE_DIR, METRIC_DIR):
    p.mkdir(parents=True, exist_ok=True)
_cached = sorted(CACHE.glob("events_fishing_*.parquet"))
print(f"download cache: {CACHE} ({len(_cached)} fishing-event months already cached)")
print(REAL_CAVEAT)
print("region:", REGION_NAME, BBOX, "| period:", START, "->", END)"""),
        md("## 2 · Download real events (cached per month)"),
        code("""raw_events, failures = {}, {}
for kind in KINDS:
    print(f"Fetching {kind} events ...")
    try:
        raw_events[kind] = await fetch_events(client, kind, GEOM, START, END, cache_dir=CACHE)
    except Exception as ex:
        failures[kind] = f"{type(ex).__name__}: {ex}"
        print("   FAILED:", failures[kind])
display(pd.Series({k: len(v) for k, v in raw_events.items()}, name="events").to_frame())
if failures:
    print("Some requests failed (see above). Re-run this cell later; finished months are cached.")"""),
        md("## 3 · Download real Sentinel-1 radar detections (unmatched to AIS) and fishing effort"),
        code("""try:
    sar = await fetch_sar(client, GEOM, START, END, unmatched_only=True, cache_dir=CACHE)
except Exception as ex:
    print("SAR download failed:", ex); sar = pd.DataFrame(columns=["date", "lat", "lon", "detections"])
try:
    effort = await fetch_fishing_effort(client, GEOM, START, END, cache_dir=CACHE)
except Exception as ex:
    print("Fishing-effort download failed:", ex); effort = pd.DataFrame()
print(f"SAR cell-days without AIS match: {len(sar):,} (total detections {int(sar.detections.sum()) if len(sar) else 0:,})")
print(f"fishing-effort rows: {len(effort):,}")"""),
        md("## 4 · Combine, interpret regions, anonymise, save"),
        code("""names = await fetch_region_names(client)
ev_raw = pd.concat([v for v in raw_events.values() if len(v)], ignore_index=True)
# keep events that overlap the study period (long port visits can start years earlier)
ev_raw = ev_raw[ev_raw.end.fillna(ev_raw.start) >= pd.Timestamp(START, tz="UTC")].reset_index(drop=True)
ev, id_map = anonymize_real(ev_raw)
id_map.to_parquet(REAL_DIR / "_private_id_map.parquet")          # private, git-ignored
eez_names = names.get("eez", {})
india_eez = [k for k, v in eez_names.items() if "india" in v.lower() and "indian ocean" not in v.lower()]
print("EEZ ids treated as Indian:", {k: eez_names[k] for k in india_eez} or "none found (lookup unavailable)")
ev["eez_names"] = ev.eez_ids.map(lambda s: ", ".join(eez_names.get(x, x) for x in s.split(",") if x))
ev["mpa_names"] = ev.mpa_ids.map(lambda s: ", ".join(names.get("mpa", {}).get(x, x) for x in s.split(",") if x))
gi = ev.event_type == "gap"
ev.loc[gi, "sar_dark_nearby"] = sar_near_gaps(ev[gi], sar, radius_km=20).values
ev.to_parquet(REAL_DIR / "events.parquet", index=False)
sar.to_parquet(REAL_DIR / "sar_unmatched.parquet", index=False)
if len(effort):
    effort.drop(columns=[c for c in ("ship_name", "mmsi", "imo", "call_sign") if c in effort]).to_parquet(
        REAL_DIR / "fishing_effort.parquet", index=False)
prov = write_provenance(REAL_DIR / "provenance.json", REGION_NAME, BBOX, START, END,
                        {**{k: len(v) for k, v in raw_events.items()}, "sar_cell_days": len(sar), "effort_rows": len(effort)})
print(json.dumps(prov, indent=2)[:1200])"""),
        md("## 5 · What is in the real data?"),
        code("""print(f"vessels: {ev.vessel_id.nunique():,} | events: {len(ev):,} | {ev.start.min():%Y-%m-%d} -> {ev.start.max():%Y-%m-%d}")
display(pd.crosstab(ev.start.dt.tz_convert(None).dt.to_period("M"), ev.event_type))
display(ev.groupby("event_type").vessel_type.value_counts().unstack(fill_value=0))
display(ev.flag.value_counts().head(10).rename("events by flag").to_frame())
g = ev[ev.event_type == "gap"]
if len(g):
    display(pd.Series({"AIS-off gaps": len(g),
                       "flagged intentional by GFW": int(g.intentional_disabling.fillna(False).sum()),
                       "median gap (h)": g.gap_duration_h.median(),
                       "gaps with unmatched SAR detections within 20 km": int((g.sar_dark_nearby > 0).sum()),
                       "gaps starting or ending inside an MPA": int(g.in_mpa.sum())}, name="value").to_frame())
e = ev[ev.event_type == "encounter"]
if len(e):
    display(e.encounter_type.value_counts().rename("encounters").to_frame())
display(ev[ev.in_mpa].groupby(["mpa_names", "event_type"]).size().rename("events inside MPAs").to_frame().head(20))"""),
        code("""fig, ax = plt.subplots(figsize=(11, 3.6))
m = pd.crosstab(ev.start.dt.tz_convert(None).dt.to_period("M"), ev.event_type)
m.plot(ax=ax, marker="o", lw=1.6)
ax.set_ylabel("events per month"); ax.set_xlabel(""); ax.set_title("Real GFW events per month - pilot region", loc="left", fontsize=10)
ax.grid(color="#e9e8e4", lw=0.5); plt.tight_layout(); plt.savefig(FIGURE_DIR / "real_events_per_month.png", dpi=150); plt.show()"""),
        md("## 6 · Maps of the real data"),
        code("""from src.india_layers import get_layers
from src import visualization as viz
layers = get_layers("IN-GOM")
# static map zooms to the Gulf of Mannar pilot; the interactive map below shows the whole downloaded area
fig = viz.plot_real_events_map(ev, sar, layers, f"Real GFW events and SAR detections - Gulf of Mannar / Palk Bay zoom ({START} to {END})",
                               FIGURE_DIR / "real_events_map.png")
plt.show()
m = viz.folium_real_map(ev, sar, layers, FIGURE_DIR / "real_events_map.html")
m"""),
        md("""## 7 · Real vessel-week interaction graph (input for the graph model)

- **Nodes:** vessel × week, with features aggregated from real events:
  - fishing hours, split into inside MPAs and outside Indian waters;
  - AIS-off gaps: counts, hours, GFW's intentional-disabling flag and satellite reception;
  - unmatched SAR detections near gaps;
  - loitering, encounters (with carriers counted separately) and port visits.
- **Edges:** real encounters between vessels, plus each vessel's previous active week."""),
        code("""nodes, edges = vessel_week_tables(ev, india_eez_ids=india_eez)
nodes.to_parquet(REAL_DIR / "graph_nodes_vessel_week.parquet", index=False)
edges.to_parquet(REAL_DIR / "graph_edges.parquet", index=False)
display(pd.Series({"nodes (vessel-weeks)": len(nodes), "vessels": nodes.vessel_id.nunique(),
                   "encounter edges (directed)": int((edges.edge_type == "encounter").sum()),
                   "temporal edges": int((edges.edge_type == "temporal").sum())}, name="count").to_frame())
cols = ["fishing_hours", "fishing_in_mpa_hours", "fishing_outside_india_eez_hours", "n_gaps", "gap_hours",
        "n_intentional_gaps", "sar_dark_near_gaps", "loiter_hours", "n_encounters", "n_carrier_encounters", "n_port_visits"]
display(nodes[cols].describe().T.round(2))
summary = {"region": REGION_ID, "start": START, "end": END, "n_events": len(ev), "n_vessels": int(ev.vessel_id.nunique()),
           "events_by_type": ev.event_type.value_counts().to_dict(), "sar_unmatched_cell_days": len(sar),
           "graph_nodes": len(nodes), "graph_edges": edges.edge_type.value_counts().to_dict(), "caveat": REAL_CAVEAT}
(METRIC_DIR / "real_data_summary.json").write_text(json.dumps(summary, indent=2, default=str))
print("saved to", REAL_DIR)
if IN_COLAB:   # copy to Google Drive, replacing the older copy, so notebooks 04b and 05 use this period
    import shutil
    DRIVE_REAL = DRIVE_DATA / "real_gfw"
    DRIVE_REAL.mkdir(parents=True, exist_ok=True)
    for f in REAL_DIR.iterdir():
        if f.is_file():
            shutil.copy2(f, DRIVE_REAL / f.name)
    print("copied to Google Drive:", DRIVE_REAL, "| period", START, "->", END)"""),
        md("""## What this real data can and cannot tell us

**Can:**
- show real fishing activity, vessel meetings, loitering and AIS-off gaps in the Gulf of Mannar and Palk Bay, where they happen relative to protected areas and the maritime boundary, and how they change over time;
- show independent **radar evidence** of unidentified vessels near AIS gaps;
- build a real interaction graph for the model.

**Cannot:**
- prove any vessel acted illegally, since gaps have many innocent causes (coverage, equipment, port stays);
- see vessels that carry no AIS, except through radar;
- provide confirmed labels: none exist publicly for these waters.

**Next:** train the temporal graph model on this real graph. Without confirmed labels, it will be trained to **rank vessel-weeks for investigation** using self-supervised learning plus weak real evidence, and it will be validated against independent evidence (radar detections, GFW's intentional-disabling flag) that is held out from its inputs."""),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}, "colab": {"provenance": []}}
    return nb


def nb04b():
    cells = [
        md("""# 04b · Temporal graph model on REAL data (Global Fishing Watch, southern Indian EEZ) — 5 YEARS (2021–2025)

This notebook trains the temporal graph-attention model on the **real** data downloaded in notebook 02 (2021–2025, 5 years). It loads the data from your Google Drive (`MyDrive/india-fishing-gnn-data/real_gfw`), so nothing is downloaded again.

**Why "forecast next week"?** There are no public labels of confirmed illegal fishing for Indian waters. Training on invented labels would be dishonest, so the model learns a task real data *can* check:

> From everything seen up to week *t*, how likely is each vessel, in week *t + 1*, to have an AIS-off gap, an encounter (with any vessel, or with a carrier), loitering, fishing inside a marine protected area, or fishing inside an EEZ that is not its own flag state's (for example an Indian boat in Sri Lankan waters, or a foreign boat in Indian waters)?

The **investigation priority** is a fixed, documented combination of these calibrated forecasts. It is an analyst's weighting, not a learned "illegality" label.

**Held-out evidence.** None of the following is ever given to the model as input or target. The notebook then checks whether high-priority vessel-weeks are followed by more of it:
- GFW's own *possible intentional AIS disabling* flag on the next week's gaps;
- **Sentinel-1 radar** detections of vessels that match no AIS, within 20 km of the next week's gaps;
- **Sentinel-1 radar** detections of vessels that match no AIS, within 5 km of where the vessel fished next week, on the same day (unidentified boats operating alongside it).

**The graph**
- **Nodes:** vessel × week. Only fishing-related vessels are kept (fishing, carrier, support and bunker vessels, plus any vessel that met one of them); pure merchant traffic is dropped.
- **Edges:**
  - each vessel's previous 1–3 active weeks (past → present only);
  - real encounters in the same week;
  - co-location: vessels active in the same 0.25° cell in the same week.

**Comparisons**
- persistence ("next week looks like this week");
- gradient-boosted trees;
- the same network without a graph;
- the network with temporal edges only;
- the full graph.

**Splits**
- **Time split (main):** train on the past (2021 to Mar 2025), validate on Apr–Jun 2025, test on Jul–Dec 2025 (the validation and test windows are unchanged from the 2-year run, so scores stay comparable), with a one-week embargo at each boundary.
- **Vessel split:** test vessels are never seen in training.

**Runtime:** **use a GPU** (Runtime → Change runtime type → T4 GPU). The full run takes about 20–40 min; `FAST_MODE = True` takes about 5–10 min. A CPU run is possible with `FAST_MODE` but is slow (about 1 h or more).

> ⚠️ Real data, real limits. Many small Indian boats have no AIS at all. Scores rank **what to look at first**, never who is guilty. Vessel identities stay anonymised."""),
        code(SETUP),
        md("## 0 · Mount Google Drive and find the saved real data"),
        code("""from pathlib import Path
DRIVE_DATA = Path("/content/drive/MyDrive/india-fishing-gnn-data")
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
from src.real_model import find_real_dir
REAL_DIR = find_real_dir()
OUT = (DRIVE_DATA / "outputs_04b") if IN_COLAB else (PROJECT_DIR / "outputs" / "real_04b")
(OUT / "figures").mkdir(parents=True, exist_ok=True)
print("real data:", REAL_DIR)
print("results will also be saved to:", OUT)
!ls -la {REAL_DIR}"""),
        md("## 1 · Settings"),
        code("""import json, warnings, time
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from IPython.display import display
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)
from src import real_model as rm
from src import visualization as viz
from src.config import FIGURE_DIR, METRIC_DIR

FAST_MODE = False            # True = 1 seed, time split only (about 5-10 min on a T4 GPU)
RUN_VESSEL_SPLIT = not FAST_MODE
DOWNLOAD_SAR_FOOTPRINT = True   # one-time (~10 min, needs GFW_API_TOKEN secret); cached on Drive afterwards
PROV = json.loads((REAL_DIR / "provenance.json").read_text()) if (REAL_DIR / "provenance.json").exists() else {}
START, END = PROV.get("start", "2021-01-01"), PROV.get("end", "2025-12-31")
BBOX = tuple(PROV.get("bbox_lat_lat_lon_lon", (5.0, 16.0, 70.0, 82.0)))
CFG = rm.RealConfig(seeds=(0,) if FAST_MODE else (0, 1, 2), study_start=START)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for p in (FIGURE_DIR, METRIC_DIR):
    p.mkdir(parents=True, exist_ok=True)
FIG = OUT / "figures"
print("device:", DEVICE, "| seeds:", CFG.seeds)
if DEVICE.type == "cpu":
    print("No GPU found. Runtime -> Change runtime type -> T4 GPU is strongly recommended.")
print(rm.REAL_MODEL_CAVEAT)"""),
        md("""## 2a · Radar imaging footprint (one-time download)

Sentinel-1 does not photograph all of the open ocean every day. If there's no radar detection of an unidentified boat next to a vessel, that only means something where the satellite actually took a picture that day.

- **Imaging footprint:** this cell downloads radar detections of **all** vessels, including those matched to AIS. Any detection within 25 km on the same day proves the area was imaged.
- **Where it's used:** the radar evidence check in section 9 is evaluated only in those imaged places.
- **What it needs:** the `GFW_API_TOKEN` secret, as in notebook 02. The first run takes longer for 5 years (2024–2025 months are reused from the Drive cache; 2021–2023 download, roughly 15–20 min); after that the footprint is read from Drive.
- **Skipping it:** set `DOWNLOAD_SAR_FOOTPRINT = False`. The radar check is then reported without a coverage correction and is not reliable."""),
        code("""SAR_ALL_FILE = REAL_DIR / f"sar_all_vessels_{START}_{END}.parquet"   # one file per study period
sar_all = None
if SAR_ALL_FILE.exists():
    sar_all = pd.read_parquet(SAR_ALL_FILE)
elif DOWNLOAD_SAR_FOOTPRINT:
    try:
        from src.gfw_real import get_token, make_client, fetch_sar, bbox_polygon
        client = make_client(get_token())
        cache = (DRIVE_DATA / "gfw_raw_cache") if IN_COLAB else None
        kw = {"cache_dir": cache} if cache else {}
        sar_all = await fetch_sar(client, bbox_polygon(BBOX), START, END, unmatched_only=False, **kw)
        sar_all.to_parquet(SAR_ALL_FILE, index=False)
    except Exception as ex:
        print("Footprint download failed - continuing without it:", type(ex).__name__, ex)
print("radar footprint:", f"{len(sar_all):,} cell-days" if sar_all is not None else "not available")"""),
        md("""## 2 · Load and clean the real events

**Cleaning rules**
- A few GFW gap records span years (the raw maximum was about 48,000 h). These mean the vessel left the area or stopped transmitting for good, so gaps longer than **30 days** are not counted as gaps. They are kept only as a separate count.
- Single-event durations are capped at one week.
- Every encounter is given a row for **both** vessels.
- Events that started before the study period are dropped. These are a few very old port visits; without this step the weekly chart would start in 2014."""),
        code("""ev_raw, sar, _ = rm.load_real(REAL_DIR)
print(f"events loaded: {len(ev_raw):,} | SAR cell-days (no AIS match): {len(sar):,} | "
      f"SAR cell-days (all vessels, imaging footprint): {len(sar_all) if sar_all is not None else 'not available'}")
ev, rep = rm.clean_events(ev_raw, CFG)
ev = rm.add_sar_evidence(ev, sar, CFG.sar_radius_km, sar_all=sar_all)   # held-out evidence only
fe = ev.event_type.eq("fishing")
rep["fishing events in radar-imaged areas (same day)"] = int((fe & ev.sar_imaged).sum())
rep["...of which with an unmatched SAR detection within 5 km"] = int((fe & ev.sar_imaged & (ev.sar_near_event > 0)).sum())
display(pd.Series(rep, name="value").to_frame())
f = ev[ev.event_type.eq("fishing")]
print("Fishing events by flag: share inside an EEZ that is not the flag state's (sanity check)")
display(f.assign(flag=f.flag.fillna("unknown")).groupby("flag").agg(
    fishing_events=("event_id", "size"), share_outside_flag_eez=("outside_flag_eez", "mean"))
    .sort_values("fishing_events", ascending=False).head(8).round(3))
g = ev[ev.gap_valid]
display(g.gap_h.describe(percentiles=[.5, .9, .99]).round(1).rename("valid gap hours").to_frame().T)"""),
        md("## 3 · Keep fishing-related vessels (drop pure merchant traffic)"),
        code("""keep, vclass, sel = rm.select_vessels(ev, CFG)
display(sel)
ev = ev[ev.vessel_id.isin(keep)].reset_index(drop=True)
print(f"kept {len(keep):,} vessels and {len(ev):,} events")
display(ev.groupby("event_type").size().rename("events kept").to_frame())"""),
        md("## 4 · Vessel-week table: inputs, next-week targets, held-out evidence"),
        code("""nodes = rm.weekly_table(ev, vclass, CFG)
print(f"vessel-weeks: {len(nodes):,} | with a next active week: {int(nodes.has_next.sum()):,}")
tcols = [f"y_{t}" for t in rm.TASKS]; ecols = [f"e_{k}" for k in rm.EVIDENCE]
rates = nodes[nodes.has_next][tcols + ecols].mean().rename("share of vessel-weeks (next week)")
display(rates.to_frame().style.format("{:.2%}"))
fig = viz.plot_weekly_activity(nodes, FIG / "real_weekly_activity.png",
                               split_lines={"validation starts": CFG.val_start, "test starts": CFG.test_start})
plt.show()"""),
        md("## 5 · Build the real temporal interaction graph"),
        code("""t0 = time.time()
G = rm.build_real_graph(nodes, ev, CFG)
display(rm.graph_summary(G).rename("count").to_frame())
_E = G["edge_index"].shape[1]
_gb = _E * CFG.hidden * 4 * 4 * CFG.layers / 1e9   # rough: ~4 edge-sized float tensors per GAT layer kept for backprop
print(f"{G['n']:,} nodes, {_E:,} edges -> roughly {_gb:.0f} GB of GPU memory for training (a T4 has 15 GB).")
if DEVICE.type == "cuda":
    _tot = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {torch.cuda.get_device_name(0)}, {_tot:.0f} GB")
    if _gb > 0.8 * _tot:
        print("WARNING: likely out of memory. Lower CFG.max_total_edges (e.g. 3_000_000), CFG.temporal_lags to 2, or CFG.hidden to 32, then re-run this cell.")
print(f"built in {time.time() - t0:.0f}s | edge features: {rm.EDGE_COLS} + 8-d learned time encoding (lag in weeks)")"""),
        md("## 6 · Leakage-safe splits"),
        code("""splits = {"time": rm.time_split(nodes, CFG)}
if RUN_VESSEL_SPLIT:
    splits["vessel"] = rm.vessel_split(nodes)
for name, (sp, gtrain) in splits.items():
    t = pd.DataFrame({"split": sp, "has_next": nodes.has_next})
    for c in tcols:
        t[c] = nodes[c] * nodes.has_next
    tab = t.groupby("split").agg(nodes=("has_next", "size"), with_target=("has_next", "sum"),
                                 **{c.replace("y_", "pos_"): (c, "sum") for c in tcols})
    print(f"--- {name} split")
    display(tab.astype(int))
    if name == "vessel":
        display(rm.cross_split_edges(G, sp).rename("count").to_frame())"""),
        md("""## 7 · Train and compare

- Every neural model is trained **inductively**: while fitting, it sees only the training part of the graph.
- Early stopping uses the validation macro PR-AUC.
- Forecasts are calibrated on validation (Platt scaling) before they are combined into the priority."""),
        code("""t0 = time.time()
res = rm.run_real_experiment(G, splits, CFG, DEVICE)
print(f"total training time: {(time.time() - t0) / 60:.1f} min")
runs, tsum, esum = rm.summarize(res)
for df, name in ((res["runs"], "runs"), (res["tasks"], "task_metrics"), (res["evidence"], "evidence_metrics")):
    df.to_csv(OUT / f"real_{name}.csv", index=False); df.to_csv(METRIC_DIR / f"real_{name}.csv", index=False)
display(runs.round(4))"""),
        md("## 8 · Forecast quality per behaviour (test set)"),
        code("""for sp in splits:
    display(tsum[tsum.split == sp].pivot(index="task", columns="model", values="pr_auc").round(3)
            .assign(base_rate=tsum[tsum.split == sp].groupby("task").base_rate.first().round(3)))
    fig = viz.plot_real_task_scores(tsum, sp, FIG / f"real_task_pr_auc_{sp}.png"); plt.show()
    fig = viz.plot_real_training(res["histories"], sp, FIG / f"real_training_{sp}.png"); plt.show()"""),
        md("""## 9 · Does the priority point at held-out evidence?

- **Lift in the top 5%:** how many times more common the evidence is among the top 5% of vessel-weeks than on average. A lift of 1 means no better than random.
- **Activity-only reference:** a vessel that simply has many events will also have more of everything. The *Activity only* row shows how much of the lift comes from that alone, and a useful model must beat it."""),
        code("""for sp in splits:
    e = esum[esum.split == sp]
    display(e.pivot(index="evidence", columns="model", values="lift_top5pct").round(2))
    display(e.pivot(index="evidence", columns="model", values="pr_auc").round(3)
            .assign(base_rate=e.groupby("evidence").base_rate.first().round(4)))
    fig = viz.plot_evidence_lift(esum, sp, FIG / f"real_evidence_lift_{sp}.png"); plt.show()"""),
        md("""## 10 · Does the graph help?

- The full graph is compared against the *same* network without a graph, over all test vessel-weeks.
- It is also compared on the subgroup where the graph carries information: vessel-weeks with a real encounter or co-located vessels."""),
        code("""s, d = G["edge_index"]; et = G["edge_type"]
has_social = np.zeros(G["n"], bool); has_social[d[et > 0]] = True
rows = []
for sp_name, (sp, _) in splits.items():
    for model in ("Temporal GAT (full graph)", "Temporal GAT (temporal edges only)", "Same model, no graph (MLP)",
                  "Gradient boosting (no graph)"):
        for (spn, m, seed), p in res["preds"].items():
            if spn != sp_name or m != model:
                continue
            for grp_name, grp in (("all test vessel-weeks", np.ones(G["n"], bool)), ("with encounter / co-location neighbours", has_social)):
                sub = np.where(grp, sp, "other")
                tm = rm.task_metrics(p["P"], G, sub, p["tasks"])
                rows.append({"split": sp_name, "model": model, "seed": seed, "group": grp_name, "macro_pr_auc": tm.pr_auc.mean()})
gh = pd.DataFrame(rows).groupby(["split", "group", "model"]).macro_pr_auc.agg(["mean", "std"]).round(4)
gh.to_csv(OUT / "real_graph_help.csv"); display(gh)"""),
        md("## 11 · Investigation-priority alerts (test weeks, time split)"),
        code("""best = rm.best_seed(res, "time", "Temporal GAT (full graph)")
alerts = rm.alert_table(G, best, splits["time"][0], top=50, weights=CFG.priority_weights)
rm.alerts_json(alerts, OUT / "real_alerts_top50.json",
               {"model": "Temporal GAT (full graph)", "split": "time", "period": f"test weeks from {CFG.test_start}",
                "priority_weights": CFG.priority_weights})
alerts.to_csv(OUT / "real_alerts_top50.csv", index=False)
show = ["rank", "vessel_id", "vessel_class", "week", "lat", "lon", "priority", "main_driver",
        "this_week_n_gaps", "this_week_n_enc", "next_week_observed"] + [c for c in alerts if c.startswith("heldout_")]
display(alerts[show].head(25))
print("share of top-50 whose next week showed held-out evidence:",
      alerts[[c for c in alerts if c.startswith("heldout_")]].any(axis=1).mean().round(3))"""),
        code("""from src.india_layers import get_layers
prov = json.loads((REAL_DIR / "provenance.json").read_text()) if (REAL_DIR / "provenance.json").exists() else {}
bbox = tuple(prov.get("bbox_lat_lat_lon_lon", (5.0, 16.0, 70.0, 82.0)))
test_weeks = nodes.week >= pd.Timestamp(CFG.test_start)
fg = ev[ev.gap_valid & ev.vclass.eq("fishing") & (ev.week >= pd.Timestamp(CFG.test_start))]
fg = fg.dropna(subset=["off_lat"]) if "off_lat" in fg else fg.iloc[:0]
fig = viz.plot_real_alerts_map(nodes[test_weeks], alerts, fg, bbox, layers=get_layers("IN-GOM"),
                               title=f"Top-50 investigation priorities and fishing-vessel AIS-off positions (test weeks from {CFG.test_start})",
                               path=FIG / "real_alerts_map.png")
plt.show()"""),
        md("## 12 · Save summary (to the project and to Drive)"),
        code("""import shutil
summary = {
    "notebook": "04b_real_graph_model", "data": "Global Fishing Watch (real)", "caveat": rm.REAL_MODEL_CAVEAT,
    "device": str(DEVICE), "config": {k: v for k, v in vars(CFG).items()}, "cleaning": rep,
    "graph": rm.graph_summary(G).to_dict(),
    "models_mean": runs.to_dict("records"),
    "tasks_mean": tsum.to_dict("records"),
    "evidence_mean": esum.to_dict("records"),
}
(OUT / "real_model_summary.json").write_text(json.dumps(summary, indent=2, default=str))
shutil.copy(OUT / "real_model_summary.json", METRIC_DIR / "real_model_summary.json")
for f in FIG.glob("real_*.png"):
    shutil.copy(f, FIGURE_DIR / f.name)
print("saved:", *sorted(p.name for p in OUT.glob("*.*")), sep="\\n  ")"""),
        md("""## How to read the results

**Questions the tables answer**
1. **Can real next-week behaviour be forecast at all?** Compare each model's PR-AUC with the base rate (the black tick). Persistence is the bar to beat: many vessels simply repeat what they did last week.
2. **Does the graph add anything?** Compare *Temporal GAT (full graph)* with the *no graph* and *temporal edges only* versions, overall and on vessel-weeks with neighbours (section 10). If they tie, the honest conclusion is that the encounter and co-location structure adds little beyond each vessel's own history on this data.
3. **Does the priority point at independent evidence?** Section 9 answers this. A lift above the *Activity only* reference means the model finds more than just the busiest vessels.

**Can conclude:** how well real vessel behaviour in the southern Indian EEZ can be forecast from GFW events, whether graph structure helps, and whether the resulting priority is enriched for independent radar and GFW risk evidence.

**Cannot conclude:**
- that any vessel acted illegally: gaps, encounters and fishing in a neighbouring EEZ all have lawful explanations (coverage, transhipment licences, access agreements);
- anything about boats without AIS, except indirectly through radar;
- that attention weights explain decisions. Milestone 4 adds proper explanations (GNNExplainer, SHAP, counterfactuals) for these alerts.

Attribution: data from **Global Fishing Watch**, non-commercial use."""),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}, "colab": {"provenance": [], "gpuType": "T4"},
                   "accelerator": "GPU"}
    return nb


def nb05():
    cells = [
        md("""# 05 · Explaining the real-data alerts (Milestone 4) — 5 YEARS (2021–2025)

Notebook 04b produced a ranked list of vessel-weeks to look at first. This notebook answers **why** each one was flagged, and checks whether those explanations can be trusted.

**For each top alert**
1. **What drove the score?** *Integrated Gradients* traces the score back to the inputs. They are grouped into readable feature groups (fishing outside its flag state's waters, protected-area fishing, encounters, AIS gaps, …) and three roles: the vessel **this week**, its **own past weeks**, and **other vessels** linked to it in the graph.
2. **Did its history matter?** Each past week is removed from the graph, then all of them, and the vessel is re-scored.
3. **Did other vessels matter?** Encounter and co-location links are removed, one at a time and all together, and the vessel is re-scored. Every link is tested exactly.
4. **What would change the score?** Counterfactuals: *"had it not fished in foreign waters this week, the priority would fall from 0.54 to 0.09"*.
5. **A plain-language reason card** is saved as JSON with the caveat.

**Can the explanations be trusted?**
- **Completeness:** the attributions must add up to the score.
- **Deletion fidelity:** removing the 5 most important inputs must lower the score far more than removing 5 random ones.
- **Sparsity:** how few inputs carry most of the explanation.
- **Stability:** the same alerts are explained with 3 independently trained models, and the explanations should agree.

**Global view:** which information the graph model and gradient boosting rely on across all test weeks, measured by permutation importance.

**Runtime:** use a **T4 GPU**, about 10–15 min. The data is read from your Drive, as in 04b; no token or download is needed.

> ⚠️ An explanation describes **the model's** reasoning, not the vessel's intent. Priority is for investigation only. It is never evidence of illegality."""),
        code(SETUP),
        md("## 0 · Mount Drive and find the saved real data"),
        code("""from pathlib import Path
DRIVE_DATA = Path("/content/drive/MyDrive/india-fishing-gnn-data")
if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
from src.real_model import find_real_dir
REAL_DIR = find_real_dir()
OUT = (DRIVE_DATA / "outputs_05") if IN_COLAB else (PROJECT_DIR / "outputs" / "real_05")
(OUT / "figures").mkdir(parents=True, exist_ok=True)
FIG = OUT / "figures"
print("real data:", REAL_DIR, "| results ->", OUT)"""),
        md("## 1 · Settings"),
        code("""import json, time, warnings
import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from IPython.display import display
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 200)
from src import real_model as rm
from src import explainers as xp
from src import visualization as viz

FAST_MODE = False                 # True = 1 seed (no stability check), 10 alerts
N_ALERTS = 10 if FAST_MODE else 20
N_FIGURES = 3                     # detailed explanation figures for the top alerts
PROV = json.loads((REAL_DIR / "provenance.json").read_text()) if (REAL_DIR / "provenance.json").exists() else {}
CFG = rm.RealConfig(seeds=(0,) if FAST_MODE else (0, 1, 2), study_start=PROV.get("start", "2021-01-01"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE, "| seeds:", CFG.seeds, "| alerts to explain:", N_ALERTS)
print(rm.REAL_MODEL_CAVEAT)"""),
        md("## 2 · Rebuild the real vessel-week graph (same steps as 04b, about 1–2 min)"),
        code("""ev_raw, _, _ = rm.load_real(REAL_DIR)
ev, rep = rm.clean_events(ev_raw, CFG)
keep, vclass, _ = rm.select_vessels(ev, CFG)
ev = ev[ev.vessel_id.isin(keep)].reset_index(drop=True)
nodes = rm.weekly_table(ev, vclass, CFG)
G = rm.build_real_graph(nodes, ev, CFG)
split, gtrain = rm.time_split(nodes, CFG)
tasks = rm.active_tasks(G, split, CFG)
display(rm.graph_summary(G).rename("count").to_frame())
print("forecast tasks:", tasks)"""),
        md("## 3 · Train the full temporal graph model (one per seed) and gradient boosting"),
        code("""models = {}
for seed in CFG.seeds:
    r = rm.train_real(G, split, gtrain, CFG, seed, DEVICE, "full", tasks)
    models[seed] = r
    print(f"seed {seed}: val macro PR-AUC {r['best_val']:.4f} ({r['epochs']} epochs, {r['seconds']:.0f}s)")
best_seed = max(models, key=lambda k: models[k]["best_val"])
R = models[best_seed]
Z, stats = rm.standardize(G["X_raw"], split == "train")
Pc = rm.calibrate(R["P"], G, split, tasks)
priority = rm.investigation_priority(Pc, tasks, CFG.priority_weights)
pred = {"P": R["P"], "Pcal": Pc, "priority": priority, "tasks": tasks}
alerts = rm.alert_table(G, pred, split, top=N_ALERTS, weights=CFG.priority_weights)
display(alerts[["rank", "vessel_id", "week", "lat", "lon", "priority", "main_driver", "next_week_observed"]])

from sklearn.ensemble import HistGradientBoostingClassifier
hgb = {}
for t in tasks:
    k = list(rm.TASKS).index(t); rr = (split == "train") & G["M"][:, k]
    hgb[t] = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, early_stopping=True,
                                            validation_fraction=0.15, random_state=0).fit(G["X_raw"].values[rr], G["Y"][rr, k])
print("gradient boosting trained for", list(hgb))"""),
        md("## 4 · Explain each alert"),
        code("""def make_explainer(r):
    platt = xp.fit_platt(r["P"], G, split, tasks)
    return xp.AlertExplainer(G, r["model"], Z, stats, tasks, platt, CFG.priority_weights, DEVICE, layers=CFG.layers)

EX = make_explainer(R)
node_of = {(v, pd.Timestamp(w)): i for i, (v, w) in enumerate(zip(nodes.vessel_id, nodes.week))}
alert_nodes = [node_of[(r.vessel_id, pd.Timestamp(r.week))] for r in alerts.itertuples()]
t0 = time.time()
cards = [EX.explain(n) for n in alert_nodes]
print(f"explained {len(cards)} alerts in {time.time() - t0:.0f}s")
summary = pd.DataFrame([{
    "rank": a.rank, "vessel_id": a.vessel_id, "week": a.week, "priority": round(c["priority"], 3),
    "top driver": f"{c['attribution'].group.iloc[0]} ({c['attribution'].role.iloc[0]})",
    "history effect": round(float(c["temporal"].delta.min()) if len(c["temporal"]) else 0.0, 3),
    "other-vessel effect": round(float(c["edges"][c["edges"].removed.str.startswith(("all encounter", "all co-location"))].delta.abs().max())
                                 if c["edges"].removed.str.startswith(("all encounter", "all co-location")).any() else 0.0, 3),
    "best counterfactual": (f"{c['counterfactual'].iloc[0, 0]} -> {c['counterfactual'].priority.iloc[0]:.2f}"
                            if len(c["counterfactual"]) else "-"),
    "reason": c["reason"]} for a, c in zip(alerts.itertuples(), cards)])
summary.to_csv(OUT / "m4_alert_explanations.csv", index=False)
display(summary.drop(columns="reason"))
for r in summary.head(5).itertuples():
    print(f"#{r.rank} {r.vessel_id} ({r.week:%Y-%m-%d}): {r.reason}")"""),
        code("""for a, c in list(zip(alerts.itertuples(), cards))[:N_FIGURES]:
    fig = viz.plot_alert_explanation(c, f"Alert #{a.rank}: {a.vessel_id}, week of {a.week:%d %b %Y} (priority {c['priority']:.2f})",
                                     FIG / f"m4_alert_{a.rank:02d}.png")
    plt.show()
    display(c["temporal"].round(3)); display(c["edges"].round(3))"""),
        md("## 5 · Global view: which information do the models rely on?"),
        code("""X_raw = G["X_raw"].values.astype(np.float32)
mu, sd = np.asarray(stats["mean"]), np.asarray(stats["sd"])
g_all = {k: torch.as_tensor(v, device=DEVICE) for k, v in (("ei", G["edge_index"]), ("ea", G["edge_attr"]), ("lag", G["edge_lag"]))}

def predict_gnn(Xr):
    x = torch.as_tensor(np.clip((Xr - mu) / sd, -10, 10).astype(np.float32), device=DEVICE)
    with torch.no_grad():
        return torch.sigmoid(R["model"](x, g_all["ei"], g_all["ea"], g_all["lag"])).cpu().numpy()

def predict_hgb(Xr):
    return np.column_stack([hgb[t].predict_proba(Xr)[:, 1] for t in tasks])

test_rows = split == "test"
imp = {"Temporal GAT (full graph)": xp.group_permutation_importance(predict_gnn, G, test_rows, tasks, X_raw, repeats=3),
       "Gradient boosting": xp.group_permutation_importance(predict_hgb, G, test_rows, tasks, X_raw, repeats=3)}
for k, v in imp.items():
    v.to_csv(OUT / f"m4_global_importance_{k.split()[0].lower()}.csv", index=False)
display(pd.concat({k: v.set_index("group").importance for k, v in imp.items()}, axis=1).round(4))
fig = viz.plot_global_importance(imp, FIG / "m4_global_importance.png"); plt.show()"""),
        md("""## 6 · Can the explanations be trusted?

- **Completeness error:** how far the attributions are from adding up to (score − baseline score). It should be close to 0.
- **Deletion fidelity:** the score drop after removing the top-5 attributed inputs, compared with 5 random inputs.
- **Sparsity:** the share of inputs needed to cover 80% of the attribution. Lower means a simpler explanation.
- **Stability:** the same alerts are explained by the models trained with the other seeds; this is the Spearman correlation of the attributions."""),
        code("""other = [s for s in CFG.seeds if s != best_seed]
stab = []
if other:
    others = {s: make_explainer(models[s]) for s in other}
    for n, c in zip(alert_nodes, cards):
        tabs = [c["attribution"]]
        for s, e in others.items():
            sg = e.subgraph(n)
            tabs.append(e.attribution_table(sg, e.integrated_gradients(sg)))
        stab.append(xp.seed_stability(tabs))
else:
    stab = [np.nan] * len(cards)
quality = pd.DataFrame({"rank": alerts["rank"], "priority": [c["priority"] for c in cards],
                        "completeness_error": [c["completeness_error"] for c in cards],
                        "drop_top_k": [c["fidelity"]["drop_top_k"] for c in cards],
                        "drop_random_k": [c["fidelity"]["drop_random_k"] for c in cards],
                        "sparsity_80pct": [c["sparsity_80pct"] for c in cards], "stability": stab})
quality.to_csv(OUT / "m4_explanation_quality.csv", index=False)
display(quality.round(4))
display(quality.drop(columns="rank").agg(["mean", "median"]).round(4))
fig = viz.plot_explanation_quality(quality, FIG / "m4_explanation_quality.png"); plt.show()"""),
        md("## 7 · Save the alert cards (JSON) and summary"),
        code("""import shutil
from src.config import FIGURE_DIR, METRIC_DIR
js = [xp.card_to_json(c, nodes, a._asdict()) for a, c in zip(alerts.itertuples(index=False), cards)]
xp.save_cards(js, OUT / "m4_alert_cards.json",
              {"model": "Temporal GAT (full graph)", "seed": int(best_seed), "split": "time",
               "period": f"test weeks from {CFG.test_start}", "method": "Integrated Gradients + exact temporal/edge "
               "occlusion + counterfactuals", "priority_weights": CFG.priority_weights})
print(json.dumps(js[0], indent=2, default=str)[:2500])
m4 = {"n_alerts": len(cards), "best_seed": int(best_seed),
      "quality_mean": quality.drop(columns="rank").mean().round(4).to_dict(),
      "top_global_groups": {k: v.head(5).group.tolist() for k, v in imp.items()},
      "caveat": rm.REAL_MODEL_CAVEAT}
(OUT / "m4_summary.json").write_text(json.dumps(m4, indent=2, default=str))
for p in (FIGURE_DIR, METRIC_DIR):
    p.mkdir(parents=True, exist_ok=True)
for f in FIG.glob("m4_*.png"):
    shutil.copy(f, FIGURE_DIR / f.name)
for f in OUT.glob("m4_*.*"):
    shutil.copy(f, METRIC_DIR / f.name)
print("saved:", *sorted(p.name for p in OUT.glob("m4_*")), sep="\\n  ")"""),
        md("""## How to read this

- **Top driver: "this vessel, this week".** The alert rests on what the vessel just did, which is the most transparent kind of alert.
- **History effect.** A large drop when its past weeks are removed means the score rests on a *repeated* pattern. That is stronger grounds for a look than a single week.
- **Other-vessel effect.** The effect of encounter and co-location links. Near zero matches 04b's finding that neighbours add little on this data.
- **Counterfactual.** The single behaviour that, had it not happened, would lower the priority the most. This is the most useful line for an analyst.
- **Quality checks.** Completeness close to 0, a top-5 deletion drop far above the random drop, and stability close to 1 mean the explanation reflects what the model actually does.

**Limits**
- Explanations describe the model, not reality or intent.
- A foreign-flag vessel may be fishing legally under an access agreement or licence.
- Protected-area boundaries come from GFW's MPA layer and may include zones where fishing is allowed.
- Every alert needs human verification before any action.

Attribution: data from **Global Fishing Watch**, non-commercial use."""),
    ]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                   "language_info": {"name": "python"}, "colab": {"provenance": [], "gpuType": "T4"},
                   "accelerator": "GPU"}
    return nb


PLACEHOLDERS = {}


def placeholder(title, body):
    nb = nbf.v4.new_notebook(cells=[md(f"# {title}\n\n**Not implemented yet.** {body}\n\n"
                                       "This notebook is a placeholder from the Milestone 1 scaffold.")])
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}}
    return nb


if __name__ == "__main__":
    out = ROOT / "notebooks"
    out.mkdir(exist_ok=True)
    nbf.write(nb01(), out / "01_synthetic_demo.ipynb")
    nbf.write(nb02(), out / "02_india_real_data.ipynb")
    nbf.write(nb03(), out / "03_baselines.ipynb")
    nbf.write(nb04(), out / "04_temporal_graph_model.ipynb")
    nbf.write(nb04b(), out / "04b_real_graph_model.ipynb")
    nbf.write(nb05(), out / "05_explain_alerts.ipynb")
    for f, (t, b) in PLACEHOLDERS.items():
        if not (out / f).exists():
            nbf.write(placeholder(t, b), out / f)
    print("notebooks written to", out)
