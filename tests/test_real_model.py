"""Pipeline test for the real-data graph model, using a tiny FAKE event table shaped like GFW output
(test fixture only - the notebook uses real Global Fishing Watch data)."""
import numpy as np
import pandas as pd
import pytest
import torch

from src import real_model as rm


def fake_events(n_weeks=30, seed=0):
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2025-01-06", tz="UTC")
    vessels = [(f"vessel_f{i:02d}", "fishing") for i in range(40)] + \
              [(f"vessel_c{i}", "carrier") for i in range(4)] + [(f"vessel_m{i:02d}", "cargo") for i in range(15)]
    risky = {v for v, _ in vessels[:10]}
    rows, k = [], 0

    def add(v, t, kind, **kw):
        nonlocal k
        k += 1
        s = t0 + pd.Timedelta(weeks=w) + pd.Timedelta(hours=float(rng.uniform(0, 150)))
        base = dict(event_id=f"e{k}", event_type=kind, start=s, end=s + pd.Timedelta(hours=kw.pop("h", 4.0)),
                    lat=9 + rng.normal(0, .3), lon=79 + rng.normal(0, .3), vessel_id=v, vessel_type=t, flag="IND",
                    eez_ids="8480", eez_names="India", mpa_ids="", in_mpa=False, dist_shore_km=20.0,
                    dist_port_km=40.0, partner_id=None, partner_type="")
        base.update(kw)
        base["duration_h"] = (base["end"] - base["start"]).total_seconds() / 3600
        rows.append(base)

    for w in range(n_weeks):
        for v, t in vessels:
            if t == "cargo":
                if rng.random() < .5:
                    add(v, t, "port_visit", h=20)
                continue
            if t == "fishing":
                for _ in range(rng.poisson(3)):
                    mpa = v in risky and rng.random() < .3
                    add(v, t, "fishing", in_mpa=mpa, mpa_ids="555" if mpa else "",
                        eez_names="Sri Lanka" if v in risky and rng.random() < .2 else "India",
                        potential_risk=bool(mpa))
                if rng.random() < (.4 if v in risky else .05):
                    add(v, t, "gap", h=30, gap_duration_h=30.0, gap_distance_km=50.0, implied_speed_kn=1.0,
                        sat_positions_per_day=20.0, off_lat=9.0, off_lon=79.0, on_lat=9.2, on_lon=79.1,
                        intentional_disabling=bool(rng.random() < .5), sar_dark_nearby=float(rng.random() < .3))
                if v in risky and rng.random() < .3:
                    c = f"vessel_c{rng.integers(4)}"
                    add(v, t, "encounter", h=3, partner_id=c, partner_type="carrier", median_distance_km=0.2,
                        median_speed_kn=1.0, potential_risk=True)
            else:
                if rng.random() < .5:
                    add(v, t, "loitering", h=12, loiter_hours=12.0)
    # one absurd gap that must be excluded (years long)
    add("vessel_f00", "fishing", "gap", h=1, gap_duration_h=48000.0)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def pipeline():
    cfg = rm.RealConfig(val_start="2025-04-21", test_start="2025-06-02", epochs=4, patience=3, seeds=(0,),
                        min_train_pos=5, min_eval_pos=1, hidden=16, heads=2)
    ev, rep = rm.clean_events(fake_events(), cfg)
    keep, vc, summ = rm.select_vessels(ev, cfg)
    ev = ev[ev.vessel_id.isin(keep)]
    nodes = rm.weekly_table(ev, vc, cfg)
    G = rm.build_real_graph(nodes, ev, cfg)
    return cfg, ev, rep, keep, vc, nodes, G


def test_cleaning_and_selection(pipeline):
    cfg, ev, rep, keep, vc, nodes, G = pipeline
    assert rep["gaps_longer_than_30d (not used as gaps)"] == 1
    assert not any(v.startswith("vessel_m") for v in keep)            # pure merchant traffic dropped
    assert any(v.startswith("vessel_c") for v in keep)                # carriers kept
    # mirrored encounters: carriers have encounter rows of their own
    assert (ev[ev.event_type.eq("encounter")].vessel_id.str.startswith("vessel_c")).any()
    assert nodes.gap_h.max() <= cfg.max_gap_h * 10


def test_outside_flag_eez():
    flag = pd.Series(["LKA", "IND", "IND", "CHN", "", "LKA"])
    eez = pd.Series(["Sri Lanka", "Sri Lanka", "India", "India", "Sri Lanka", ""])
    out, known = rm.outside_flag_eez(flag, eez)
    assert list(out) == [False, True, False, True, False, False]
    assert list(known) == [True, True, True, True, False, True]


def test_sar_evidence_same_day_only():
    t = pd.Timestamp("2025-03-03 06:00", tz="UTC")
    ev = pd.DataFrame({"event_type": ["fishing", "fishing", "fishing"], "lat": [9.0, 9.0, 12.0],
                       "lon": [79.0, 79.0, 75.0], "start": [t, t + pd.Timedelta(days=2), t]})
    sar = pd.DataFrame({"date": [pd.Timestamp("2025-03-03", tz="UTC")], "lat": [9.02], "lon": [79.01], "detections": [3]})
    out = rm.add_sar_evidence(ev, sar, 5.0)
    assert list(out.sar_near_event) == [3.0, 0.0, 0.0]


def test_sar_imaged_footprint():
    t = pd.Timestamp("2025-03-03 06:00", tz="UTC")
    ev = pd.DataFrame({"event_type": ["fishing"] * 3, "lat": [9.0, 9.5, 12.0], "lon": [79.0, 79.0, 75.0],
                       "start": [t] * 3})
    d = pd.Timestamp("2025-03-03", tz="UTC")
    sar = pd.DataFrame({"date": [d], "lat": [9.01], "lon": [79.0], "detections": [1]})
    sar_all = pd.DataFrame({"date": [d, d], "lat": [9.01, 9.6], "lon": [79.0, 79.1], "detections": [1, 4]})
    out = rm.add_sar_evidence(ev, sar, 5.0, sar_all=sar_all, cover_km=25)
    assert list(out.sar_imaged) == [True, True, False]       # third spot was not imaged that day
    assert list(out.sar_near_event) == [1.0, 0.0, 0.0]


def test_targets_are_next_week(pipeline):
    cfg, ev, rep, keep, vc, nodes, G = pipeline
    nd = nodes.set_index(["vessel_id", "week"])
    r = nodes[nodes.has_next].iloc[5]
    nxt = nd.loc[(r.vessel_id, r.week + pd.Timedelta(days=7))]
    assert r.y_gap == float(nxt.n_gaps > 0)
    assert r.y_encounter == float(nxt.n_enc > 0)
    assert r.e_intentional_gap == float(nxt.ev_intentional_gap > 0)
    # held-out evidence and targets are not model inputs
    cols = set(G["X_raw"].columns)
    assert not any(c.startswith(("ev_", "e_", "y_")) for c in cols)


def test_graph_edges(pipeline):
    cfg, ev, rep, keep, vc, nodes, G = pipeline
    s, d = G["edge_index"]
    et = G["edge_type"]
    w = nodes.week.values
    assert (w[s[et == 0]] < w[d[et == 0]]).all()                      # temporal: past -> present only
    assert (nodes.vessel_id.values[s[et == 0]] == nodes.vessel_id.values[d[et == 0]]).all()
    assert (w[s[et > 0]] == w[d[et > 0]]).all()                       # encounter / co-location: same week
    assert (et == 1).sum() > 0 and (et == 2).sum() > 0
    assert G["edge_attr"].shape == (len(et), len(rm.EDGE_COLS))


def test_total_edge_budget_shrinks_coloc_edges(pipeline):
    cfg, ev, rep, keep, vc, nodes, G = pipeline
    n_other = int((G["edge_type"] < 2).sum())
    n_coloc = int((G["edge_type"] == 2).sum())
    tight = rm.RealConfig(**{**vars(cfg), "max_total_edges": n_other + n_coloc // 2, "min_coloc_edges": 10})
    G2 = rm.build_real_graph(nodes, ev, tight)
    assert (G2["edge_type"] < 2).sum() == n_other                  # temporal + encounter are never dropped
    assert 10 <= (G2["edge_type"] == 2).sum() <= n_coloc // 2 + 1  # co-location shrinks to fit the budget


def test_time_split_embargo(pipeline):
    cfg, ev, rep, keep, vc, nodes, G = pipeline
    split, gtrain = rm.time_split(nodes, cfg)
    w = nodes.week
    assert (w[split == "train"] + pd.Timedelta(days=7) < pd.Timestamp(cfg.val_start)).all()
    assert (w[split == "val"] >= pd.Timestamp(cfg.val_start)).all()
    assert (w[split == "val"] + pd.Timedelta(days=7) < pd.Timestamp(cfg.test_start)).all()
    assert (split == "embargo").any()


def test_training_and_evaluation(pipeline):
    cfg, ev, rep, keep, vc, nodes, G = pipeline
    splits = {"time": rm.time_split(nodes, cfg)}
    res = rm.run_real_experiment(G, splits, cfg, torch.device("cpu"), verbose=False)
    runs, tasks, evid = rm.summarize(res)
    assert {"Temporal GAT (full graph)", "Same model, no graph (MLP)", "Gradient boosting (no graph)"} <= set(runs.model)
    assert runs.macro_pr_auc.notna().all()
    assert set(evid.evidence) == set(rm.EVIDENCE)
    pred = rm.best_seed(res, "time", "Temporal GAT (full graph)")
    assert ((pred["priority"] >= 0) & (pred["priority"] <= 1)).all()
    al = rm.alert_table(G, pred, splits["time"][0], top=10, weights=cfg.priority_weights)
    assert len(al) == 10 and al.priority.is_monotonic_decreasing
    assert al.vessel_id.str.startswith("vessel_").all()


def test_vessel_split_disjoint(pipeline):
    cfg, ev, rep, keep, vc, nodes, G = pipeline
    split, gtrain = rm.vessel_split(nodes)
    by_v = pd.Series(split).groupby(nodes.vessel_id.values).nunique()
    assert (by_v == 1).all()
    assert set(rm.cross_split_edges(G, split).index)
