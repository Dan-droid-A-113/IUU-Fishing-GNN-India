"""Milestone 4 explainer tests on a tiny FAKE GFW-shaped table (test fixture only)."""
import json

import numpy as np
import pytest
import torch

from src import explainers as xp
from src import real_model as rm
from tests.test_real_model import fake_events


@pytest.fixture(scope="module")
def trained():
    cfg = rm.RealConfig(val_start="2025-04-21", test_start="2025-06-02", epochs=15, patience=15, seeds=(0,),
                        min_train_pos=5, min_eval_pos=1, hidden=16, heads=2, dropout=0.0)
    ev, _ = rm.clean_events(fake_events(), cfg)
    keep, vc, _ = rm.select_vessels(ev, cfg)
    ev = ev[ev.vessel_id.isin(keep)]
    nodes = rm.weekly_table(ev, vc, cfg)
    G = rm.build_real_graph(nodes, ev, cfg)
    split, gtrain = rm.time_split(nodes, cfg)
    tasks = rm.active_tasks(G, split, cfg)
    r = rm.train_real(G, split, gtrain, cfg, 0, torch.device("cpu"), "full", tasks)
    Z, _ = rm.standardize(G["X_raw"], split == "train")
    platt = xp.fit_platt(r["P"], G, split, tasks)
    ex = xp.AlertExplainer(G, r["model"], Z, r["stats"], tasks, platt, cfg.priority_weights, torch.device("cpu"))
    return cfg, G, split, tasks, r, ex


def test_subgraph_is_exact(trained):
    cfg, G, split, tasks, r, ex = trained
    Pc = rm.calibrate(r["P"], G, split, tasks)
    prio = rm.investigation_priority(Pc, tasks, cfg.priority_weights)
    for node in np.flatnonzero(split == "test")[:5]:
        s = ex.subgraph(node)
        with torch.no_grad():
            assert abs(float(ex.priority(s)) - prio[node]) < 1e-4


def test_explanation_card(trained):
    cfg, G, split, tasks, r, ex = trained
    node = int(np.flatnonzero(split == "test")[3])
    card = ex.explain(node, steps=64)
    assert card["completeness_error"] < 0.02 * max(1.0, abs(card["priority"] - card["baseline_priority"]) * 50)
    assert set(card["attribution"].role) <= set(xp.ROLES)
    assert "all past weeks" in set(card["temporal"].removed) or card["n_past_weeks"] == 0
    assert card["edges"].removed.str.contains("every edge").any()
    assert card["reason"].startswith("Flagged mainly because")
    js = xp.card_to_json(card, G["nodes"])
    json.dumps(js, default=str)
    assert "not evidence of illegality" in js["caveat"]


def test_feature_groups_cover_all_columns(trained):
    cfg, G, split, tasks, r, ex = trained
    assert "other" not in {xp.feature_group(c) for c in G["X_raw"].columns}


def test_permutation_importance_and_stability(trained):
    cfg, G, split, tasks, r, ex = trained
    Z, _ = rm.standardize(G["X_raw"], split == "train")
    X = G["X_raw"].values

    def predict(Xn):
        return np.column_stack([Xn[:, list(G["X_raw"].columns).index("n_enc")]] * len(tasks))
    imp = xp.group_permutation_importance(predict, G, split == "test", tasks, X, repeats=1)
    assert imp.iloc[0].group == "encounters"
    t = ex.attribution_table(ex.subgraph(int(np.flatnonzero(split == "test")[0])),
                             ex.integrated_gradients(ex.subgraph(int(np.flatnonzero(split == "test")[0])), 8))
    assert xp.seed_stability([t, t]) == pytest.approx(1.0)
