import numpy as np
import pandas as pd
import pytest
import torch

from src.baselines import WindowTensors
from src.config import WindowConfig
from src.graph_builder import gap_prism_pairs
from src.temporal_gnn import GapDecoder, GNNConfig, build_graph, graph_summary, train_tgat
from src.windows import attach_synthetic_labels, build_windows


@pytest.fixture(scope="module")
def dsmall(processed, layers, synth):
    ais, gaps = processed["ais"], processed["gaps"]
    prism = gap_prism_pairs(ais, gaps)
    W = build_windows(ais, gaps, layers, prism, WindowConfig())
    lab = attach_synthetic_labels(W, synth["truth_reports"], processed["id_map"])
    return {"X": W["X"], "S": W["S"], "agg": W["agg"], "labels": lab, "ais": ais, "gaps": gaps, "prism": prism,
            "events": synth["events"], "id_map": processed["id_map"], "truth": synth["truth_reports"],
            "layers": layers, "T0": W["T0"]}


@pytest.fixture(scope="module")
def graph(dsmall):
    return build_graph(dsmall, WindowConfig())


def test_graph_structure(graph, dsmall):
    s = graph_summary(graph)
    n = len(dsmall["labels"])
    ei = graph["edge_index"]
    assert s["nodes (vessel-windows)"] == n and ei.max() < n and ei.min() >= 0
    assert s["proximity edges (directed)"] > 0 and s["temporal edges"] > 0
    assert graph["edge_attr"].shape == (ei.shape[1], 10) and np.isfinite(graph["edge_attr"]).all()


def test_temporal_edges_never_point_to_the_past(graph, dsmall):
    win = dsmall["labels"].win.values
    t = graph["edge_type"] == 2
    src, dst = graph["edge_index"][:, t]
    assert (win[src] < win[dst]).all()


def test_interaction_edges_stay_within_a_window(graph, dsmall):
    win = dsmall["labels"].win.values
    t = graph["edge_type"] < 2
    src, dst = graph["edge_index"][:, t]
    assert (win[src] == win[dst]).all()


def test_rendezvous_edges_labelled(graph):
    # the small config always plants rendezvous; some should become labelled edges
    assert graph["edge_y"].sum() > 0
    assert (graph["edge_y"][graph["edge_type"] == 2] == 0).all()


def test_gap_dataset(graph):
    gd = graph["gap_ds"]
    assert gd["F"].shape[0] == len(gd["gaps"]) == gd["end_km"].shape[0]
    assert np.isfinite(gd["F"]).all()
    assert (gd["prev_node"] >= -1).all()


def test_decoder_respects_speed_limit():
    dec = GapDecoder(n_in=5, h_node=4)
    dur = torch.tensor([1.0, 6.0])
    path, logsig = dec(torch.randn(2, 5) * 50, torch.randn(2, 4), torch.ones(2), dur)
    step = torch.diff(torch.cat([torch.zeros(2, 1, 2), path], 1), dim=1).norm(dim=-1)
    vmax = 15 * 1.852 * dur / 8 * np.sqrt(2)
    assert (step <= vmax[:, None] + 1e-4).all()


@pytest.mark.parametrize("use_graph", [True, False])
def test_training_smoke(graph, dsmall, use_graph):
    lab = dsmall["labels"]
    rng = np.random.default_rng(0)
    split = np.where(rng.random(len(lab)) < 0.6, "train", np.where(rng.random(len(lab)) < 0.5, "val", "test"))
    y = lab.y.values
    T = WindowTensors(dsmall["X"], dsmall["S"], dsmall["agg"], y, np.flatnonzero(split == "train"))
    r = train_tgat(graph, T, split, GNNConfig(epochs=3, patience=5, hidden=16, heads=2), 0,
                   torch.device("cpu"), use_graph=use_graph)
    assert len(r["history"]) == 3
    h = r["history"]
    assert {"loss_risk", "loss_fishing", "loss_gap", "loss_edge", "loss_traj"} <= set(h.columns)
    assert np.isfinite(h.loss_total).all()
    assert r["risk"].shape == (len(lab),) and ((r["risk"] >= 0) & (r["risk"] <= 1)).all()
    if not use_graph:
        assert (h.loss_edge == 0).all()
