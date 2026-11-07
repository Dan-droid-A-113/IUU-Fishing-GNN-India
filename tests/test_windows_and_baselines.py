import numpy as np
import pandas as pd
import pytest
import torch

from src.baselines import (GRUClassifier, MLPClassifier, TCNClassifier, WindowTensors, predict, rule_score,
                           train_model)
from src.config import TrainConfig, WindowConfig
from src.evaluation import binary_metrics, event_level, threshold_max_f1
from src.graph_builder import gap_prism_pairs
from src.windows import FI, STEP_FEATURES, attach_synthetic_labels, build_windows


@pytest.fixture(scope="module")
def windows(processed, layers, synth):
    prism = gap_prism_pairs(processed["ais"], processed["gaps"])
    W = build_windows(processed["ais"], processed["gaps"], layers, prism, WindowConfig())
    lab = attach_synthetic_labels(W, synth["truth_reports"], processed["id_map"])
    return W, lab


def test_window_shapes(windows):
    W, lab = windows
    N, T, F = W["X"].shape
    assert T == 36 and F == len(STEP_FEATURES) and N == len(lab) == len(W["agg"])
    assert np.isfinite(W["X"]).all() and np.isfinite(W["agg"].values).all()


def test_unobserved_steps_are_not_interpolated(windows):
    X = windows[0]["X"]
    unobs = X[:, :, FI["observed"]] == 0
    for f in ("sog", "fishing_motion", "in_mpa", "nn_close", "port_dist"):
        assert (X[:, :, FI[f]][unobs] == 0).all(), f


def test_fully_dark_windows_are_kept(layers):
    from src.preprocessing import run_preprocessing
    from tests.conftest import make_track
    # 14-h silence at sea (no port / region edge nearby) must yield at least one window with no fixes
    v = make_track("v1", "2024-01-01", list(range(0, 60, 5)) + list(range(900, 960, 5)),
                   lat0=8.9, lon0=78.9, dlat=0.0001)
    res = run_preprocessing(v, layers)
    assert (res["gaps"].gap_context == "at_sea").all()
    W = build_windows(res["ais"], res["gaps"], layers, None, WindowConfig())
    obs = W["X"][:, :, FI["observed"]].sum(1)
    dark = obs == 0
    assert dark.any()
    assert (W["X"][dark][:, :, FI["in_gap"]].sum(1) > 0).all()


def test_labels(windows):
    _, lab = windows
    assert set(lab.y.unique()) <= {0, 1} and lab.y.sum() > 0
    assert (lab.context[lab.y == 1] == "suspicious").all()
    assert (lab.active_hours <= 6).all()


def test_rule_score_range(windows):
    r = rule_score(windows[0]["agg"])
    assert (r >= 0).all() and (r <= 1).all()


@pytest.mark.parametrize("cls", [MLPClassifier, GRUClassifier, TCNClassifier])
def test_model_forward_shapes(cls):
    B, T, F, A = 5, 36, 29, 40
    m = cls(A if cls is MLPClassifier else F)
    out = m(torch.randn(B, A), torch.randn(B, T, F))
    assert out.shape == (B,)


def test_training_runs_and_predicts(windows):
    W, lab = windows
    y = lab.y.values
    idx = np.arange(len(y))
    rng = np.random.default_rng(0)
    rng.shuffle(idx)
    tr, va = idx[: len(idx) // 2], idx[len(idx) // 2:]
    D = WindowTensors(W["X"], W["S"], W["agg"], y, tr)
    res = train_model("TCN", D, tr, va, TrainConfig(epochs=2, patience=5), seed=0)
    p = predict(res["model"], D, va, torch.device("cpu"))
    assert p.shape == (len(va),) and ((p >= 0) & (p <= 1)).all()
    assert len(res["history"]) == 2 and {"train_loss", "val_loss", "val_pr_auc"} <= set(res["history"])


def test_metrics_and_threshold():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.7, 0.9])
    thr = threshold_max_f1(y, p)
    m = binary_metrics(y, p, thr, active_hours=np.full(5, 6.0))
    assert m["f1"] == 1.0 and m["fp"] == 0 and m["false_alerts_per_1000_vessel_h"] == 0


def test_event_level():
    t = pd.Timestamp("2024-01-01", tz="UTC")
    lab = pd.DataFrame({"vessel_id": ["v1", "v1", "v2"], "t0": [t, t + pd.Timedelta("6h"), t],
                        "t1": [t + pd.Timedelta("6h"), t + pd.Timedelta("12h"), t + pd.Timedelta("6h")]})
    ev = pd.DataFrame({"event_id": ["E1", "E2"], "event_type": ["x", "y"], "vessels": ["R1", "R2"],
                       "synthetic_demo_label": ["synthetic_demo_suspicious"] * 2,
                       "start": [t + pd.Timedelta("7h"), t + pd.Timedelta("1h")],
                       "end": [t + pd.Timedelta("8h"), t + pd.Timedelta("2h")]})
    idm = pd.DataFrame({"raw_id": ["R1", "R2"], "vessel_id": ["v1", "v2"]})
    out = event_level(lab, np.array([False, True, False]), ev, idm)
    assert out.set_index("event_id").detected.to_dict() == {"E1": True, "E2": False}
