"""Evaluation utilities.

Milestone 1 implements the synthetic-truth checks for gap detection.
Classification / event-level / XAI metrics are added in Milestones 2-4.

Hidden synthetic truth is joined here ONLY for evaluation - never as input.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

CAUSE_ORDER = ["dark_intentional", "equipment_glitch", "receiver_outage", "coverage", "between_voyages"]


def attach_gap_truth(gaps: pd.DataFrame, truth_reports: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    """Label each detected gap with its simulated cause (synthetic data only).

    Cause priority inside the silent interval: between_voyages (port stay or
    region exit) > dark_intentional > equipment_glitch > receiver_outage > coverage.
    """
    tr = truth_reports.merge(id_map, left_on="raw_id", right_on="raw_id")
    tr = tr.sort_values(["vessel_id", "timestamp"])
    by_v = {v: g for v, g in tr.groupby("vessel_id")}
    causes, susp, events = [], [], []
    for g in gaps.itertuples(index=False):
        t = by_v.get(g.vessel_id)
        if t is None:
            causes.append("unknown"); susp.append(False); events.append(None)
            continue
        ts = t.timestamp.values
        t0 = np.datetime64(g.gap_start.tz_convert(None))
        t1 = np.datetime64(g.gap_end.tz_convert(None))
        i0 = max(np.searchsorted(ts, t0, side="right") - 1, 0)
        i1 = min(np.searchsorted(ts, t1, side="left"), len(t) - 1)
        inside = t.iloc[np.searchsorted(ts, t0, side="right"):np.searchsorted(ts, t1, side="left")]
        if t.voyage_id.values[i0] != t.voyage_id.values[i1]:
            c = "between_voyages"
        else:
            reasons = set(inside.miss_reason)
            c = next((r for r in CAUSE_ORDER[:-1] if r in reasons), "coverage")
        causes.append(c)
        susp.append(bool(inside.suspicious.any()) if len(inside) else False)
        ev = inside.event_id.dropna().unique()
        events.append(",".join(map(str, ev)) if len(ev) else None)
    out = gaps.copy()
    out["truth_cause"] = causes
    out["truth_suspicious"] = susp
    out["truth_event_ids"] = events
    return out


def planted_dark_windows(truth_reports: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    """Contiguous runs of intentionally untransmitted reports per voyage (synthetic truth)."""
    tr = truth_reports.merge(id_map, on="raw_id").sort_values(["vessel_id", "timestamp"])
    tr["dark"] = tr.miss_reason.eq("dark_intentional")
    tr["run"] = (tr.dark != tr.groupby("vessel_id").dark.shift()).cumsum()
    d = tr[tr.dark].groupby(["vessel_id", "run"]).agg(
        start=("timestamp", "min"), end=("timestamp", "max"),
        suspicious=("suspicious", "any"), event_id=("event_id", lambda s: s.dropna().iloc[0] if s.notna().any() else None)
    ).reset_index()
    d["duration_min"] = (d.end - d.start).dt.total_seconds() / 60
    return d.drop(columns="run")


def gap_detection_report(gaps_truth: pd.DataFrame, dark_windows: pd.DataFrame,
                         threshold_min: float) -> Dict[str, object]:
    """Recall of planted blackouts and composition of all detected gaps.

    This is a *pipeline-validation* metric: it shows the gap detector finds
    every planted blackout longer than the threshold, and that most detected
    gaps are benign - i.e. gap duration alone is not a suspicion label.
    """
    detectable = dark_windows[dark_windows.duration_min >= threshold_min]
    hits = 0
    for w in detectable.itertuples(index=False):
        g = gaps_truth[(gaps_truth.vessel_id == w.vessel_id) &
                       (gaps_truth.gap_start <= w.end) & (gaps_truth.gap_end >= w.start)]
        hits += int(len(g) > 0)
    comp = gaps_truth.truth_cause.value_counts().to_dict()
    within = gaps_truth[gaps_truth.truth_cause != "between_voyages"]
    return {
        "planted_dark_windows": int(len(dark_windows)),
        "planted_dark_windows_detectable": int(len(detectable)),
        "planted_dark_windows_detected": int(hits),
        "planted_blackout_recall": float(hits / len(detectable)) if len(detectable) else float("nan"),
        "detected_gaps_total": int(len(gaps_truth)),
        "detected_gaps_by_truth_cause": {k: int(v) for k, v in comp.items()},
        "within_voyage_gaps": int(len(within)),
        "share_within_voyage_gaps_benign": float((within.truth_cause != "dark_intentional").mean()) if len(within) else float("nan"),
    }


# ==========================================================================
# Milestone 2 - classification, alert-rate and event-level metrics
# ==========================================================================
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve  # noqa: E402


def threshold_max_f1(y, p) -> float:
    """Decision threshold chosen on VALIDATION data (maximum F1)."""
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.maximum(pr + rc, 1e-12)
    i = int(np.nanargmax(f1[:-1])) if len(th) else 0
    return float(th[i]) if len(th) else 0.5


def binary_metrics(y, p, thr, active_hours=None) -> Dict[str, float]:
    y = np.asarray(y).astype(int)
    pred = (np.asarray(p) >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    out = {"threshold": float(thr), "precision": prec, "recall": rec,
           "f1": 2 * prec * rec / max(prec + rec, 1e-12),
           "pr_auc": float(average_precision_score(y, p)) if y.any() else float("nan"),
           "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn), "positives": int(y.sum()),
           "prevalence": float(y.mean())}
    if active_hours is not None:
        out["false_alerts_per_1000_vessel_h"] = 1000 * fp / max(float(np.sum(active_hours)), 1e-9)
    return out


def event_level(labels: pd.DataFrame, pred: np.ndarray, events: pd.DataFrame, id_map: pd.DataFrame) -> pd.DataFrame:
    """Planted suspicious events (synthetic truth) detected by >= 1 alert of an involved vessel
    overlapping the event interval. Only events with at least one window in `labels` count."""
    to_v = dict(zip(id_map.raw_id, id_map.vessel_id))
    L = labels.assign(alert=np.asarray(pred).astype(bool))
    rows = []
    sus = events[events.synthetic_demo_label == "synthetic_demo_suspicious"]
    for e in sus.itertuples(index=False):
        vids = [to_v.get(x) for x in e.vessels.split(",")]
        w = L[L.vessel_id.isin(vids) & (L.t0 < e.end) & (L.t1 > e.start)]
        if w.empty:
            continue
        rows.append({"event_id": e.event_id, "event_type": e.event_type, "n_windows": len(w),
                     "detected": bool(w.alert.any())})
    return pd.DataFrame(rows)
