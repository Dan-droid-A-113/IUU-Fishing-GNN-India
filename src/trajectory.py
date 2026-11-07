"""Trajectory reconstruction during AIS gaps (Milestone 3).

The neural decoder lives in `temporal_gnn.GapDecoder`; this module prepares
the gap dataset and the non-learning references.

Design choices (valid for real data, not only the simulator):
  * The decoder is trained on the *observed reappearance position* only, which
    real AIS provides. It sees nothing after the silence starts: only pre-gap
    kinematics, the gap's duration and context, and the graph embedding of
    the vessel's PREVIOUS 6-h window (causal).
  * The reappearance error is then used as ONE feature of the risk head, never
    as a standalone rule.
  * Hidden simulator paths are used only to *evaluate* the intermediate path
    (ADE); they never enter training.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

KM_PER_DEG = 111.32
K_POINTS = 8          # predicted points per gap (last = reappearance)
GAP_FEATURES = ["sog_last", "sog_mean", "cog_sin", "cog_cos", "course_change", "log_dur", "dur_h",
                "fms_last", "mpa_km", "eez_km", "log_port", "silence", "silence_known", "n_neigh"]


def _offsets_km(lat0, lon0, lat, lon):
    east = (np.asarray(lon) - lon0) * KM_PER_DEG * np.cos(np.deg2rad(lat0))
    north = (np.asarray(lat) - lat0) * KM_PER_DEG
    return np.stack([east, north], -1)


def build_gap_dataset(d: Dict, node_of: Dict, T0, window_h: int = 6, min_min: float = 60,
                      max_min: float = 24 * 60) -> Dict:
    """At-sea AIS gaps usable for reconstruction, with inputs, targets and references."""
    ais, gaps = d["ais"], d["gaps"]
    g = gaps[(gaps.gap_context == "at_sea") & gaps.gap_duration_min.between(min_min, max_min)].reset_index(drop=True)
    last = ais.iloc[g.start_idx.values]
    dur = g.gap_duration_min.values
    cog = np.deg2rad(last.cog.fillna(0).values)
    sil = g.concurrent_silence_frac.values
    F = np.column_stack([
        np.nan_to_num(last.sog.values) / 10, np.nan_to_num(g.pre_gap_sog_mean.values) / 10,
        np.sin(cog), np.cos(cog), np.nan_to_num(g.pre_gap_course_change_deg.values) / 180,
        np.log1p(dur) / 7, dur / 1440, last.fishing_motion_score.values,
        np.clip(g.start_mpa_distance_m.values / 10_000, -2, 5), np.clip(g.start_eez_distance_m.values / 20_000, -3, 5),
        np.log1p(g.start_port_distance_m.values / 1000) / 4,
        np.nan_to_num(sil, nan=0.0), (~np.isnan(sil)).astype(float), np.clip(g.n_neighbours_before.values / 5, 0, 3),
    ]).astype(np.float32)
    types = ["fishing", "reefer_carrier", "cargo", "tanker", "tug", "passenger", "other"]
    vt = g.vessel_type.where(g.vessel_type.isin(types), "other")
    F = np.concatenate([F, np.stack([(vt == t).values for t in types], 1).astype(np.float32)], 1)
    end = _offsets_km(g.start_lat.values, g.start_lon.values, g.end_lat.values, g.end_lon.values)
    wh = pd.Timedelta(hours=window_h)
    w_s = ((g.gap_start - T0) // wh).astype(int).values
    w_e = ((g.gap_end - T0) // wh).astype(int).values
    prev = np.array([node_of.get((v, w - 1), -1) for v, w in zip(g.vessel_id, w_s)])
    end_node = np.array([node_of.get((v, w), -1) for v, w in zip(g.vessel_id, w_e)])
    span_g, span_n = [], []
    for k, (v, a, b) in enumerate(zip(g.vessel_id, w_s, w_e)):
        for w in range(a, b + 1):
            n = node_of.get((v, w))
            if n is not None:
                span_g.append(k)
                span_n.append(n)
    # references: dead reckoning (last speed & course held) and "stay put"
    frac = np.arange(1, K_POINTS + 1) / K_POINTS
    t_h = dur[:, None] / 60 * frac[None, :]
    sog_kmh = np.nan_to_num(last.sog.values)[:, None] * 1.852
    dr = np.stack([sog_kmh * t_h * np.sin(cog)[:, None], sog_kmh * t_h * np.cos(cog)[:, None]], -1)
    out = {"gaps": g, "F": F, "end_km": end.astype(np.float32), "dur_h": (dur / 60).astype(np.float32),
           "prev_node": prev, "end_node": end_node, "span_gap": np.array(span_g, int),
           "span_node": np.array(span_n, int), "dead_reckoning": dr.astype(np.float32), "frac": frac}
    out["truth_path"] = _truth_paths(d, g, frac) if "truth" in d else None
    return out


def _truth_paths(d, g, frac):
    """Hidden simulator positions at the K evaluation times (synthetic evaluation only)."""
    tr = d["truth"].merge(d["id_map"], on="raw_id")
    by_v = {v: x.sort_values("timestamp") for v, x in tr.groupby("vessel_id")}
    P = np.full((len(g), len(frac), 2), np.nan, np.float32)
    for k, r in enumerate(g.itertuples(index=False)):
        x = by_v.get(r.vessel_id)
        if x is None:
            continue
        t = (x.timestamp - r.gap_start).dt.total_seconds().values
        dur_s = (r.gap_end - r.gap_start).total_seconds()
        m = (t >= -600) & (t <= dur_s + 600)
        if m.sum() < 2:
            continue
        tq = frac * dur_s
        la = np.interp(tq, t[m], x.lat_true.values[m])
        lo = np.interp(tq, t[m], x.lon_true.values[m])
        P[k] = _offsets_km(r.start_lat, r.start_lon, la, lo)
    return P


def path_errors(pred: np.ndarray, truth: np.ndarray) -> Dict[str, float]:
    """ADE (mean over the K points) and FDE (last point) in km, ignoring missing truth."""
    ok = ~np.isnan(truth).any(axis=(1, 2))
    if not ok.any():
        return {"ade_km": float("nan"), "fde_km": float("nan"), "n": 0}
    e = np.linalg.norm(pred[ok] - truth[ok], axis=-1)
    return {"ade_km": float(e.mean()), "fde_km": float(e[:, -1].mean()),
            "median_fde_km": float(np.median(e[:, -1])), "n": int(ok.sum())}


def endpoint_coverage(mu: np.ndarray, logsig: np.ndarray, end: np.ndarray, level_chi2: float = 4.605) -> float:
    """Share of reappearances inside the predicted 90% uncertainty ellipse (chi2, 2 dof)."""
    z2 = (((end - mu) / np.exp(logsig)) ** 2).sum(-1)
    return float((z2 <= level_chi2).mean())
