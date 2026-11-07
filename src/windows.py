"""Vessel-window samples for Milestone 2+ models.

One sample = one vessel x one UTC-aligned window (default 6 h), represented as
  * a sequence of T = window_h*60/step_min steps (default 36 x 10 min) with an
    explicit `observed` mask - unobserved steps are NOT interpolated; they carry
    only "minutes since last fix", "inside detected AIS gap" and
    "possible-dark-encounter link" indicators;
  * a static vector (vessel class one-hot, log length);
  * a window-level aggregate vector (for the MLP and the rule reference).

A window exists when the vessel sent at least one fix in it or when the window
overlaps one of the vessel's at-sea AIS gaps (so fully dark windows are kept).

Neighbour context uses a single cKDTree over (x, y, time-bin) - sparse, never
all-pairs. It gives non-graph baselines scalar summaries of nearby traffic; the
Milestone-3 graph model will instead pass messages between vessel nodes.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .config import WindowConfig

STEP_FEATURES = [
    "observed", "sog", "cog_sin", "cog_cos", "abs_turn", "fishing_motion", "in_mpa", "mpa_dist",
    "eez_dist", "other_side", "port_dist", "coast_dist", "since_obs", "in_gap", "nn_dist", "n_2km",
    "nn_close", "nn_is_carrier", "nn_is_fishing", "nn_rel_speed", "dark_link", "gap_silence",
]
STATIC_CLASSES = ["fishing", "reefer_carrier", "cargo", "tanker", "tug", "passenger", "other"]
STATIC_FEATURES = [f"type_{c}" for c in STATIC_CLASSES] + ["log_length"]
FI = {f: i for i, f in enumerate(STEP_FEATURES)}


def _steps(ts: pd.Series, T0: pd.Timestamp, step_min: int) -> np.ndarray:
    return ((ts - T0).dt.total_seconds() // (step_min * 60)).astype(np.int64).values


def _cell_table(ais: pd.DataFrame, layers, wcfg: WindowConfig, T0) -> pd.DataFrame:
    """Last fix per (vessel, 10-min step) plus nearest-neighbour context."""
    a = ais.assign(step=_steps(ais.timestamp, T0, wcfg.step_min))
    c = a.sort_values("timestamp").groupby(["vessel_id", "step"], sort=False).last().reset_index()
    x, y = layers.to_metric(c.latitude.values, c.longitude.values)
    pts = np.column_stack([x, y, c.step.values * 1e7])       # bins far apart in the 3rd axis
    tree = cKDTree(pts)
    d, j = tree.query(pts, k=2)
    nn_d = d[:, 1]
    nn_j = j[:, 1]
    none = ~np.isfinite(nn_d) | (nn_d > 1e6)
    nn_d = np.where(none, 50_000.0, np.minimum(nn_d, 50_000.0))
    nn_j = np.where(none, 0, nn_j)
    n2 = tree.query_ball_point(pts, r=wcfg.nn_radius_m, return_length=True) - 1
    vt = c.vessel_type.values
    c["nn_dist_m"] = nn_d
    c["n_2km"] = n2
    c["nn_close"] = (nn_d <= wcfg.close_m) & ~none
    c["nn_is_carrier"] = (vt[nn_j] == "reefer_carrier") & (nn_d <= wcfg.nn_radius_m)
    c["nn_is_fishing"] = (vt[nn_j] == "fishing") & (nn_d <= wcfg.nn_radius_m)
    c["nn_rel_speed"] = np.where(nn_d <= wcfg.nn_radius_m,
                                 np.abs(c.sog.fillna(0).values - c.sog.fillna(0).values[nn_j]), 0.0)
    return c


def build_windows(ais: pd.DataFrame, gaps: pd.DataFrame, layers, prism: Optional[pd.DataFrame] = None,
                  wcfg: WindowConfig = WindowConfig()) -> Dict[str, object]:
    T = wcfg.n_steps
    T0 = ais.timestamp.min().floor("D")
    step_td = pd.Timedelta(minutes=wcfg.step_min)
    cells = _cell_table(ais, layers, wcfg, T0)
    cells["win"] = cells.step // T
    at_sea = gaps[gaps.gap_context == "at_sea"] if "gap_context" in gaps else gaps

    # ---- window index: observed windows + windows covered by at-sea gaps
    keys = set(zip(cells.vessel_id, cells.win))
    g_s = _steps(at_sea.gap_start, T0, wcfg.step_min)
    g_e = _steps(at_sea.gap_end, T0, wcfg.step_min)
    for v, s, e in zip(at_sea.vessel_id.values, g_s, g_e):
        for w in range(s // T, e // T + 1):
            keys.add((v, w))
    meta = pd.DataFrame(sorted(keys), columns=["vessel_id", "win"])
    meta["t0"] = T0 + pd.to_timedelta(meta.win * wcfg.window_h, unit="h")
    meta["t1"] = meta.t0 + pd.Timedelta(hours=wcfg.window_h)
    row_of = {k: i for i, k in enumerate(zip(meta.vessel_id, meta.win))}
    N = len(meta)
    X = np.zeros((N, T, len(STEP_FEATURES)), np.float32)

    # ---- observed cells
    ri = np.array([row_of[k] for k in zip(cells.vessel_id, cells.win)])
    pi = (cells.step % T).values
    sog = cells.sog.fillna(0).values
    cog = np.deg2rad(cells.cog.fillna(0).values)
    f = {
        "observed": 1.0, "sog": np.clip(sog / 10, 0, 3), "cog_sin": np.sin(cog), "cog_cos": np.cos(cog),
        "abs_turn": np.clip(np.abs(cells.turn_rate.fillna(0).values) / 20, 0, 3),
        "fishing_motion": cells.fishing_motion_score.values, "in_mpa": cells.in_mpa.values.astype(float),
        "mpa_dist": np.clip(cells.mpa_distance_m.values / 10_000, -2, 5),
        "eez_dist": np.clip(cells.eez_distance_m.values / 20_000, -3, 5),
        "other_side": (cells.eez_distance_m.values < 0).astype(float),
        "port_dist": np.log1p(cells.port_distance_m.values / 1000) / 4,
        "coast_dist": np.log1p(cells.coast_distance_m.values / 1000) / 4,
        "nn_dist": np.log1p(cells.nn_dist_m.values / 1000) / 4, "n_2km": np.clip(cells.n_2km.values / 5, 0, 2),
        "nn_close": cells.nn_close.values.astype(float), "nn_is_carrier": cells.nn_is_carrier.values.astype(float),
        "nn_is_fishing": cells.nn_is_fishing.values.astype(float),
        "nn_rel_speed": np.clip(cells.nn_rel_speed.values / 10, 0, 2),
    }
    for k, val in f.items():
        X[ri, pi, FI[k]] = val

    # ---- minutes since last fix (for every cell, observed or not)
    vessel_times = {v: g.timestamp.values for v, g in ais.sort_values("timestamp").groupby("vessel_id")}
    since = np.zeros((N, T), np.float32)
    for v, idx in meta.groupby("vessel_id").indices.items():
        wins = meta.win.values[idx]
        steps_end = (wins[:, None] * T + np.arange(1, T + 1)[None, :]).ravel() * wcfg.step_min
        ends = np.datetime64(T0.tz_convert(None), "ns") + steps_end.astype("timedelta64[m]")
        mt = vessel_times[v]
        k = np.searchsorted(mt, ends.astype(mt.dtype), side="right") - 1
        mins = np.where(k >= 0, (ends - mt[np.clip(k, 0, None)]).astype("timedelta64[s]").astype(float) / 60, 720)
        since[idx] = np.clip(mins, 0, 720).reshape(len(idx), T)
    X[:, :, FI["since_obs"]] = np.log1p(since) / np.log1p(720)

    # ---- inside detected at-sea gap
    def _mark(vessel, s, e, feat):
        for w in range(s // T, e // T + 1):
            r = row_of.get((vessel, w))
            if r is None:
                continue
            lo, hi = max(s, w * T), min(e, w * T + T - 1)
            if hi >= lo:
                X[r, lo - w * T:hi - w * T + 1, FI[feat]] = 1.0

    def _set(vessel, s, e, feat, val):
        for w in range(s // T, e // T + 1):
            r = row_of.get((vessel, w))
            if r is not None:
                lo, hi = max(s, w * T), min(e, w * T + T - 1)
                if hi >= lo:
                    X[r, lo - w * T:hi - w * T + 1, FI[feat]] = val

    sil = at_sea.concurrent_silence_frac.values if "concurrent_silence_frac" in at_sea else np.full(len(at_sea), np.nan)
    for v, s, e, q in zip(at_sea.vessel_id.values, g_s, g_e, sil):
        _mark(v, s + 1, e - 1, "in_gap")
        # reception context of the gap: share of nearby vessels also silent (0 if unknown)
        _set(v, s + 1, e - 1, "gap_silence", 0.0 if np.isnan(q) else float(q))
    # ---- possible dark-encounter links (space-time prism), both vessels
    if prism is not None and len(prism):
        ps = _steps(prism.first_bin, T0, wcfg.step_min)
        pe = _steps(prism.last_bin, T0, wcfg.step_min)
        for a, b, s, e in zip(prism.vessel_a.values, prism.vessel_b.values, ps, pe):
            _mark(a, s, e, "dark_link")
            _mark(b, s, e, "dark_link")

    # ---- static
    vinfo = ais.groupby("vessel_id").agg(vessel_type=("vessel_type", "first"), length_m=("length_m", "median"))
    meta = meta.join(vinfo, on="vessel_id")
    meta["region_id"] = ais.region_id.iloc[0] if ais.region_id.nunique() == 1 else \
        meta.vessel_id.map(ais.groupby("vessel_id").region_id.agg(lambda s: s.mode().iloc[0]))
    vt = meta.vessel_type.where(meta.vessel_type.isin(STATIC_CLASSES), "other")
    S = np.zeros((N, len(STATIC_FEATURES)), np.float32)
    for i, cls in enumerate(STATIC_CLASSES):
        S[:, i] = (vt == cls).values
    S[:, -1] = np.log1p(meta.length_m.fillna(20).values) / 6
    # active vessel-hours: observed or known-silent-at-sea steps
    active = (X[:, :, FI["observed"]] > 0) | (X[:, :, FI["in_gap"]] > 0)
    meta["active_hours"] = active.sum(1) * wcfg.step_min / 60
    # voyage segments touched by each window (for time-disjoint splitting)
    segs = cells.groupby(["vessel_id", "win"]).segment_id.agg(lambda s: set(s.dropna()))
    seg_sets = [set(segs.get(k, set())) for k in zip(meta.vessel_id, meta.win)]
    if "end_idx" in at_sea and len(at_sea):  # fully dark windows inherit the gap's segment
        gseg = ais.segment_id.values[at_sea.end_idx.values]
        for v, s, e, sg in zip(at_sea.vessel_id.values, g_s, g_e, gseg):
            for w in range(s // T, e // T + 1):
                r = row_of.get((v, w))
                if r is not None:
                    seg_sets[r].add(sg)
    meta["segments"] = [",".join(sorted(x)) for x in seg_sets]
    agg = window_aggregates(X, S, meta, at_sea if len(at_sea) else gaps, wcfg)
    return {"meta": meta, "X": X, "S": S, "agg": agg, "T0": T0, "wcfg": wcfg}


def window_aggregates(X, S, meta, gaps, wcfg: WindowConfig) -> pd.DataFrame:
    """Window-level summary features for the MLP / rule reference."""
    obs = X[:, :, FI["observed"]] > 0
    n_obs = obs.sum(1)
    safe = np.maximum(n_obs, 1)

    def mmean(v):
        return np.where(n_obs > 0, (v * obs).sum(1) / safe, 0.0)

    sog_kn = X[:, :, FI["sog"]] * 10
    fms = X[:, :, FI["fishing_motion"]]
    in_mpa = X[:, :, FI["in_mpa"]]
    other = X[:, :, FI["other_side"]]
    fishy = obs & (fms > 0.3)
    close = obs & (X[:, :, FI["nn_close"]] > 0)
    big = np.where(obs, 1, 0)
    a = pd.DataFrame({
        "obs_frac": n_obs / X.shape[1],
        "sog_mean": mmean(sog_kn),
        "sog_std": np.sqrt(np.maximum(mmean(sog_kn ** 2) - mmean(sog_kn) ** 2, 0)),
        "sog_max": np.where(obs, sog_kn, 0).max(1),
        "frac_fishing_band": mmean(((sog_kn >= 1.5) & (sog_kn <= 5)).astype(float)),
        "frac_slow": mmean((sog_kn < wcfg.slow_kn).astype(float)),
        "fms_mean": mmean(fms), "fms_max": np.where(obs, fms, 0).max(1),
        "turn_mean": mmean(X[:, :, FI["abs_turn"]]),
        "frac_in_mpa": mmean(in_mpa),
        "min_mpa_km": np.where(obs, X[:, :, FI["mpa_dist"]], 5).min(1) * 10,
        "frac_fish_in_mpa": mmean((fishy & (in_mpa > 0)).astype(float)),
        "frac_other_side": mmean(other),
        "min_abs_eez_km": np.where(obs, np.abs(X[:, :, FI["eez_dist"]]), 5).min(1) * 20,
        "frac_fish_other_side": mmean((fishy & (other > 0)).astype(float)),
        "min_port_log": np.where(obs, X[:, :, FI["port_dist"]], 2).min(1),
        "mean_coast_log": mmean(X[:, :, FI["coast_dist"]]),
        "max_since_obs": X[:, :, FI["since_obs"]].max(1),
        "frac_in_gap": X[:, :, FI["in_gap"]].mean(1),
        "min_nn_log": np.where(obs, X[:, :, FI["nn_dist"]], 4).min(1),
        "close_min": close.sum(1) * wcfg.step_min,
        "close_slow_min": (close & (sog_kn < wcfg.slow_kn)).sum(1) * wcfg.step_min,
        "close_carrier_min": (close & (X[:, :, FI["nn_is_carrier"]] > 0)).sum(1) * wcfg.step_min,
        "max_n_2km": X[:, :, FI["n_2km"]].max(1) * 5,
        "dark_link_frac": X[:, :, FI["dark_link"]].mean(1),
    })
    # gap features: gaps overlapping the window (longest one describes the window)
    gcols = ["max_gap_min", "gap_disp_km", "gap_speed_kn", "gap_silence", "gap_silence_known",
             "gap_crossed", "gap_start_port", "gap_end_port", "gap_min_mpa_km", "pre_gap_sog", "pre_gap_course"]
    G = np.zeros((len(meta), len(gcols)))
    G[:, 3] = -1
    key = {k: i for i, k in enumerate(zip(meta.vessel_id, meta.t0))}
    wh = pd.Timedelta(hours=wcfg.window_h)
    g = gaps.sort_values("gap_duration_min")
    for r in g.itertuples(index=False):
        t = r.gap_start.floor(f"{wcfg.window_h}h")
        while t < r.gap_end:
            i = key.get((r.vessel_id, t))
            if i is not None:  # later (longer) gaps overwrite shorter ones
                sil = getattr(r, "concurrent_silence_frac", np.nan)
                G[i] = [r.gap_duration_min, r.displacement_m / 1000, r.implied_speed_kn,
                        -1 if pd.isna(sil) else sil, 0 if pd.isna(sil) else 1,
                        float(getattr(r, "crossed_boundary", False)), float(getattr(r, "start_near_port", False)),
                        float(getattr(r, "end_near_port", False)),
                        getattr(r, "min_mpa_distance_m", 50_000) / 1000,
                        0 if pd.isna(r.pre_gap_sog_mean) else r.pre_gap_sog_mean,
                        0 if pd.isna(r.pre_gap_course_change_deg) else r.pre_gap_course_change_deg]
            t += wh
    for j, c in enumerate(gcols):
        a[c] = G[:, j]
    a["log_max_gap"] = np.log1p(a.max_gap_min)
    for j, c in enumerate(STATIC_FEATURES):
        a[c] = S[:, j]
    return a


# --------------------------------------------------------------------------
# Synthetic labels (hidden truth -> evaluation/training targets only)
# --------------------------------------------------------------------------
def attach_synthetic_labels(W: Dict, truth_reports: pd.DataFrame, id_map: pd.DataFrame,
                            gaps_truth: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """y = 1 if the hidden simulation marks the vessel as behaving suspiciously at
    any time inside the window (label name: synthetic_demo_suspicious).
    Also returns the benign context of negative windows for false-positive analysis."""
    meta, wcfg, T0 = W["meta"], W["wcfg"], W["T0"]
    tr = truth_reports.merge(id_map, on="raw_id")
    tr = tr.assign(win=((tr.timestamp - T0).dt.total_seconds() // (wcfg.window_h * 3600)).astype(int))
    g = tr.groupby(["vessel_id", "win"])
    lab = g.suspicious.any().rename("y")
    ev = g.apply(lambda d: ",".join(sorted(set(d.loc[d.suspicious, "event_id"].dropna()))), include_groups=False) \
        .rename("event_ids")
    beh = g.behavior.agg(lambda s: set(s)).rename("behaviors")
    ev_all = g.event_id.agg(lambda s: set(s.dropna())).rename("all_events")
    miss = g.miss_reason.agg(lambda s: set(s)).rename("miss")
    fish = g.behavior.agg(lambda s: float((s == "fishing").mean())).rename("fishing_frac")
    dark = g.miss_reason.agg(lambda s: bool((s == "dark_intentional").any())).rename("dark_in_window")
    L = pd.concat([lab, ev, beh, ev_all, miss, fish, dark], axis=1)
    out = meta.join(L, on=["vessel_id", "win"])
    out["y"] = out.y.fillna(False).astype(int)
    out["event_ids"] = out.event_ids.fillna("")
    out["fishing_frac"] = out.fishing_frac.fillna(0.0)
    out["dark_in_window"] = out.dark_in_window.fillna(False).astype(bool)
    agg = W["agg"]

    def ctx(i, r):
        b = r.behaviors if isinstance(r.behaviors, set) else set()
        m = r.miss if isinstance(r.miss, set) else set()
        if r.y:
            return "suspicious"
        if m & {"receiver_outage", "coverage", "equipment_glitch"} and agg.max_gap_min.iat[i] >= 60:
            return "benign_gap (outage/coverage/fault)"
        if b & {"anchored", "tug_assist"}:
            return "anchorage / tug assist"
        if "loitering" in b:
            return "carrier loitering"
        if agg.close_min.iat[i] >= 30 and "fishing" in b:
            return "legal parallel fishing"
        if agg.frac_in_mpa.iat[i] > 0:
            return "legal MPA transit"
        if agg.frac_other_side.iat[i] > 0:
            return "legal boundary crossing (ferry/transit)"
        if agg.max_n_2km.iat[i] >= 3:
            return "dense traffic"
        return "other normal"

    out["context"] = [ctx(i, r) for i, r in enumerate(out.itertuples(index=False))]
    return out.drop(columns=["behaviors", "miss", "all_events"])
