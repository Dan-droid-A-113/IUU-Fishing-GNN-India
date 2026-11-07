"""Dynamic interaction-graph construction.

Milestone 1 provides the sparse spatial neighbour search (BallTree on
time-binned positions, never all-pairs). It is used now to group interacting
vessels for leakage-safe splitting. Milestone 3 extends it with full edge
features (separation, duration, relative speed/course, parallel-motion score,
low-speed duration, gap overlap, MPA distance, SAR evidence) and the PyG
temporal-graph snapshots G_t = (V_t, E_t, X_t, Z_t).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_R = 6_371_008.8


def binned_positions(ais: pd.DataFrame, bin_min: int = 10, exclude_port_m: float = 0.0) -> pd.DataFrame:
    """One mean position per vessel per time bin (optionally excluding port areas)."""
    df = ais
    if exclude_port_m and "port_distance_m" in df:
        df = df[df.port_distance_m > exclude_port_m]
    tbin = df.timestamp.dt.floor(f"{bin_min}min").rename("tbin")
    agg = {"latitude": "mean", "longitude": "mean"}
    if "sog" in df:
        agg["sog"] = "mean"
    return df.groupby([tbin, df.vessel_id])[list(agg)].agg(agg).reset_index()


def proximity_pairs(ais: pd.DataFrame, radius_m: float = 1000.0, bin_min: int = 10,
                    min_bins: int = 2, exclude_port_m: float = 3000.0) -> pd.DataFrame:
    """Vessel pairs that were within `radius_m` in at least `min_bins` time bins.

    Uses a haversine BallTree per time bin (O(n log n) per bin). Positions within
    `exclude_port_m` of a port are ignored so that shared harbours do not
    create spurious "encounters".
    """
    pos = binned_positions(ais, bin_min, exclude_port_m)
    rows = []
    for tb, g in pos.groupby("tbin", sort=True):
        if len(g) < 2:
            continue
        X = np.radians(g[["latitude", "longitude"]].values)
        vid = g.vessel_id.values
        ind, dist = BallTree(X, metric="haversine").query_radius(X, r=radius_m / EARTH_R,
                                                                 return_distance=True)
        for i in range(len(g)):
            for j, d in zip(ind[i], dist[i]):
                if j > i:
                    a, b = sorted((vid[i], vid[j]))
                    rows.append((tb, a, b, d * EARTH_R))
    cols = ["tbin", "vessel_a", "vessel_b", "dist_m"]
    if not rows:
        return pd.DataFrame(columns=["vessel_a", "vessel_b", "n_bins", "first_bin", "last_bin", "min_dist_m"])
    p = pd.DataFrame(rows, columns=cols)
    out = (p.groupby(["vessel_a", "vessel_b"])
           .agg(n_bins=("tbin", "size"), first_bin=("tbin", "min"), last_bin=("tbin", "max"),
                min_dist_m=("dist_m", "min"))
           .reset_index())
    return out[out.n_bins >= min_bins].reset_index(drop=True)


def gap_prism_pairs(ais: pd.DataFrame, gaps: pd.DataFrame, vmax_kn: float = 10.0,
                    max_partner_sog_kn: float = 2.0, bin_min: int = 10, min_bins: int = 3,
                    exclude_port_m: float = 3000.0, max_gap_min: float = 12 * 60) -> pd.DataFrame:
    """Candidate *possible dark encounters* (space-time prism test).

    While vessel A is silent (at-sea gap from P0 at t0 to P1 at t1), a slow-moving
    vessel B at position X in time bin t is a candidate partner if A could have
    reached X and still made it to P1:  d(P0,X) <= v*(t-t0)  and  d(X,P1) <= v*(t1-t).
    This finds rendezvous that happen *during* a blackout, which plain proximity
    search cannot see. It is an inferred link, not evidence of a meeting.
    """
    cols = ["ais_gap_id", "vessel_a", "vessel_b", "n_bins", "first_bin", "last_bin", "link_type"]
    g = gaps
    if "gap_context" in g:
        g = g[g.gap_context == "at_sea"]
    g = g[g.gap_duration_min <= max_gap_min]
    if not len(g):
        return pd.DataFrame(columns=cols)
    pos = binned_positions(ais, bin_min, exclude_port_m)
    pos = pos[pos.sog <= max_partner_sog_kn]
    tb = pos.tbin.values
    v_ms = vmax_kn * 0.514444
    rows = []
    for r in g.itertuples(index=False):
        m = (tb > np.datetime64(r.gap_start.tz_convert(None))) & (tb < np.datetime64(r.gap_end.tz_convert(None)))
        c = pos[m & (pos.vessel_id.values != r.vessel_id)]
        if c.empty:
            continue
        dt0 = (c.tbin - r.gap_start).dt.total_seconds().values
        dt1 = (r.gap_end - c.tbin).dt.total_seconds().values
        d0 = _hav(r.start_lat, r.start_lon, c.latitude.values, c.longitude.values)
        d1 = _hav(c.latitude.values, c.longitude.values, r.end_lat, r.end_lon)
        ok = c[(d0 <= v_ms * dt0) & (d1 <= v_ms * dt1)]
        for vb, s in ok.groupby("vessel_id").tbin:
            if len(s) >= min_bins:
                rows.append((r.ais_gap_id, r.vessel_id, vb, len(s), s.min(), s.max(), "possible_dark_encounter"))
    return pd.DataFrame(rows, columns=cols)


def _hav(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


# --------------------------------------------------------------------------
# Milestone 3: time-stamped interaction edges between vessel-window nodes
# --------------------------------------------------------------------------
EDGE_FEATURES = ["log_min_dist", "frac_within_2km", "frac_close", "frac_slow_close", "rel_speed",
                 "parallel", "log_port_dist", "is_dark_link", "is_temporal", "dark_overlap"]


def window_interaction_edges(ais: pd.DataFrame, layers, T0, wcfg) -> pd.DataFrame:
    """Proximity interactions per 6-h window, aggregated from 10-min steps.

    One cKDTree over (x, y, step) returns only same-step pairs within
    `wcfg.nn_radius_m` (sparse; never all-pairs). A pair becomes an edge when the
    two vessels were within 2 km for at least two steps (>= 20 min) in the window.
    """
    from scipy.spatial import cKDTree
    from .windows import _cell_table

    T = wcfg.n_steps
    c = _cell_table(ais, layers, wcfg, T0)
    x, y = layers.to_metric(c.latitude.values, c.longitude.values)
    tree = cKDTree(np.column_stack([x, y, c.step.values * 1e7]))
    pr = tree.query_pairs(r=wcfg.nn_radius_m, output_type="ndarray")
    cols = ["win", "vessel_a", "vessel_b", "n_2km", "n_close", "min_dist_m", "slow_close", "rel_speed",
            "parallel", "port_min_m"]
    if len(pr) == 0:
        return pd.DataFrame(columns=cols)
    i, j = pr[:, 0], pr[:, 1]
    d = np.hypot(x[i] - x[j], y[i] - y[j])
    v = c.vessel_id.values
    swap = v[i] > v[j]
    i, j = np.where(swap, j, i), np.where(swap, i, j)
    sog = c.sog.fillna(0).values
    cog = np.deg2rad(c.cog.fillna(0).values)
    moving = (sog[i] > 0.5) & (sog[j] > 0.5)
    df = pd.DataFrame({
        "win": c.step.values[i] // T, "vessel_a": v[i], "vessel_b": v[j], "d": d,
        "close": d <= wcfg.close_m,
        "slow_close": (d <= wcfg.close_m) & (sog[i] < wcfg.slow_kn) & (sog[j] < wcfg.slow_kn),
        "rel": np.abs(sog[i] - sog[j]),
        "par": np.where(moving, np.cos(cog[i] - cog[j]), 0.0),
        "port": np.minimum(c.port_distance_m.values[i], c.port_distance_m.values[j]),
    })
    e = (df.groupby(["win", "vessel_a", "vessel_b"])
         .agg(n_2km=("d", "size"), n_close=("close", "sum"), min_dist_m=("d", "min"),
              slow_close=("slow_close", "sum"), rel_speed=("rel", "mean"), parallel=("par", "mean"),
              port_min_m=("port", "min"))
         .reset_index())
    return e[e.n_2km >= 2].reset_index(drop=True)
