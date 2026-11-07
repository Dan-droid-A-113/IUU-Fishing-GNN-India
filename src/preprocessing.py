"""AIS preprocessing (brief section 8).

Steps: sentinel handling -> invalid coordinates -> duplicates -> sorting ->
impossible-speed flagging -> region assignment -> kinematics -> explicit AIS-gap
events with context -> geospatial features -> voyage segmentation ->
(optional) resampling that never interpolates across long gaps.

Long AIS gaps are kept as explicit missing-data events (`gaps` table). They are
never filled with interpolated "observations".
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import PreprocessConfig
from .india_layers import IndiaLayers, assign_region

EARTH_R = 6_371_008.8
MS_TO_KN = 1 / 0.514444


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def angle_diff(a, b):
    """Signed smallest difference b - a in degrees, in [-180, 180)."""
    return (np.asarray(b) - np.asarray(a) + 180) % 360 - 180


# --------------------------------------------------------------------------
# 1-6 cleaning
# --------------------------------------------------------------------------
def clean_ais(df: pd.DataFrame, cfg: PreprocessConfig = PreprocessConfig()) -> Tuple[pd.DataFrame, Dict]:
    rep = {"rows_in": int(len(df))}
    df = df.copy()
    # AIS "not available" sentinels -> NaN (kept as missing, not dropped)
    rep["sog_not_available"] = int((df.sog >= 102.2).sum())
    df.loc[df.sog >= 102.2, "sog"] = np.nan
    rep["cog_not_available"] = int((df.cog >= 360).sum())
    df.loc[df.cog >= 360, "cog"] = np.nan
    rep["heading_not_available"] = int((df.heading >= 360).sum())
    df.loc[df.heading >= 360, "heading"] = np.nan
    # invalid coordinates / timestamps
    bad = (df.latitude.isna() | df.longitude.isna() | df.timestamp.isna()
           | ~df.latitude.between(-90, 90) | ~df.longitude.between(-180, 180)
           | ((df.latitude == 0) & (df.longitude == 0)))
    rep["invalid_coordinates_removed"] = int(bad.sum())
    df = df[~bad]
    # exact duplicates (same vessel, same timestamp)
    n = len(df)
    df = df.drop_duplicates(subset=["vessel_id", "timestamp"], keep="first")
    rep["duplicates_removed"] = int(n - len(df))
    # sorting
    df = df.sort_values(["vessel_id", "timestamp"]).reset_index(drop=True)
    # impossible speed: a fix that implies > max speed from BOTH neighbours is a jump
    df["impossible_speed_flag"] = _jump_flags(df, cfg.max_speed_kn)
    rep["impossible_speed_flagged"] = int(df.impossible_speed_flag.sum())
    if cfg.drop_impossible:
        df = df[~df.impossible_speed_flag].reset_index(drop=True)
    # region assignment
    df["region_id"] = assign_region(df.latitude.values, df.longitude.values)
    rep["outside_regions_removed"] = int(df.region_id.isna().sum())
    df = df[df.region_id.notna()].reset_index(drop=True)
    rep["rows_out"] = int(len(df))
    return df, rep


def _jump_flags(df, vmax_kn):
    g = df.groupby("vessel_id", sort=False)
    lat, lon = df.latitude.values, df.longitude.values
    t = df.timestamp.values.astype("datetime64[s]").astype(np.int64).astype(float)
    same_prev = (df.vessel_id == g.vessel_id.shift(1)).values
    same_next = (df.vessel_id == g.vessel_id.shift(-1)).values
    def spd(i0, i1):
        d = haversine_m(lat[i0], lon[i0], lat[i1], lon[i1])
        dt = np.maximum(np.abs(t[i1] - t[i0]), 1.0)
        return d / dt * MS_TO_KN
    idx = np.arange(len(df))
    v_prev = np.zeros(len(df))
    v_next = np.zeros(len(df))
    ip = idx[same_prev]
    v_prev[ip] = spd(ip - 1, ip)
    inx = idx[same_next]
    v_next[inx] = spd(inx, inx + 1)
    dt_prev = np.full(len(df), np.inf)
    dt_next = np.full(len(df), np.inf)
    dt_prev[ip] = (t[ip] - t[ip - 1]) / 60
    dt_next[inx] = (t[inx + 1] - t[inx]) / 60
    # interior points: both sides too fast. Next to a silence (or at a track end) only
    # one neighbour is informative, so an extreme jump on that side is enough.
    both = (v_prev > vmax_kn) & (v_next > vmax_kn)
    after_silence = (dt_prev > 60) & (v_next > vmax_kn * 3)
    before_silence = (dt_next > 60) & (v_prev > vmax_kn * 3)
    return both | after_silence | before_silence


# --------------------------------------------------------------------------
# 7 kinematics + fishing-motion heuristic
# --------------------------------------------------------------------------
def add_kinematics(df: pd.DataFrame, cfg: PreprocessConfig = PreprocessConfig()) -> pd.DataFrame:
    df = df.sort_values(["vessel_id", "timestamp"]).reset_index(drop=True)
    g = df.groupby("vessel_id", sort=False)
    prev_lat, prev_lon = g.latitude.shift(), g.longitude.shift()
    dt = g.timestamp.diff().dt.total_seconds() / 60.0
    df["dt_minutes"] = dt
    df["reporting_interval"] = dt
    df["distance_m"] = haversine_m(prev_lat, prev_lon, df.latitude, df.longitude)
    df["implied_speed_kn"] = df.distance_m / (dt * 60).clip(lower=1) * MS_TO_KN
    df["acceleration"] = g.sog.diff() / dt.clip(lower=0.5)          # knots per minute
    dcog = pd.Series(angle_diff(g.cog.shift(), df.cog), index=df.index)
    df["turn_rate"] = dcog / dt.clip(lower=0.5)                     # degrees per minute
    df.loc[df.sog.fillna(0) < 0.5, "turn_rate"] = np.nan              # COG meaningless when stopped
    df["missingness_flag"] = (dt > cfg.gap_threshold_min).fillna(False)
    df["track_piece"] = df.missingness_flag.astype(int).groupby(df.vessel_id).cumsum()
    # fishing-motion score: share of fixes in the fishing speed band x turning intensity.
    lo, hi = cfg.fishing_speed_band_kn
    in_band = df.sog.between(lo, hi).astype(float)
    abs_turn = np.abs(dcog).where(df.sog.fillna(0) >= 0.5)
    keys = [df.vessel_id, df.track_piece]
    w = cfg.fishing_window
    band_m = in_band.groupby(keys).transform(lambda s: s.rolling(w, center=True, min_periods=3).mean())
    turn_m = abs_turn.groupby(keys).transform(lambda s: s.rolling(w, center=True, min_periods=3).mean())
    df["fishing_motion_score"] = (band_m * np.clip(turn_m.fillna(0) / 45.0, 0, 1)).fillna(0.0)
    return df


# --------------------------------------------------------------------------
# 8 geospatial features
# --------------------------------------------------------------------------
def add_geo_features(df: pd.DataFrame, layers: IndiaLayers) -> pd.DataFrame:
    f = layers.features(df.latitude.values, df.longitude.values)
    keep = ["in_mpa", "mpa_distance_m", "india_side", "eez_distance_m", "port_distance_m",
            "nearest_port", "coast_distance_m", "on_land", "fishing_ground", "region_edge_distance_m"]
    out = df.reset_index(drop=True).copy()
    for c in keep:
        out[c] = f[c].values
    # SAR evidence is unknown until a SAR layer is attached (Mode C, Milestone 5)
    out["sar_detection_flag"] = pd.array([pd.NA] * len(out), dtype="boolean")
    return out


# --------------------------------------------------------------------------
# 7b explicit AIS-gap events
# --------------------------------------------------------------------------
def detect_gaps(df: pd.DataFrame, cfg: PreprocessConfig = PreprocessConfig(),
                layers: Optional[IndiaLayers] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (messages with `ais_gap_id`/`gap_duration_min` on the reappearance fix, gaps table)."""
    df = df.sort_values(["vessel_id", "timestamp"]).reset_index(drop=True)
    is_gap = df.dt_minutes > cfg.gap_threshold_min
    end_idx = np.flatnonzero(is_gap.values)
    start_idx = end_idx - 1
    gaps = pd.DataFrame({
        "ais_gap_id": [f"gap_{i:05d}" for i in range(len(end_idx))],
        "vessel_id": df.vessel_id.values[end_idx],
        "vessel_type": df.vessel_type.values[end_idx],
        "region_id": df.region_id.values[end_idx],
        "gap_start": df.timestamp.values[start_idx],
        "gap_end": df.timestamp.values[end_idx],
        "gap_duration_min": df.dt_minutes.values[end_idx],
        "start_lat": df.latitude.values[start_idx], "start_lon": df.longitude.values[start_idx],
        "end_lat": df.latitude.values[end_idx], "end_lon": df.longitude.values[end_idx],
        "start_idx": start_idx, "end_idx": end_idx,
    })
    gaps["gap_start"] = pd.to_datetime(gaps.gap_start, utc=True)
    gaps["gap_end"] = pd.to_datetime(gaps.gap_end, utc=True)
    gaps["displacement_m"] = haversine_m(gaps.start_lat, gaps.start_lon, gaps.end_lat, gaps.end_lon)
    gaps["implied_speed_kn"] = gaps.displacement_m / (gaps.gap_duration_min * 60) * MS_TO_KN
    gaps["long_gap"] = gaps.gap_duration_min >= cfg.long_gap_min
    df["ais_gap_id"] = None
    df["gap_duration_min"] = np.nan
    df.loc[end_idx, "ais_gap_id"] = gaps.ais_gap_id.values
    df.loc[end_idx, "gap_duration_min"] = gaps.gap_duration_min.values
    if len(gaps):
        gaps = _pre_gap_kinematics(gaps, df)
        gaps = _concurrent_silence(gaps, df, cfg)
        if layers is not None:
            gaps = _gap_geo_context(gaps, layers, cfg)
    return df, gaps


def _pre_gap_kinematics(gaps, df, window_min=30):
    """Speed / course behaviour in the `window_min` before the vessel went silent."""
    sog_mean, sog_last, course_change, n_pre = [], [], [], []
    ts = df.timestamp.values
    for s in gaps.start_idx.values:
        vid = df.vessel_id.values[s]
        lo = s
        while lo - 1 >= 0 and df.vessel_id.values[lo - 1] == vid and \
                (ts[s] - ts[lo - 1]) <= np.timedelta64(window_min, "m"):
            lo -= 1
        seg = df.iloc[lo:s + 1]
        sog_mean.append(seg.sog.mean())
        sog_last.append(df.sog.values[s])
        c = seg.cog.dropna().values
        course_change.append(float(np.abs(angle_diff(c[0], c[-1]))) if len(c) >= 2 else np.nan)
        n_pre.append(len(seg))
    gaps["pre_gap_sog_mean"] = sog_mean
    gaps["pre_gap_sog_last"] = sog_last
    gaps["pre_gap_course_change_deg"] = course_change
    gaps["pre_gap_n_msgs"] = n_pre
    return gaps


def _concurrent_silence(gaps, df, cfg):
    """Data-driven reception indicator.

    Among other vessels heard within `coverage_radius_m` of the gap start in
    the hour before the gap, what fraction were ALSO silent during the gap?
    High values point to receiver/coverage problems rather than one vessel
    switching off. NaN when no neighbours were around (unknown).
    """
    order = np.argsort(df.timestamp.values)
    t_sorted = df.timestamp.values[order]
    lat, lon, vid = df.latitude.values[order], df.longitude.values[order], df.vessel_id.values[order]
    look = np.timedelta64(int(cfg.coverage_lookback_min), "m")
    frac, n_nb = [], []
    for g in gaps.itertuples(index=False):
        t0, t1 = np.datetime64(g.gap_start.tz_convert(None)), np.datetime64(g.gap_end.tz_convert(None))
        a, b = np.searchsorted(t_sorted, [t0 - look, t0])
        m = vid[a:b] != g.vessel_id
        d = haversine_m(g.start_lat, g.start_lon, lat[a:b][m], lon[a:b][m])
        nb = set(vid[a:b][m][d < cfg.coverage_radius_m])
        n_nb.append(len(nb))
        if not nb:
            frac.append(np.nan)
            continue
        # inner window avoids counting a neighbour's own last/first fix
        c, e = np.searchsorted(t_sorted, [t0 + np.timedelta64(10, "m"), t1 - np.timedelta64(10, "m")])
        heard = set(vid[c:e]) & nb if e > c else set()
        frac.append(1 - len(heard) / len(nb))
    gaps["n_neighbours_before"] = n_nb
    gaps["concurrent_silence_frac"] = frac
    return gaps


def _gap_geo_context(gaps, layers, cfg):
    fs = layers.features(gaps.start_lat.values, gaps.start_lon.values)
    fe = layers.features(gaps.end_lat.values, gaps.end_lon.values)
    for c in ("mpa_distance_m", "eez_distance_m", "port_distance_m", "region_edge_distance_m", "in_mpa"):
        gaps[f"start_{c}"] = fs[c].values
        gaps[f"end_{c}"] = fe[c].values
    gaps["start_near_port"] = fs.port_distance_m.values < cfg.port_radius_m
    gaps["end_near_port"] = fe.port_distance_m.values < cfg.port_radius_m
    gaps["start_near_edge"] = fs.region_edge_distance_m.values < cfg.region_edge_buffer_m
    gaps["end_near_edge"] = fe.region_edge_distance_m.values < cfg.region_edge_buffer_m
    gaps["crossed_boundary"] = fs.india_side.values != fe.india_side.values
    gaps["min_mpa_distance_m"] = np.minimum(fs.mpa_distance_m.values, fe.mpa_distance_m.values)
    gaps["min_abs_eez_distance_m"] = np.minimum(np.abs(fs.eez_distance_m.values), np.abs(fe.eez_distance_m.values))
    # context category (descriptive only - NOT a suspicion label)
    ctx = np.full(len(gaps), "at_sea", dtype=object)
    ctx[gaps.start_near_edge.values | gaps.end_near_edge.values] = "region_exit_entry"
    ctx[gaps.start_near_port.values & gaps.end_near_port.values] = "port_to_port"
    gaps["gap_context"] = ctx
    return gaps


# --------------------------------------------------------------------------
# voyage segmentation (used for leakage-safe splitting)
# --------------------------------------------------------------------------
def segment_voyages(df: pd.DataFrame, gaps: pd.DataFrame, max_gap_h: float = 12.0) -> pd.DataFrame:
    """A new segment starts after a port-to-port gap, a region exit/entry, or a very long gap."""
    df = df.copy()
    brk = np.zeros(len(df), bool)
    if len(gaps):
        cut = (gaps.gap_context.isin(["port_to_port", "region_exit_entry"]) |
               (gaps.gap_duration_min > max_gap_h * 60)) if "gap_context" in gaps else \
              (gaps.gap_duration_min > max_gap_h * 60)
        brk[gaps.end_idx.values[cut.values]] = True
    first = (df.vessel_id != df.vessel_id.shift()).values
    seg_no = pd.Series((brk | first).astype(int)).groupby(df.vessel_id.values).cumsum()
    df["segment_id"] = df.vessel_id + "_s" + seg_no.astype(str).str.zfill(3)
    return df


# --------------------------------------------------------------------------
# optional resampling (visualisation / regular-grid models)
# --------------------------------------------------------------------------
def resample_tracks(df: pd.DataFrame, cfg: PreprocessConfig = PreprocessConfig()) -> pd.DataFrame:
    """Linear resampling onto a regular grid ONLY where the surrounding real fixes are
    at most `max_interp_gap_min` apart. Grid times inside longer silences are
    omitted - blackout intervals stay explicit missing data."""
    freq = f"{cfg.resample_min}min"
    out = []
    for vid, g in df.groupby("vessel_id", sort=False):
        t = g.timestamp.values.astype("datetime64[s]").astype(np.int64)
        if len(t) < 2:
            continue
        grid = pd.date_range(g.timestamp.iloc[0].ceil(freq), g.timestamp.iloc[-1].floor(freq), freq=freq)
        if len(grid) == 0:
            continue
        tg = grid.values.astype("datetime64[s]").astype(np.int64)
        j = np.clip(np.searchsorted(t, tg, side="right"), 1, len(t) - 1)
        span_min = (t[j] - t[j - 1]) / 60.0
        exact = t[j - 1] == tg
        ok = (span_min <= cfg.max_interp_gap_min) | exact
        if not ok.any():
            continue
        tg, j, span_min = tg[ok], j[ok], span_min[ok]
        w = (tg - t[j - 1]) / np.maximum(t[j] - t[j - 1], 1)
        lat, lon = g.latitude.values, g.longitude.values
        sog = g.sog.values
        cog = g.cog.values
        near = np.where(w < 0.5, j - 1, j)
        out.append(pd.DataFrame({
            "vessel_id": vid, "timestamp": pd.to_datetime(tg, unit="s", utc=True),
            "latitude": lat[j - 1] + w * (lat[j] - lat[j - 1]),
            "longitude": lon[j - 1] + w * (lon[j] - lon[j - 1]),
            "sog": sog[j - 1] + w * (sog[j] - sog[j - 1]),
            "cog": cog[near],
            "vessel_type": g.vessel_type.iloc[0], "region_id": g.region_id.values[near],
            "is_interpolated": w > 0, "bracket_span_min": span_min,
        }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# --------------------------------------------------------------------------
# full pipeline
# --------------------------------------------------------------------------
def run_preprocessing(ais: pd.DataFrame, layers: IndiaLayers,
                      cfg: PreprocessConfig = PreprocessConfig()) -> Dict[str, object]:
    """ais: canonical, anonymised AIS. Returns dict(ais, gaps, report)."""
    clean, report = clean_ais(ais, cfg)
    # fixes far inland are impossible for vessels (e.g. corrupted positions after a silence)
    inland = layers.land.inland_distance_m(clean.latitude.values, clean.longitude.values)
    report["far_inland_removed"] = int((inland > 3000).sum())
    clean = clean[inland <= 3000].reset_index(drop=True)
    report["rows_out"] = int(len(clean))
    kin = add_kinematics(clean, cfg)
    geo = add_geo_features(kin, layers)
    msgs, gaps = detect_gaps(geo, cfg, layers)
    msgs = segment_voyages(msgs, gaps)
    report.update(n_vessels=int(msgs.vessel_id.nunique()), n_gaps=int(len(gaps)),
                  n_long_gaps=int(gaps.long_gap.sum()) if len(gaps) else 0,
                  n_segments=int(msgs.segment_id.nunique()),
                  positions_on_land=int(msgs.on_land.sum()))
    return {"ais": msgs, "gaps": gaps, "report": report}
