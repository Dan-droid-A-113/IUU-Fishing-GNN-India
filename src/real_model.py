"""Temporal graph model on REAL Global Fishing Watch data (notebook 04b).

There are no public "confirmed illegal fishing" labels for Indian waters, so the
model is trained on a task that real data *can* supervise honestly:

    forecast what a vessel does in its NEXT week from everything seen up to now.

Targets (week t+1, same vessel, real GFW events):
    AIS-off gap (1 h - 30 d), encounter, encounter with a carrier, loitering,
    fishing inside an MPA, fishing inside a non-Indian EEZ.

The *investigation priority* is a fixed, documented combination (noisy-OR) of the
calibrated forecast probabilities - an analyst's weighting, not a learned
"illegality" label.

Held-out evidence (never used as model input or target) is used to check whether
the priority ranks the right vessel-weeks higher:
    * GFW's own `intentional_disabling` flag on next-week gaps,
    * unmatched Sentinel-1 SAR detections within 20 km of a next-week gap,
    * unmatched Sentinel-1 SAR detections within 5 km of the vessel's next-week
      fishing positions on the same day (unidentified vessels operating alongside).
None of these is proof of wrongdoing.

Graph
    node  = vessel x ISO week (fishing, carrier, support and bunker vessels, plus
            any vessel that met one of them); features = cleaned weekly aggregates.
    edges = temporal   : the vessel's previous 1-3 active weeks (past -> present only)
            encounter  : real GFW encounters in the same week (both directions)
            co-location: up to k random vessels active in the same 0.25 deg cell that week
Model   = MLP node encoder + 2 GATv2 layers with edge features and a learned
          time encoding (lag in weeks) + one multi-task forecasting head.
Ablations: temporal edges only, and no graph at all (same encoder and head).
"""
from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import PROCESSED_DIR
from .temporal_gnn import TimeEncoding

try:
    from torch_geometric.nn import GATv2Conv
except ImportError:  # pragma: no cover
    GATv2Conv = None

DRIVE_REAL_DIR = Path("/content/drive/MyDrive/india-fishing-gnn-data/real_gfw")
LOCAL_REAL_DIR = PROCESSED_DIR / "real_gfw"

REAL_MODEL_CAVEAT = (
    "REAL DATA (Global Fishing Watch). Scores are an investigation PRIORITY derived from forecasts of "
    "next-week activity patterns. They are not evidence of illegal fishing; every alert needs human "
    "verification. Vessel identities are anonymised.")

TASKS = {
    "gap": "AIS-off gap (1 h to 30 days)",
    "encounter": "encounter with any vessel",
    "carrier_encounter": "encounter with a carrier",
    "loitering": "loitering",
    "fishing_in_mpa": "fishing inside a marine protected area",
    "fishing_outside_flag_eez": "fishing in an EEZ that is not its flag state's",
}
EVIDENCE = {
    "intentional_gap": "gap flagged as possible intentional AIS disabling by GFW",
    "sar_near_gap": "unmatched SAR detection within 20 km of a gap",
    "sar_near_fishing": "unmatched SAR detection within 5 km of its fishing, same day (imaged areas only)",
}
# evidence that can only be observed in part of the data: evaluate only there
EVIDENCE_MASK = {"sar_near_fishing": "m_sar_imaged"}
VESSEL_CLASSES = ["fishing", "carrier", "support", "bunker", "cargo_tanker", "passenger", "other", "unknown"]

# weekly aggregates used as model inputs (sign-log transformed, then standardised)
COUNT_FEATURES = [
    "n_events", "n_fishing", "fishing_h", "fishing_mpa_h", "fishing_foreign_h", "fishing_outside_flag_h",
    "n_gaps", "gap_h", "n_long_gaps", "n_extreme_gaps", "max_gap_km", "max_implied_speed_kn", "min_sat_ppd",
    "n_loiter", "loiter_h", "n_enc", "n_carrier_enc", "n_fishing_enc", "enc_min_dist_km",
    "n_port", "port_h", "n_in_mpa", "dist_shore_km", "dist_port_km", "weeks_since_prev",
]
EDGE_COLS = ["is_temporal", "is_encounter", "is_coloc", "enc_log_dist", "enc_log_dur", "enc_speed",
             "src_is_fishing", "src_is_carrier"]
TARGET_SOURCE = {"gap": "n_gaps", "encounter": "n_enc", "carrier_encounter": "n_carrier_enc",
                 "loitering": "n_loiter", "fishing_in_mpa": "fishing_mpa_h", "fishing_outside_flag_eez": "fishing_outside_flag_h"}


@dataclass
class RealConfig:
    # cleaning
    study_start: Optional[str] = None  # drop events that started before the study period (old port visits)
    min_gap_h: float = 1.0
    max_gap_h: float = 720.0          # gaps longer than 30 days = vessel out of area / inactive, not a gap
    long_gap_h: float = 168.0
    max_event_h: float = 168.0        # clip single-event durations to one week
    keep_classes: tuple = ("fishing", "carrier", "support", "bunker")
    # graph
    cell_deg: float = 0.25
    coloc_k: int = 3
    max_coloc_edges: int = 1_500_000   # memory guard (random subsample above this)
    max_total_edges: int = 4_500_000   # overall GPU-memory budget; co-location edges shrink to fit (kept >= min_coloc_edges)
    min_coloc_edges: int = 600_000
    temporal_lags: int = 3
    max_lag_weeks: int = 8
    # time split (Mondays); a one-week embargo is applied automatically
    val_start: str = "2025-04-07"
    test_start: str = "2025-07-07"
    # model
    hidden: int = 64
    heads: int = 4
    layers: int = 2
    dropout: float = 0.2
    lr: float = 2e-3
    weight_decay: float = 1e-4
    epochs: int = 150
    patience: int = 15
    seeds: tuple = (0, 1, 2)
    min_train_pos: int = 30
    min_eval_pos: int = 5
    priority_weights: Dict[str, float] = field(default_factory=lambda: {
        # loitering has weight 0: it is ordinary behaviour for fishing vessels (about 60% of weeks)
        "gap": 1.0, "carrier_encounter": 1.0, "fishing_in_mpa": 1.0, "fishing_outside_flag_eez": 1.0,
        "encounter": 0.5, "loitering": 0.0})
    sar_radius_km: float = 5.0


# ==========================================================================
# Loading and cleaning
# ==========================================================================
def find_real_dir(real_dir: Optional[Path] = None) -> Path:
    for p in ([Path(real_dir)] if real_dir else []) + [DRIVE_REAL_DIR, LOCAL_REAL_DIR]:
        if (p / "events.parquet").exists():
            return p
    raise FileNotFoundError(
        "events.parquet not found. Mount Google Drive (it should be in MyDrive/india-fishing-gnn-data/real_gfw) "
        "or run notebook 02 first.")


def load_real(real_dir: Optional[Path] = None):
    d = find_real_dir(real_dir)
    ev = pd.read_parquet(d / "events.parquet")
    sar = pd.read_parquet(d / "sar_unmatched.parquet") if (d / "sar_unmatched.parquet").exists() else pd.DataFrame()
    for c in ("start", "end"):
        ev[c] = pd.to_datetime(ev[c], utc=True, errors="coerce")
    return ev, sar, d


def vessel_class(t) -> str:
    s = str(t).lower().strip()
    if s in ("", "nan", "none", "<na>", "unknown"):
        return "unknown"
    if "fish" in s or s in ("trawlers", "squid_jigger", "tuna_purse_seines", "set_longlines", "drifting_longlines"):
        return "fishing"
    for k, v in (("carrier", "carrier"), ("reefer", "carrier"), ("support", "support"), ("bunker", "bunker"),
                 ("cargo", "cargo_tanker"), ("tanker", "cargo_tanker"), ("passenger", "passenger")):
        if k in s:
            return v
    return "other"


def classify(types: pd.Series) -> pd.Series:
    u = types.astype(str).unique()
    return types.astype(str).map({t: vessel_class(t) for t in u})


def _num(ev, col, default=np.nan):
    return pd.to_numeric(ev[col], errors="coerce") if col in ev else pd.Series(default, index=ev.index, dtype=float)


def _flag(ev, col):
    if col not in ev:
        return pd.Series(False, index=ev.index)
    return ev[col].astype("boolean").fillna(False).astype(bool)


def clean_events(ev: pd.DataFrame, cfg: RealConfig):
    """Cap durations, classify gaps (valid / long / extreme), mirror encounters to the partner side."""
    ev = ev.dropna(subset=["start"]).copy()
    rep = {"events_in": len(ev)}
    if cfg.study_start:
        old = ev.start < pd.Timestamp(cfg.study_start, tz="UTC")
        rep["events starting before the study period (dropped)"] = int(old.sum())
        ev = ev[~old].copy()
    ev["vclass"] = classify(ev.vessel_type)
    ev["dur_h"] = _num(ev, "duration_h").clip(0, cfg.max_event_h).fillna(0)
    is_gap = ev.event_type.eq("gap")
    gh = _num(ev, "gap_duration_h").fillna(_num(ev, "duration_h"))
    ev["gap_h"] = np.where(is_gap, gh, np.nan)
    ev["gap_valid"] = is_gap & gh.between(cfg.min_gap_h, cfg.max_gap_h)
    ev["gap_long"] = ev.gap_valid & (gh > cfg.long_gap_h)
    ev["gap_extreme"] = is_gap & (gh > cfg.max_gap_h)
    ev["gap_km"] = _num(ev, "gap_distance_km").clip(0, 3000)
    ev["implied_kn"] = _num(ev, "implied_speed_kn").clip(0, 40)
    ev["sat_ppd"] = _num(ev, "sat_positions_per_day").clip(0, 500)
    rep.update({"gaps": int(is_gap.sum()), "gaps_valid_1h_30d": int(ev.gap_valid.sum()),
                "gaps_longer_than_30d (not used as gaps)": int(ev.gap_extreme.sum()),
                "gaps_shorter_than_min": int((is_gap & (gh < cfg.min_gap_h)).sum()),
                "max_raw_gap_h": float(np.nanmax(gh[is_gap])) if is_gap.any() else 0.0})
    names = ev["eez_names"].fillna("").astype(str) if "eez_names" in ev else pd.Series("", index=ev.index)
    ev["foreign_eez"] = names.str.len().gt(0) & ~names.str.lower().str.contains("india")
    ev["outside_flag_eez"], ev["flag_known"] = outside_flag_eez(ev.get("flag", pd.Series("", index=ev.index)), names)
    rep["fishing events with known flag"] = int((ev.event_type.eq("fishing") & ev.flag_known).sum())
    ev["in_mpa"] = _flag(ev, "in_mpa")
    # encounters: make sure each participating vessel has its own row
    if "partner_id" in ev:
        enc = ev[ev.event_type.eq("encounter") & ev.partner_id.fillna("").ne("")]
        mir = enc.copy()
        mir["vessel_id"], mir["partner_id"] = enc.partner_id.values, enc.vessel_id.values
        if "partner_type" in enc:
            mir["vessel_type"], mir["partner_type"] = enc.partner_type.values, enc.vessel_type.values
        if "partner_flag" in enc:
            mir["flag"], mir["partner_flag"] = enc.partner_flag.values, enc.flag.values
        mir["vclass"] = classify(mir.vessel_type)
        ev = pd.concat([ev, mir], ignore_index=True)
        key = ev.vessel_id.astype(str) + "|" + ev.partner_id.fillna("").astype(str) + "|" + \
            ev.start.dt.floor("h").astype(str) + "|" + ev.event_type
        dup = ev.event_type.eq("encounter") & key.duplicated()
        ev = ev[~dup].reset_index(drop=True)
        rep["encounter_rows_after_mirroring"] = int(ev.event_type.eq("encounter").sum())
    ev["week"] = ev.start.dt.tz_convert(None).dt.to_period("W-SUN").dt.start_time
    return ev, rep


# coastal states whose EEZs fall in the study area -> words that identify their EEZ names
FLAG_EEZ_WORDS = {"IND": ("india",), "LKA": ("sri lank",), "MDV": ("maldiv",)}


def outside_flag_eez(flag: pd.Series, eez_names: pd.Series):
    """True when a vessel is inside an EEZ that does not belong to its flag state.

    Indian boats in Sri Lankan waters, Sri Lankan boats in Indian waters and distant-water
    fleets in either count; a Sri Lankan boat in Sri Lankan waters does not. Unknown flags
    are never counted (returned as flag_known = False). High seas (no EEZ) never count."""
    f = flag.fillna("").astype(str).str.upper().str.strip()
    known = ~f.isin(["", "NAN", "NONE", "<NA>", "UNK"])
    n = eez_names.fillna("").astype(str).str.lower().str.replace("indian ocean", "", regex=False)
    in_eez = n.str.contains("[a-z]", regex=True)       # numeric ids = name lookup failed -> unknown
    own = np.zeros(len(f), bool)
    for code, words in FLAG_EEZ_WORDS.items():
        m = f.eq(code).values
        if m.any():
            own[m] = [any(w in s for w in words) for s in n.values[m]]
    return known.values & in_eez.values & ~own, known.values


def add_sar_evidence(ev: pd.DataFrame, sar: pd.DataFrame, radius_km: float = 5.0,
                     sar_all: Optional[pd.DataFrame] = None, cover_km: float = 25.0) -> pd.DataFrame:
    """Held-out radar evidence for each fishing event (never a model input).

    sar_near_event : unmatched (no-AIS) SAR detections within `radius_km`, same UTC day.
    sar_imaged     : the satellite demonstrably imaged the spot that day, i.e. there is at
                     least one SAR detection of ANY vessel (matched or not) within `cover_km`.
                     Sentinel-1 does not image all open ocean every day, so "no dark detection"
                     only means something where an image was taken. Without `sar_all` the
                     footprint is unknown and every event is treated as imaged (old behaviour)."""
    ev = ev.copy()
    ev["sar_near_event"] = 0.0
    ev["sar_imaged"] = sar_all is None
    if sar is None or not len(sar):
        return ev
    from scipy.spatial import cKDTree
    R = 6371.0088

    def xyz(lat, lon, day):
        la, lo = np.radians(lat), np.radians(lon)
        return np.column_stack([R * np.cos(la) * np.cos(lo), R * np.cos(la) * np.sin(lo), R * np.sin(la), day * 1e5])

    s = sar.dropna(subset=["lat", "lon", "date"])
    sday = (pd.to_datetime(s.date, utc=True).dt.floor("D") - pd.Timestamp("2000-01-01", tz="UTC")).dt.days.values
    tree = cKDTree(xyz(s.lat.values, s.lon.values, sday))
    w = np.asarray(pd.to_numeric(s.detections, errors="coerce").fillna(1).values, float)
    m = ev.event_type.eq("fishing") & ev.lat.notna() & ev.lon.notna()
    e = ev[m]
    eday = (e.start.dt.floor("D") - pd.Timestamp("2000-01-01", tz="UTC")).dt.days.values
    hits = tree.query_ball_point(xyz(e.lat.values, e.lon.values, eday), r=radius_km)
    ev.loc[m, "sar_near_event"] = [w[h].sum() if len(h) else 0.0 for h in hits]
    if sar_all is not None and len(sar_all):
        a = sar_all.dropna(subset=["lat", "lon", "date"])
        aday = (pd.to_datetime(a.date, utc=True).dt.floor("D") - pd.Timestamp("2000-01-01", tz="UTC")).dt.days.values
        cov = cKDTree(xyz(a.lat.values, a.lon.values, aday))
        n = cov.query_ball_point(xyz(e.lat.values, e.lon.values, eday), r=cover_km, return_length=True)
        ev.loc[m, "sar_imaged"] = (np.asarray(n) > 0) | (ev.loc[m, "sar_near_event"].values > 0)
    return ev


def select_vessels(ev: pd.DataFrame, cfg: RealConfig):
    """Fishing-related vessels + everything that met one of them. Pure merchant traffic is dropped."""
    known = ev[ev.vclass != "unknown"]
    vc = known.groupby("vessel_id").vclass.agg(lambda s: s.mode().iloc[0])
    vc = vc.reindex(ev.vessel_id.unique()).fillna("unknown")
    fishers = set(ev.loc[ev.event_type.eq("fishing"), "vessel_id"])
    core = set(vc.index[vc.isin(cfg.keep_classes)]) | fishers
    partners = set()
    if "partner_id" in ev:
        e = ev[ev.event_type.eq("encounter") & ev.vessel_id.isin(core)]
        partners = set(e.partner_id.dropna()) - {""}
    keep = core | partners
    tab = pd.DataFrame({"class": vc, "kept": vc.index.isin(keep),
                        "reason": np.where(vc.index.isin(core), "fishing-related",
                                           np.where(vc.index.isin(partners), "met a fishing-related vessel",
                                                    "dropped (no fishing link)"))})
    summary = tab.groupby(["class", "reason"]).size().unstack(fill_value=0)
    return keep, vc, summary


# ==========================================================================
# Vessel-week table (inputs, next-week targets, held-out evidence)
# ==========================================================================
def weekly_table(ev: pd.DataFrame, vclass: pd.Series, cfg: RealConfig) -> pd.DataFrame:
    et = ev.event_type
    fish, gap, loit, enc, port = (et.eq(k) for k in ("fishing", "gap", "loitering", "encounter", "port_visit"))
    pcls = ev.partner_id.map(vclass).fillna(classify(ev.get("partner_type", pd.Series("", index=ev.index)))) \
        if "partner_id" in ev else pd.Series("unknown", index=ev.index)
    loiter_h = _num(ev, "loiter_hours").fillna(ev.dur_h).clip(0, cfg.max_event_h)
    port_h = _num(ev, "port_duration_h").fillna(ev.dur_h).clip(0, cfg.max_event_h)
    sar = _num(ev, "sar_dark_nearby").fillna(0)
    h = pd.DataFrame({
        "vessel_id": ev.vessel_id, "week": ev.week,
        "n_events": 1.0,
        "n_fishing": fish.astype(float), "fishing_h": ev.dur_h.where(fish, 0),
        "fishing_mpa_h": ev.dur_h.where(fish & ev.in_mpa, 0), "fishing_foreign_h": ev.dur_h.where(fish & ev.foreign_eez, 0),
        "fishing_outside_flag_h": ev.dur_h.where(fish & ev.outside_flag_eez, 0),
        "n_gaps": ev.gap_valid.astype(float), "gap_h": ev.gap_h.where(ev.gap_valid, 0),
        "n_long_gaps": ev.gap_long.astype(float), "n_extreme_gaps": ev.gap_extreme.astype(float),
        "max_gap_km": ev.gap_km.where(ev.gap_valid), "max_implied_speed_kn": ev.implied_kn.where(ev.gap_valid),
        "min_sat_ppd": ev.sat_ppd.where(ev.gap_valid),
        "n_loiter": loit.astype(float), "loiter_h": loiter_h.where(loit, 0),
        "n_enc": enc.astype(float), "n_carrier_enc": (enc & pcls.eq("carrier")).astype(float),
        "n_fishing_enc": (enc & pcls.eq("fishing")).astype(float),
        "enc_min_dist_km": _num(ev, "median_distance_km").where(enc),
        "n_port": port.astype(float), "port_h": port_h.where(port, 0),
        "n_in_mpa": ev.in_mpa.astype(float),
        "dist_shore_km": _num(ev, "dist_shore_km").clip(0, 2000), "dist_port_km": _num(ev, "dist_port_km").clip(0, 3000),
        "lat": _num(ev, "lat"), "lon": _num(ev, "lon"),
        # held-out evidence (never model inputs)
        "ev_intentional_gap": (ev.gap_valid & _flag(ev, "intentional_disabling")).astype(float),
        "ev_sar_near_gap": (ev.gap_valid & (sar > 0)).astype(float),
        "ev_sar_near_fishing": (fish & _flag(ev, "sar_imaged") & (_num(ev, "sar_near_event").fillna(0) > 0)).astype(float),
        "ev_sar_imaged": (fish & _flag(ev, "sar_imaged")).astype(float),
    })
    agg = {c: "sum" for c in h.columns if c not in ("vessel_id", "week")}
    agg.update({"max_gap_km": "max", "max_implied_speed_kn": "max", "min_sat_ppd": "min", "enc_min_dist_km": "min",
                "dist_shore_km": "mean", "dist_port_km": "mean", "lat": "mean", "lon": "mean"})
    for c in [c for c in h.columns if c.startswith("ev_")]:
        agg[c] = "max"
    nodes = h.groupby(["vessel_id", "week"], sort=True).agg(agg).reset_index()
    nodes["vessel_class"] = nodes.vessel_id.map(vclass).fillna("unknown")
    prev = nodes.groupby("vessel_id").week.shift()
    nodes["weeks_since_prev"] = ((nodes.week - prev).dt.days / 7).fillna(52).clip(0, 52)
    # next-week targets and evidence (only defined when the vessel is active in week t+1)
    src = list(TARGET_SOURCE.values()) + [f"ev_{k}" for k in EVIDENCE] + ["ev_sar_imaged"]
    nxt = nodes[["vessel_id", "week"] + src].copy()
    nxt["week"] = nxt.week - pd.Timedelta(days=7)
    nxt.columns = ["vessel_id", "week"] + [f"next_{c}" for c in src]
    nodes = nodes.merge(nxt, on=["vessel_id", "week"], how="left")
    nodes["has_next"] = nodes[f"next_{src[0]}"].notna()
    for t, c in TARGET_SOURCE.items():
        nodes[f"y_{t}"] = (nodes[f"next_{c}"] > 0).astype(float)
    for k in EVIDENCE:
        nodes[f"e_{k}"] = (nodes[f"next_ev_{k}"] > 0).astype(float)
    # where each piece of evidence could have been observed at all
    nodes["m_sar_imaged"] = nodes.has_next & (nodes["next_ev_sar_imaged"] > 0)
    nodes = nodes.drop(columns=[f"next_{c}" for c in src])
    return nodes.reset_index(drop=True)


# ==========================================================================
# Graph
# ==========================================================================
def build_real_graph(nodes: pd.DataFrame, ev: pd.DataFrame, cfg: RealConfig, seed: int = 0) -> Dict:
    n = len(nodes)
    key = pd.Series(np.arange(n), index=pd.MultiIndex.from_frame(nodes[["vessel_id", "week"]]))
    cls = nodes.vessel_class.values
    parts = []
    # (1) temporal edges: previous active weeks of the same vessel (past -> present)
    order = np.arange(n)
    pos_in_vessel = nodes.groupby("vessel_id").cumcount().values     # nodes are sorted by vessel, week
    for lag in range(1, cfg.temporal_lags + 1):
        src = order - lag
        ok = pos_in_vessel >= lag
        dw = np.zeros(n)
        dw[ok] = (nodes.week.values[ok] - nodes.week.values[src[ok]]) / np.timedelta64(7, "D")
        ok &= (dw <= cfg.max_lag_weeks)
        e = pd.DataFrame({"src": src[ok], "dst": order[ok], "lag_w": dw[ok], "is_temporal": 1.0})
        parts.append(e)
    # (2) encounter edges (same week, both directions already present after mirroring)
    if "partner_id" in ev:
        enc = ev[ev.event_type.eq("encounter") & ev.partner_id.fillna("").ne("")]
        s = key.reindex(pd.MultiIndex.from_arrays([enc.partner_id, enc.week])).values
        d = key.reindex(pd.MultiIndex.from_arrays([enc.vessel_id, enc.week])).values
        ok = ~(np.isnan(s) | np.isnan(d))
        if ok.any():
            e = pd.DataFrame({"src": s[ok].astype(int), "dst": d[ok].astype(int),
                              "dist": _num(enc, "median_distance_km").values[ok],
                              "dur": enc.dur_h.values[ok], "spd": _num(enc, "median_speed_kn").values[ok]})
            e = e.groupby(["src", "dst"]).agg(dist=("dist", "min"), dur=("dur", "sum"), spd=("spd", "mean")).reset_index()
            e = e[e.src != e.dst]
            parts.append(pd.DataFrame({"src": e.src, "dst": e.dst, "lag_w": 0.0, "is_encounter": 1.0,
                                       "enc_log_dist": np.log1p(e.dist.fillna(0.5).clip(0, 50)),
                                       "enc_log_dur": np.log1p(e.dur.fillna(0)) / 3,
                                       "enc_speed": (e.spd.fillna(0) / 10).clip(0, 2)}))
    # (3) co-location edges: k random partners in the same cell and week
    ok = nodes.lat.notna().values & nodes.lon.notna().values
    idx = np.flatnonzero(ok)
    if len(idx) > 1 and cfg.coloc_k > 0:
        cell = (np.floor(nodes.lat.values[idx] / cfg.cell_deg).astype(np.int64) * 100000
                + np.floor(nodes.lon.values[idx] / cfg.cell_deg).astype(np.int64))
        grp = pd.factorize(pd.Series(cell).astype(str) + "|" + nodes.week.values[idx].astype(str))[0]
        rng = np.random.default_rng(seed)
        perm = np.lexsort((rng.random(len(idx)), grp))
        g_sorted, i_sorted = grp[perm], idx[perm]
        starts = np.r_[0, np.flatnonzero(np.diff(g_sorted)) + 1]
        sizes = np.diff(np.r_[starts, len(g_sorted)])
        size_of = np.repeat(sizes, sizes)
        start_of = np.repeat(starts, sizes)
        rank = np.arange(len(g_sorted)) - start_of
        src_l, dst_l = [], []
        for off in range(1, cfg.coloc_k + 1):
            m = size_of > off
            j = start_of[m] + (rank[m] + off) % size_of[m]
            src_l += [i_sorted[m], i_sorted[j]]
            dst_l += [i_sorted[j], i_sorted[m]]
        if src_l:
            e = pd.DataFrame({"src": np.concatenate(src_l), "dst": np.concatenate(dst_l)}).drop_duplicates()
            used = int(sum(len(p_) for p_ in parts))
            cap = min(cfg.max_coloc_edges, max(cfg.min_coloc_edges, cfg.max_total_edges - used))
            if len(e) > cap:
                e = e.sample(cap, random_state=seed)
            parts.append(e.assign(lag_w=0.0, is_coloc=1.0))
    E = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["src", "dst", "lag_w"])
    for c in EDGE_COLS:
        if c not in E:
            E[c] = 0.0
    E[EDGE_COLS] = E[EDGE_COLS].fillna(0.0)
    E["src_is_fishing"] = (cls[E.src.values.astype(int)] == "fishing").astype(float)
    E["src_is_carrier"] = (cls[E.src.values.astype(int)] == "carrier").astype(float)
    etype = np.select([E.is_temporal.values > 0, E.is_encounter.values > 0], [0, 1], 2)
    return {
        "nodes": nodes, "n": n,
        "edge_index": np.vstack([E.src.values, E.dst.values]).astype(np.int64),
        "edge_attr": E[EDGE_COLS].values.astype(np.float32),
        "edge_lag": E.lag_w.values.astype(np.float32),
        "edge_type": etype.astype(np.int64),      # 0 temporal, 1 encounter, 2 co-location
        "X_raw": node_features(nodes),
        "Y": nodes[[f"y_{t}" for t in TASKS]].values.astype(np.float32),
        "M": np.repeat(nodes.has_next.values[:, None], len(TASKS), 1),
        "E": nodes[[f"e_{k}" for k in EVIDENCE]].values.astype(np.float32),
        "EM": np.column_stack([nodes[EVIDENCE_MASK[k]].values.astype(bool) if k in EVIDENCE_MASK
                               else nodes.has_next.values for k in EVIDENCE]),
    }


def node_features(nodes: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=nodes.index)
    for c in COUNT_FEATURES:
        v = nodes[c].astype(float).fillna(0)
        X[c] = np.sign(v) * np.log1p(np.abs(v))
        if nodes[c].isna().any():
            X[c + "_missing"] = nodes[c].isna().astype(float)
    X["lat"] = nodes.lat.fillna(nodes.lat.median()) if nodes.lat.notna().any() else 0.0
    X["lon"] = nodes.lon.fillna(nodes.lon.median()) if nodes.lon.notna().any() else 0.0
    doy = nodes.week.dt.dayofyear.values / 365.25 * 2 * np.pi
    X["season_sin"], X["season_cos"] = np.sin(doy), np.cos(doy)
    for c in VESSEL_CLASSES:
        X[f"class_{c}"] = (nodes.vessel_class == c).astype(float)
    return X


def graph_summary(G: Dict) -> pd.Series:
    et = G["edge_type"]
    nd = G["nodes"]
    return pd.Series({
        "nodes (vessel-weeks)": G["n"], "vessels": nd.vessel_id.nunique(),
        "fishing-vessel nodes": int((nd.vessel_class == "fishing").sum()),
        "nodes with a next week (have targets)": int(nd.has_next.sum()),
        "temporal edges": int((et == 0).sum()), "encounter edges (directed)": int((et == 1).sum()),
        "co-location edges (directed)": int((et == 2).sum()), "node features": G["X_raw"].shape[1],
    })


# ==========================================================================
# Splits (leakage-safe)
# ==========================================================================
def time_split(nodes: pd.DataFrame, cfg: RealConfig):
    """Train on the past, validate, test on the future. A node's target is its next week,
    so nodes whose target week crosses a boundary are embargoed (no loss, no evaluation)."""
    w = nodes.week.values
    nw = w + np.timedelta64(7, "D")
    v0, t0 = np.datetime64(pd.Timestamp(cfg.val_start)), np.datetime64(pd.Timestamp(cfg.test_start))
    split = np.full(len(nodes), "embargo", dtype=object)
    split[nw < v0] = "train"
    split[(w >= v0) & (nw < t0)] = "val"
    split[w >= t0] = "test"
    graph_train = w < v0                                   # nodes allowed in the training subgraph
    return split, graph_train


def vessel_split(nodes: pd.DataFrame, fracs=(0.70, 0.15), salt: str = "split"):
    u = nodes.vessel_id.astype(str).map(lambda v: int(hashlib.sha256((salt + v).encode()).hexdigest()[:8], 16) / 2**32)
    split = np.where(u < fracs[0], "train", np.where(u < fracs[0] + fracs[1], "val", "test")).astype(object)
    return split, split == "train"


def cross_split_edges(G: Dict, split: np.ndarray) -> pd.Series:
    s, d = G["edge_index"]
    et = G["edge_type"]
    cross = split[s] != split[d]
    return pd.Series({name: int((cross & (et == k)).sum()) for k, name in
                      ((1, "encounter edges crossing splits"), (2, "co-location edges crossing splits"))})


# ==========================================================================
# Model
# ==========================================================================
class RealGraphNet(nn.Module):
    MODES = ("full", "temporal_only", "no_graph")

    def __init__(self, n_in: int, n_edge: int, n_tasks: int, cfg: RealConfig, mode: str = "full"):
        super().__init__()
        assert mode in self.MODES
        if mode != "no_graph" and GATv2Conv is None:
            raise ImportError("torch_geometric is required: pip install torch_geometric")
        H = cfg.hidden
        self.mode = mode
        self.enc = nn.Sequential(nn.Linear(n_in, H), nn.LayerNorm(H), nn.GELU(), nn.Dropout(cfg.dropout),
                                 nn.Linear(H, H), nn.LayerNorm(H))
        self.tenc = TimeEncoding(8)
        self.convs = nn.ModuleList()
        if mode != "no_graph":
            self.convs = nn.ModuleList([GATv2Conv(H, H // cfg.heads, heads=cfg.heads, concat=True, edge_dim=n_edge + 8,
                                                  dropout=cfg.dropout, add_self_loops=True, fill_value="mean")
                                        for _ in range(cfg.layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(H) for _ in self.convs])
        self.drop = nn.Dropout(cfg.dropout)
        self.head = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Dropout(cfg.dropout), nn.Linear(H, n_tasks))

    def forward(self, x, ei, ea, lag, return_attention: bool = False):
        h = self.enc(x)
        att = []
        if len(self.convs) and ei.shape[1]:
            e = torch.cat([ea, self.tenc(lag)], -1)
            for conv, norm in zip(self.convs, self.norms):
                if return_attention:
                    m, a = conv(h, ei, e, return_attention_weights=True)
                    att.append(a)
                else:
                    m = conv(h, ei, e)
                h = norm(h + self.drop(Fn.elu(m)))
        out = self.head(h)
        return (out, h, att) if return_attention else out


def standardize(X: pd.DataFrame, rows: np.ndarray):
    mu = X.values[rows].mean(0)
    sd = np.maximum(X.values[rows].std(0), 0.05)
    Z = np.clip((X.values - mu) / sd, -10, 10).astype(np.float32)
    return Z, {"mean": mu, "sd": sd, "columns": list(X.columns)}


def _edge_mask(G, mode):
    et = G["edge_type"]
    if mode == "full":
        return np.ones(len(et), bool)
    if mode == "temporal_only":
        return et == 0
    return np.zeros(len(et), bool)


def active_tasks(G: Dict, split: np.ndarray, cfg: RealConfig):
    tr, va = split == "train", split == "val"
    keep = []
    for k, t in enumerate(TASKS):
        ptr = G["Y"][tr & G["M"][:, k], k].sum()
        pva = G["Y"][va & G["M"][:, k], k].sum()
        if ptr >= cfg.min_train_pos and pva >= cfg.min_eval_pos:
            keep.append(t)
    return keep


def _set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _macro_ap(P, Y, M, rows, cols):
    aps = []
    for j, k in enumerate(cols):
        r = rows & M[:, k]
        if Y[r, k].sum() > 0 and Y[r, k].sum() < r.sum():
            aps.append(average_precision_score(Y[r, k], P[r, j]))
    return float(np.mean(aps)) if aps else float("nan")


def train_real(G: Dict, split: np.ndarray, graph_train: np.ndarray, cfg: RealConfig, seed: int, device,
               mode: str = "full", tasks: Optional[Sequence[str]] = None, verbose: bool = False) -> Dict:
    _set_seed(seed)
    tasks = list(tasks or TASKS)
    cols = [list(TASKS).index(t) for t in tasks]
    tr = split == "train"
    Z, stats = standardize(G["X_raw"], tr)
    x = torch.as_tensor(Z, device=device)
    em = _edge_mask(G, mode)
    ei_all = torch.as_tensor(G["edge_index"][:, em], device=device)
    ea_all = torch.as_tensor(G["edge_attr"][em], device=device)
    lag_all = torch.as_tensor(G["edge_lag"][em], device=device)
    gt = torch.as_tensor(graph_train, device=device)
    keep = gt[ei_all[0]] & gt[ei_all[1]]                  # inductive: training subgraph only
    ei_tr, ea_tr, lag_tr = ei_all[:, keep], ea_all[keep], lag_all[keep]
    Y = torch.as_tensor(G["Y"][:, cols], device=device)
    Mtr = torch.as_tensor(G["M"][:, cols] & tr[:, None], device=device)
    p = (G["Y"][:, cols] * (G["M"][:, cols] & tr[:, None])).sum(0)
    nn_ = (G["M"][:, cols] & tr[:, None]).sum(0)
    pw = torch.as_tensor(np.clip(np.sqrt((nn_ - p) / np.maximum(p, 1)), 1, 30), dtype=torch.float32, device=device)
    model = RealGraphNet(Z.shape[1], G["edge_attr"].shape[1], len(cols), cfg, mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    va = split == "val"
    hist, best, best_state, bad = [], -1.0, None, 0
    t0 = time.time()
    for ep in range(1, cfg.epochs + 1):
        model.train()
        opt.zero_grad()
        logit = model(x, ei_tr, ea_tr, lag_tr)
        l = Fn.binary_cross_entropy_with_logits(logit, Y, pos_weight=pw, reduction="none")
        per_task = (l * Mtr).sum(0) / Mtr.sum(0).clamp(min=1)
        loss = per_task.mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        model.eval()
        with torch.no_grad():
            P = torch.sigmoid(model(x, ei_all, ea_all, lag_all)).cpu().numpy()
        vap = _macro_ap(P, G["Y"], G["M"], va, cols)
        sched.step(vap)
        hist.append({"epoch": ep, "loss": loss.item(), "val_macro_pr_auc": vap,
                     **{f"loss_{t}": v for t, v in zip(tasks, per_task.detach().cpu().numpy())}})
        if verbose and (ep % 10 == 0 or ep == 1):
            print(f"  ep{ep:03d} loss {loss.item():.4f} val macro PR-AUC {vap:.4f}")
        if vap > best + 1e-4:
            best, best_state, bad = vap, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        P = torch.sigmoid(model(x, ei_all, ea_all, lag_all)).cpu().numpy()
    return {"model": model, "P": P, "tasks": tasks, "history": pd.DataFrame(hist), "best_val": best,
            "seconds": time.time() - t0, "epochs": len(hist), "n_params": sum(q.numel() for q in model.parameters()),
            "stats": stats, "mode": mode, "seed": seed}


# ==========================================================================
# Baselines
# ==========================================================================
def persistence_scores(G: Dict, tasks: Sequence[str]) -> np.ndarray:
    """'Next week looks like this week': the vessel's current-week amount of the same behaviour."""
    nd = G["nodes"]
    tie = 1e-3 * np.log1p(nd.n_events.values)
    return np.column_stack([np.log1p(nd[TARGET_SOURCE[t]].fillna(0).values) + tie for t in tasks])


def train_hgb(G: Dict, split: np.ndarray, tasks: Sequence[str], seed: int = 0) -> np.ndarray:
    """Gradient-boosted trees on the same node features (strong tabular, no-graph baseline)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = G["X_raw"].values
    tr = split == "train"
    out = []
    for t in tasks:
        k = list(TASKS).index(t)
        r = tr & G["M"][:, k]
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                             early_stopping=True, validation_fraction=0.15, random_state=seed)
        clf.fit(X[r], G["Y"][r, k])
        out.append(clf.predict_proba(X)[:, 1])
    return np.column_stack(out)


# ==========================================================================
# Calibration, priority, evaluation
# ==========================================================================
def calibrate(P: np.ndarray, G: Dict, split: np.ndarray, tasks: Sequence[str], is_prob: bool = True) -> np.ndarray:
    """Per-task Platt scaling fitted on the validation split (forecasts become probabilities)."""
    from sklearn.linear_model import LogisticRegression
    out = np.zeros_like(P, dtype=float)
    va = split == "val"
    for j, t in enumerate(tasks):
        k = list(TASKS).index(t)
        r = va & G["M"][:, k]
        z = np.log(np.clip(P[:, j], 1e-6, 1)) - np.log(np.clip(1 - P[:, j], 1e-6, 1)) if is_prob else P[:, j]
        y = G["Y"][r, k]
        if y.sum() == 0 or y.sum() == len(y):
            out[:, j] = y.mean() if len(y) else 0.0
            continue
        lr = LogisticRegression(C=1.0).fit(z[r, None], y)
        out[:, j] = lr.predict_proba(z[:, None])[:, 1]
    return out


def investigation_priority(Pcal: np.ndarray, tasks: Sequence[str], weights: Dict[str, float]) -> np.ndarray:
    """Weighted noisy-OR: probability-like chance of at least one flagged behaviour next week."""
    logq = np.zeros(len(Pcal))
    for j, t in enumerate(tasks):
        w = weights.get(t, 0.0)
        if w:
            logq += w * np.log(np.clip(1 - Pcal[:, j], 1e-9, 1))
    return 1 - np.exp(logq)


def task_metrics(P: np.ndarray, G: Dict, split: np.ndarray, tasks: Sequence[str], part: str = "test") -> pd.DataFrame:
    rows = []
    for j, t in enumerate(tasks):
        k = list(TASKS).index(t)
        r = (split == part) & G["M"][:, k]
        y = G["Y"][r, k]
        npos = int(y.sum())
        ok = 0 < npos < len(y)
        rows.append({"task": t, "n": int(r.sum()), "n_pos": npos, "base_rate": y.mean() if len(y) else np.nan,
                     "pr_auc": average_precision_score(y, P[r, j]) if ok else np.nan,
                     "roc_auc": roc_auc_score(y, P[r, j]) if ok else np.nan})
    return pd.DataFrame(rows)


def evidence_metrics(score: np.ndarray, G: Dict, split: np.ndarray, part: str = "test") -> pd.DataFrame:
    """How well a score ranks vessel-weeks whose NEXT week carries held-out evidence."""
    rows = []
    for k, name in enumerate(EVIDENCE):
        r = (split == part) & (G["EM"][:, k] if "EM" in G else G["M"][:, 0])
        y = G["E"][r, k]
        s = score[r]
        npos = int(y.sum())
        row = {"evidence": name, "n": int(r.sum()), "n_pos": npos, "base_rate": y.mean() if len(y) else np.nan}
        if 0 < npos < len(y):
            order = np.argsort(-s, kind="stable")
            for q in (0.01, 0.05):
                top = order[:max(1, int(round(q * len(s))))]
                row[f"lift_top{int(q * 100)}pct"] = y[top].mean() / y.mean()
                row[f"recall_top{int(q * 100)}pct"] = y[top].sum() / npos
            row["pr_auc"] = average_precision_score(y, s)
            row["roc_auc"] = roc_auc_score(y, s)
        rows.append(row)
    return pd.DataFrame(rows)


def run_real_experiment(G: Dict, splits: Dict[str, tuple], cfg: RealConfig, device,
                        modes=("full", "temporal_only", "no_graph"), include_hgb: bool = True,
                        verbose: bool = True) -> Dict:
    names = {"full": "Temporal GAT (full graph)", "temporal_only": "Temporal GAT (temporal edges only)",
             "no_graph": "Same model, no graph (MLP)"}
    task_rows, ev_rows, run_rows, preds, hists = [], [], [], {}, {}
    act = G["nodes"].n_events.values.astype(float)
    for sp_name, (split, gtrain) in splits.items():
        tasks = active_tasks(G, split, cfg)
        if verbose:
            print(f"[{sp_name}] tasks with enough positives: {tasks}")
        cands = []
        # reference scores
        Pp = persistence_scores(G, tasks)
        cands.append(("Persistence (this week = next week)", 0, Pp, None))
        if include_hgb:
            t0 = time.time()
            Ph = train_hgb(G, split, tasks)
            cands.append(("Gradient boosting (no graph)", 0, Ph, None))
            if verbose:
                print(f"  gradient boosting done in {time.time() - t0:.0f}s")
        for mode in modes:
            for seed in cfg.seeds:
                r = train_real(G, split, gtrain, cfg, seed, device, mode, tasks)
                if verbose:
                    print(f"  {names[mode]:38s} seed {seed}: val macro PR-AUC {r['best_val']:.4f} "
                          f"({r['epochs']} epochs, {r['seconds']:.0f}s)")
                hists[(sp_name, names[mode], seed)] = r["history"]
                cands.append((names[mode], seed, r["P"], r))
        for model_name, seed, P, raw in cands:
            Pc = calibrate(P, G, split, tasks, is_prob=not model_name.startswith("Persistence"))
            prio = investigation_priority(Pc, tasks, cfg.priority_weights)
            tm = task_metrics(P, G, split, tasks).assign(split=sp_name, model=model_name, seed=seed)
            em = evidence_metrics(prio, G, split).assign(split=sp_name, model=model_name, seed=seed)
            task_rows.append(tm)
            ev_rows.append(em)
            run_rows.append({"split": sp_name, "model": model_name, "seed": seed,
                             "macro_pr_auc": tm.pr_auc.mean(), "macro_roc_auc": tm.roc_auc.mean(),
                             "val_macro_pr_auc": _macro_ap(P, G["Y"], G["M"], split == "val",
                                                           [list(TASKS).index(t) for t in tasks])})
            preds[(sp_name, model_name, seed)] = {"P": P, "Pcal": Pc, "priority": prio, "tasks": tasks, "raw": raw}
        ev_rows.append(evidence_metrics(act, G, split).assign(split=sp_name, model="Activity only (n events this week)", seed=0))
    return {"tasks": pd.concat(task_rows, ignore_index=True), "evidence": pd.concat(ev_rows, ignore_index=True),
            "runs": pd.DataFrame(run_rows), "preds": preds, "histories": hists}


def summarize(res: Dict):
    t = (res["tasks"].groupby(["split", "task", "model"], sort=False)
         .agg(pr_auc=("pr_auc", "mean"), pr_auc_sd=("pr_auc", "std"), roc_auc=("roc_auc", "mean"),
              base_rate=("base_rate", "first"), n_pos=("n_pos", "first")).reset_index())
    num = [c for c in res["evidence"].columns if c not in ("evidence", "split", "model", "seed")]
    e = res["evidence"].groupby(["split", "evidence", "model"], sort=False)[num].mean().reset_index()
    r = res["runs"].groupby(["split", "model"], sort=False).agg(
        macro_pr_auc=("macro_pr_auc", "mean"), macro_pr_auc_sd=("macro_pr_auc", "std"),
        macro_roc_auc=("macro_roc_auc", "mean"), n_seeds=("seed", "nunique")).reset_index()
    return r, t, e


def best_seed(res: Dict, split: str, model: str):
    r = res["runs"]
    r = r[(r.split == split) & (r.model == model)].sort_values("val_macro_pr_auc", ascending=False)
    return res["preds"][(split, model, int(r.seed.iloc[0]))]


# ==========================================================================
# Alerts
# ==========================================================================
def alert_table(G: Dict, pred: Dict, split: np.ndarray, part: str = "test", top: int = 50,
                weights: Optional[Dict[str, float]] = None, checkable_only: bool = True,
                one_per_vessel: bool = True) -> pd.DataFrame:
    """Top-priority vessel-weeks. `checkable_only` keeps weeks whose next week was observed,
    so each alert can be compared with what actually happened."""
    nd = G["nodes"]
    tasks = pred["tasks"]
    r = np.flatnonzero((split == part) & (nd.has_next.values if checkable_only else True))
    order = r[np.argsort(-pred["priority"][r], kind="stable")]
    if one_per_vessel:    # each vessel's highest-priority week only
        order = order[~pd.Series(nd.vessel_id.values[order]).duplicated().values]
    order = order[:top]
    w = np.array([(weights or {}).get(t, 1.0) for t in tasks])
    Pc = pred["Pcal"][order]
    driver = np.array(tasks)[np.argmax(Pc * w, 1)]
    out = pd.DataFrame({
        "rank": np.arange(1, len(order) + 1), "vessel_id": nd.vessel_id.values[order],
        "vessel_class": nd.vessel_class.values[order], "week": nd.week.values[order],
        "lat": nd.lat.values[order].round(1), "lon": nd.lon.values[order].round(1),
        "priority": pred["priority"][order].round(3), "main_driver": [TASKS[d] for d in driver],
    })
    for j, t in enumerate(tasks):
        out[f"p_{t}"] = Pc[:, j].round(3)
    for c in ("n_gaps", "n_enc", "n_carrier_enc", "fishing_h", "fishing_mpa_h", "fishing_outside_flag_h"):
        out[f"this_week_{c}"] = nd[c].values[order].round(1)
    out["next_week_observed"] = [", ".join(t for t in TASKS if G["M"][i, 0] and nd[f"y_{t}"].values[i] > 0) or
                                 ("no flagged behaviour" if G["M"][i, 0] else "vessel not seen next week")
                                 for i in order]
    for k in EVIDENCE:
        out[f"heldout_{k}"] = nd[f"e_{k}"].values[order].astype(bool) & nd.has_next.values[order]
    out["caveat"] = "investigation priority only; not evidence of illegality"
    return out


def alerts_json(alerts: pd.DataFrame, path: Path, meta: Dict):
    import json
    recs = alerts.assign(week=alerts.week.astype(str)).to_dict("records")
    Path(path).write_text(json.dumps({"caveat": REAL_MODEL_CAVEAT, "meta": meta, "alerts": recs}, indent=2, default=str))
