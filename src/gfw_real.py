"""Mode B/C - REAL data from Global Fishing Watch (GFW) for Indian waters.

What GFW's public API provides (and what it does not):
  * Events derived from real AIS: apparent fishing, encounters (vessel meetings,
    e.g. fishing-carrier), loitering, port visits, and AIS-off gaps (with off/on
    positions and satellite-reception quality).
  * Sentinel-1 SAR vessel detections (4Wings "sar-presence"), optionally only
    detections NOT matched to any AIS vessel ("dark" detections).
  * Gridded apparent fishing effort per vessel.
  * It does NOT provide raw ping-by-ping AIS tracks.

Terms: GFW APIs are for non-commercial use; attribute "Global Fishing Watch"
in all outputs. Never commit the API token. Public outputs use anonymised
vessel identifiers.

The API calls are asynchronous (the official `gfw-api-python-client`); in
Colab/Jupyter call them with top-level `await`.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import RAW_DIR

GFW_DIR = RAW_DIR / "gfw"
REAL_CAVEAT = (
    "REAL DATA from Global Fishing Watch (AIS-derived events and Sentinel-1 SAR detections). "
    "Events describe apparent activity patterns; they are not evidence of illegal activity. "
    "Many small Indian fishing boats do not carry AIS, so coverage is incomplete. "
    "Attribution: Global Fishing Watch, globalfishingwatch.org (non-commercial use)."
)

EVENT_DATASETS = {
    "fishing": "public-global-fishing-events:latest",
    "encounter": "public-global-encounters-events:latest",
    "loitering": "public-global-loitering-events:latest",
    "port_visit": "public-global-port-visits-events:latest",
    "gap": "public-global-gaps-events:latest",
}


def bbox_polygon(bbox) -> Dict:
    """(lat_min, lat_max, lon_min, lon_max) -> GeoJSON Polygon (lon, lat order)."""
    a0, a1, o0, o1 = bbox
    return {"type": "Polygon", "coordinates": [[[o0, a0], [o1, a0], [o1, a1], [o0, a1], [o0, a0]]]}


def month_chunks(start: str, end: str) -> List[Tuple[str, str]]:
    """Split [start, end] into calendar-month requests (keeps each API call small)."""
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    out, cur = [], s
    while cur <= e:
        nxt = min((cur + pd.offsets.MonthBegin(1)).normalize(), e + pd.Timedelta(days=1))
        out.append((cur.strftime("%Y-%m-%d"), nxt.strftime("%Y-%m-%d")))
        cur = nxt
    return out


def get_token() -> str:
    """Read the token from Colab Secrets (key icon, name GFW_API_TOKEN), the environment, or a prompt."""
    tok = os.environ.get("GFW_API_TOKEN")
    if not tok:
        try:
            from google.colab import userdata  # type: ignore
            tok = userdata.get("GFW_API_TOKEN")
        except Exception:
            tok = None
    if not tok:
        import getpass
        tok = getpass.getpass("Paste your Global Fishing Watch API token (input hidden): ").strip()
    if not tok:
        raise RuntimeError("No GFW API token provided.")
    return tok


def make_client(token: str):
    import gfwapiclient as gfw
    return gfw.Client(access_token=token)


async def _retry(fn, tries: int = 5, **kw):
    for k in range(tries):
        try:
            return await fn(**kw)
        except Exception as e:  # rate limits / transient server errors -> back off
            msg = f"{type(e).__name__}: {e}"
            transient = any(s in msg for s in ("429", "Too Many", "imeout", "502", "503", "504", "500", "Connect"))
            if transient and k < tries - 1:
                await asyncio.sleep(3 * 2 ** k)
                continue
            raise


def _items(result) -> List[dict]:
    d = result.data()
    d = d if isinstance(d, list) else [d]
    return [x.model_dump() if hasattr(x, "model_dump") else dict(x) for x in d]


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------
async def fetch_events(client, kind: str, geometry: Dict, start: str, end: str, cache_dir: Path = GFW_DIR,
                       page: int = 1000, max_per_chunk: int = 50_000, concurrency: int = 4,
                       verbose: bool = True) -> pd.DataFrame:
    """All events of one kind inside `geometry`, month by month (several months in
    parallel), cached as Parquet so an interrupted run resumes where it stopped."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    state = {"page": page}

    async def one_month(s, e):
        f = cache_dir / f"events_{kind}_{s}.parquet"
        if f.exists():
            return pd.read_parquet(f)
        async with sem:
            t0 = time.time()
            rows, offset = [], 0
            while True:
                lim = state["page"]
                try:
                    res = await _retry(client.events.get_all_events, datasets=[EVENT_DATASETS[kind]],
                                       geometry=geometry, start_date=s, end_date=e, limit=lim, offset=offset)
                except Exception as ex:          # server rejects big pages -> fall back once
                    if lim > 200 and ("422" in str(ex) or "limit" in str(ex).lower()):
                        state["page"] = 200
                        continue
                    raise
                items = _items(res)
                rows += items
                if len(items) < lim or len(rows) >= max_per_chunk:
                    break
                offset += lim
            df = normalize_events(rows, kind)
            df.to_parquet(f, index=False)
            if verbose:
                print(f"  {kind:10s} {s}: {len(df):6d} events  ({time.time() - t0:.0f}s)", flush=True)
            return df

    frames = await asyncio.gather(*[one_month(s, e) for s, e in month_chunks(start, end)])
    frames = [f for f in frames if len(f)]
    out = pd.concat(frames, ignore_index=True) if frames else normalize_events([], kind)
    return out.drop_duplicates("event_id").reset_index(drop=True)


def _col(flat: pd.DataFrame, name: str, default=np.nan):
    return flat[name] if name in flat.columns else pd.Series(default, index=flat.index)


def _join(v):
    if isinstance(v, (list, tuple, np.ndarray)):
        return ",".join(str(x) for x in v)
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)


EVENT_COLUMNS = ["event_id", "event_type", "start", "end", "duration_h", "lat", "lon", "vessel_gfw_id", "ssvid",
                 "vessel_name", "flag", "vessel_type", "mpa_ids", "eez_ids", "in_mpa", "dist_shore_km", "dist_port_km"]


def normalize_events(rows: List[dict], kind: str) -> pd.DataFrame:
    """Flatten GFW event JSON (model_dump output) into one tidy row per event."""
    if not rows:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    flat = pd.json_normalize(rows, sep=".")
    out = pd.DataFrame({
        "event_id": _col(flat, "id").astype(str),
        "event_type": kind,
        "start": pd.to_datetime(_col(flat, "start"), utc=True, errors="coerce"),
        "end": pd.to_datetime(_col(flat, "end"), utc=True, errors="coerce"),
        "lat": pd.to_numeric(_col(flat, "position.lat"), errors="coerce"),
        "lon": pd.to_numeric(_col(flat, "position.lon"), errors="coerce"),
        "vessel_gfw_id": _col(flat, "vessel.id", "").astype(str),
        "ssvid": _col(flat, "vessel.ssvid", "").astype(str),
        "vessel_name": _col(flat, "vessel.name", "").astype(str),
        "flag": _col(flat, "vessel.flag", "").astype(str),
        "vessel_type": _col(flat, "vessel.type", "").astype(str).str.lower(),
        "mpa_ids": _col(flat, "regions.mpa", "").map(_join),
        "eez_ids": _col(flat, "regions.eez", "").map(_join),
        "dist_shore_km": pd.to_numeric(_col(flat, "distances.start_distance_from_shore_km"), errors="coerce"),
        "dist_port_km": pd.to_numeric(_col(flat, "distances.start_distance_from_port_km"), errors="coerce"),
    })
    out["duration_h"] = (out.end - out.start).dt.total_seconds() / 3600
    out["in_mpa"] = out.mpa_ids.str.len() > 0
    num = lambda c: pd.to_numeric(_col(flat, c), errors="coerce")  # noqa: E731
    if kind == "fishing":
        out["fishing_distance_km"] = num("fishing.total_distance_km")
        out["fishing_speed_kn"] = num("fishing.average_speed_knots")
        out["potential_risk"] = _col(flat, "fishing.potential_risk", None)
    elif kind == "encounter":
        out["partner_gfw_id"] = _col(flat, "encounter.vessel.id", "").astype(str)
        out["partner_ssvid"] = _col(flat, "encounter.vessel.ssvid", "").astype(str)
        out["partner_type"] = _col(flat, "encounter.vessel.type", "").astype(str).str.lower()
        out["partner_flag"] = _col(flat, "encounter.vessel.flag", "").astype(str)
        out["median_distance_km"] = num("encounter.median_distance_kilometers")
        out["median_speed_kn"] = num("encounter.median_speed_knots")
        out["encounter_type"] = _col(flat, "encounter.type", "").astype(str)
        out["potential_risk"] = _col(flat, "encounter.potential_risk", None)
    elif kind == "gap":
        out["intentional_disabling"] = _col(flat, "gap.intentional_disabling", None)
        out["gap_distance_km"] = num("gap.distance_km")
        out["gap_duration_h"] = num("gap.duration_hours")
        out["implied_speed_kn"] = num("gap.implied_speed_knots")
        out["sat_positions_12h_before"] = num("gap.positions_12_hours_before_sat")
        out["sat_positions_per_day"] = num("gap.positions_per_day_sat_reception")
        out["off_lat"], out["off_lon"] = num("gap.off_position.lat"), num("gap.off_position.lon")
        out["on_lat"], out["on_lon"] = num("gap.on_position.lat"), num("gap.on_position.lon")
    elif kind == "loitering":
        out["loiter_hours"] = num("loitering.total_time_hours")
        out["loiter_distance_km"] = num("loitering.total_distance_km")
        out["loiter_speed_kn"] = num("loitering.average_speed_knots")
    elif kind == "port_visit":
        out["port_duration_h"] = num("port_visit.duration_hrs")
        out["port_confidence"] = _col(flat, "port_visit.confidence", "").astype(str)
        out["port_name"] = _col(flat, "port_visit.start_anchorage.name", "").astype(str)
    for c in ("potential_risk", "intentional_disabling"):
        if c in out:
            out[c] = out[c].astype("object").where(out[c].notna(), None).map(
                lambda v: None if v is None else bool(v)).astype("boolean")
    return out


# --------------------------------------------------------------------------
# SAR detections and fishing effort (4Wings reports)
# --------------------------------------------------------------------------
async def fetch_sar(client, geometry: Dict, start: str, end: str, cache_dir: Path = GFW_DIR,
                    unmatched_only: bool = True, verbose: bool = True) -> pd.DataFrame:
    """Daily Sentinel-1 vessel detections per grid cell; `unmatched_only` keeps detections
    with no matching AIS vessel (candidate dark vessels)."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = "unmatched" if unmatched_only else "all"
    frames = []
    for s, e in month_chunks(start, end):
        f = cache_dir / f"sar_{tag}_{s}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
            continue
        rows, err = None, None
        for filt in ((["matched='false'"], ["matched=false"], None) if unmatched_only else (None,)):
            try:
                res = await _retry(client.fourwings.create_sar_presence_report, spatial_resolution="HIGH",
                                   temporal_resolution="DAILY", filters=filt, start_date=s, end_date=e,
                                   geojson=geometry)
                rows = _items(res)
                break
            except Exception as ex:  # try the next filter spelling
                err = ex
        if rows is None:
            raise RuntimeError(f"SAR request failed for {s}: {err}")
        df = pd.DataFrame(rows)
        keep = [c for c in ("date", "lat", "lon", "detections", "vessel_type", "flag") if c in df]
        df = df[keep] if len(df) else pd.DataFrame(columns=["date", "lat", "lon", "detections"])
        df["matched_filter"] = tag
        df.to_parquet(f, index=False)
        frames.append(df)
        if verbose:
            print(f"  SAR ({tag}) {s}: {len(df):5d} cell-days")
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce")
    return out


async def fetch_fishing_effort(client, geometry: Dict, start: str, end: str, cache_dir: Path = GFW_DIR,
                               verbose: bool = True) -> pd.DataFrame:
    """Monthly apparent-fishing hours per vessel and grid cell."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for s, e in month_chunks(start, end):
        f = cache_dir / f"effort_{s}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f))
            continue
        res = await _retry(client.fourwings.create_fishing_effort_report, spatial_resolution="LOW",
                           temporal_resolution="MONTHLY", group_by="VESSEL_ID", start_date=s, end_date=e,
                           geojson=geometry)
        df = pd.DataFrame(_items(res))
        df.to_parquet(f, index=False)
        frames.append(df)
        if verbose:
            print(f"  effort {s}: {len(df):5d} rows")
    return pd.concat(frames, ignore_index=True)


async def fetch_region_names(client) -> Dict[str, Dict[str, str]]:
    """Id -> name lookups for EEZs and MPAs (used to interpret event regions)."""
    out = {}
    for key, fn in (("eez", client.references.get_eez_regions), ("mpa", client.references.get_mpa_regions)):
        try:
            df = (await _retry(fn)).df()
            idc = "id" if "id" in df else df.columns[0]
            lab = "label" if "label" in df else ("name" if "name" in df else df.columns[-1])
            out[key] = dict(zip(df[idc].astype(str), df[lab].astype(str)))
        except Exception as ex:
            print(f"region lookup {key} unavailable: {ex}")
            out[key] = {}
    return out


def write_provenance(path: Path, region_name: str, bbox, start: str, end: str, counts: Dict[str, int]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rec = {"source": "Global Fishing Watch API v3 (gfw-api-python-client)",
           "access_date_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "licence": "GFW API terms - non-commercial use, attribution required",
           "region": region_name, "bbox_lat_lat_lon_lon": list(bbox), "start": start, "end": end,
           "datasets": EVENT_DATASETS | {"sar": "public-global-sar-presence:latest",
                                         "effort": "public-global-fishing-effort:latest"},
           "row_counts": counts, "caveat": REAL_CAVEAT}
    path.write_text(json.dumps(rec, indent=2))
    return rec


# --------------------------------------------------------------------------
# Anonymisation for any public output
# --------------------------------------------------------------------------
def anonymize_real(events: pd.DataFrame, salt: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    salt = salt or os.environ.get("AIS_ID_SALT")
    if not salt:
        try:  # optional private salt kept in Colab Secrets
            from google.colab import userdata  # type: ignore
            salt = userdata.get("AIS_ID_SALT")
        except Exception:
            salt = None
    # Without a private salt the IDs are only pseudonymous: anyone who knows this default can re-derive them.
    salt = salt or "india-fishing-gnn-real-data"
    h = lambda x: "" if not x or x == "nan" else "vessel_" + hmac.new(salt.encode(), x.encode(), hashlib.sha256).hexdigest()[:8]  # noqa: E731
    ids = pd.unique(pd.concat([events.vessel_gfw_id, events.get("partner_gfw_id", pd.Series(dtype=str))]).dropna())
    mapping = pd.DataFrame({"vessel_gfw_id": ids, "vessel_id": [h(x) for x in ids]})
    m = dict(zip(mapping.vessel_gfw_id, mapping.vessel_id))
    out = events.copy()
    out["vessel_id"] = out.vessel_gfw_id.map(m)
    if "partner_gfw_id" in out:
        out["partner_id"] = out.partner_gfw_id.map(m)
    drop = [c for c in ("vessel_gfw_id", "ssvid", "vessel_name", "partner_gfw_id", "partner_ssvid") if c in out]
    return out.drop(columns=drop), mapping


# --------------------------------------------------------------------------
# Real-data evidence features and vessel-week graph tables
# --------------------------------------------------------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def sar_near_gaps(gaps: pd.DataFrame, sar: pd.DataFrame, radius_km: float = 20.0) -> pd.Series:
    """Number of unmatched SAR detections within `radius_km` of a gap's off/on positions
    on any day the gap was open - independent satellite evidence of activity while AIS was silent."""
    if gaps.empty or sar.empty:
        return pd.Series(0, index=gaps.index, dtype=int)
    s_day = sar.date.dt.floor("D").values
    out = []
    for g in gaps.itertuples():
        d0, d1 = pd.Timestamp(g.start).floor("D"), pd.Timestamp(g.end).floor("D")
        m = (s_day >= np.datetime64(d0.tz_convert(None))) & (s_day <= np.datetime64(d1.tz_convert(None)))
        if not m.any():
            out.append(0)
            continue
        c = sar[m]
        n = 0
        for la, lo in ((g.off_lat, g.off_lon), (g.on_lat, g.on_lon)):
            if pd.notna(la):
                n = max(n, int(c.detections[haversine_km(la, lo, c.lat.values, c.lon.values) <= radius_km].sum()))
        out.append(n)
    return pd.Series(out, index=gaps.index)


def vessel_week_tables(ev: pd.DataFrame, india_eez_ids: Iterable[str] = ()) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Nodes = (vessel, ISO week) with aggregated real event features; edges = encounters
    (both directions) and temporal links to the same vessel's previous active week."""
    ev = ev.copy()
    ev["week"] = ev.start.dt.tz_convert(None).dt.to_period("W-SUN").dt.start_time
    india = set(map(str, india_eez_ids))
    ev["intentional_num"] = (pd.to_numeric(ev["intentional_disabling"].astype("Float64"), errors="coerce").fillna(0)
                             if "intentional_disabling" in ev else 0.0)
    ev["outside_india_eez"] = ev.eez_ids.map(lambda s: bool(s) and bool(india) and not (set(s.split(",")) & india))
    g = ev.groupby(["vessel_id", "week"])

    def s(kind, col, fn="sum", where=None):
        x = ev[ev.event_type == kind]
        if where is not None:
            x = x[where(x)]
        if col not in x:
            return pd.Series(dtype=float)
        return x.groupby(["vessel_id", "week"])[col].agg(fn)

    nodes = pd.DataFrame({
        "n_events": g.size(),
        "vessel_type": g.vessel_type.agg(lambda v: v.mode().iloc[0] if len(v.mode()) else ""),
        "flag": g.flag.agg(lambda v: v.mode().iloc[0] if len(v.mode()) else ""),
        "fishing_hours": s("fishing", "duration_h"),
        "fishing_in_mpa_hours": s("fishing", "duration_h", where=lambda x: x.in_mpa),
        "fishing_outside_india_eez_hours": s("fishing", "duration_h", where=lambda x: x.outside_india_eez),
        "n_gaps": s("gap", "event_id", "count"),
        "gap_hours": s("gap", "gap_duration_h"),
        "n_intentional_gaps": s("gap", "intentional_num"),
        "max_gap_km": s("gap", "gap_distance_km", "max"),
        "max_gap_implied_speed_kn": s("gap", "implied_speed_kn", "max"),
        "min_sat_positions_per_day": s("gap", "sat_positions_per_day", "min"),
        "sar_dark_near_gaps": s("gap", "sar_dark_nearby", "sum"),
        "n_loitering": s("loitering", "event_id", "count"),
        "loiter_hours": s("loitering", "loiter_hours"),
        "n_encounters": s("encounter", "event_id", "count"),
        "n_carrier_encounters": s("encounter", "event_id", "count", where=lambda x: x.partner_type.eq("carrier")),
        "n_port_visits": s("port_visit", "event_id", "count"),
        "mean_dist_shore_km": g.dist_shore_km.mean(),
        "events_in_mpa": g.in_mpa.sum(),
    }).fillna({"fishing_hours": 0, "fishing_in_mpa_hours": 0, "fishing_outside_india_eez_hours": 0, "n_gaps": 0,
               "gap_hours": 0, "n_intentional_gaps": 0, "sar_dark_near_gaps": 0, "n_loitering": 0,
               "loiter_hours": 0, "n_encounters": 0, "n_carrier_encounters": 0, "n_port_visits": 0})
    nodes = nodes[nodes.index.map(lambda k: isinstance(k, tuple))]   # drop artefacts of empty series
    nodes.index = pd.MultiIndex.from_tuples(nodes.index, names=["vessel_id", "week"])
    nodes = nodes.reset_index()
    enc = ev[(ev.event_type == "encounter") & ev.partner_id.notna() & (ev.partner_id != "")] if "partner_id" in ev else ev.iloc[:0]
    e1 = enc[["vessel_id", "partner_id", "week", "median_distance_km", "median_speed_kn", "duration_h", "encounter_type"]]
    e2 = e1.rename(columns={"vessel_id": "partner_id", "partner_id": "vessel_id"})
    edges = pd.concat([e1, e2], ignore_index=True).rename(columns={"vessel_id": "src", "partner_id": "dst"})
    edges["edge_type"] = "encounter"
    # partners without their own events still become nodes
    extra = edges[["dst", "week"]].rename(columns={"dst": "vessel_id"}).drop_duplicates()
    nodes = nodes.merge(extra, on=["vessel_id", "week"], how="outer").fillna({"n_events": 0})
    nodes = nodes.sort_values(["vessel_id", "week"]).reset_index(drop=True)
    for c in nodes.columns:   # uniform numeric dtypes (bool/int/NaN mixes break Parquet)
        if c not in ("vessel_id", "week", "vessel_type", "flag"):
            nodes[c] = pd.to_numeric(nodes[c], errors="coerce").astype(float)
    prev = nodes.groupby("vessel_id").week.shift()
    t = nodes[prev.notna()].assign(src=nodes.vessel_id, prev_week=prev)
    temporal = pd.DataFrame({"src": t.vessel_id, "dst": t.vessel_id, "week": t.week, "prev_week": t.prev_week,
                             "edge_type": "temporal"})
    return nodes, pd.concat([edges, temporal], ignore_index=True)
