"""Data input/output: canonical AIS schema, anonymisation, adapters.

Three data modes (brief section 5):
  A  synthetic   -> `src.synthetic.generate_synthetic_dataset` (works offline)
  B  open AIS    -> `load_local_ais_csv` for any properly licensed file you have
                    downloaded, and `GFWAdapter` (interface only; needs your own
                    Global Fishing Watch API token and acceptance of GFW terms)
  C  SAR         -> `SARDetectionsAdapter` (interface only; tabular detections)

Synthetic and real data never share files: synthetic lives in data/synthetic,
real downloads in data/raw, and every table carries a `source` column.
"""
from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

CANONICAL_COLUMNS = [
    "vessel_id", "timestamp", "latitude", "longitude", "sog", "cog", "heading",
    "nav_status", "vessel_type", "length_m", "width_m", "draft_m", "source", "region_id",
]
DERIVED_COLUMNS = [
    "dt_minutes", "distance_m", "acceleration", "turn_rate", "reporting_interval",
    "missingness_flag", "fishing_motion_score", "mpa_distance_m", "eez_distance_m",
    "port_distance_m", "ais_gap_id", "gap_duration_min", "sar_detection_flag",
]

# Column maps from provider formats to the canonical names (raw id kept as `mmsi`
# only until anonymisation).
COLUMN_MAPS: Dict[str, Dict[str, str]] = {
    "synthetic_demo": {
        "mmsi": "mmsi", "timestamp": "timestamp", "lat": "latitude", "lon": "longitude",
        "sog": "sog", "cog": "cog", "heading": "heading", "nav_status": "nav_status",
        "ship_type": "ship_type", "length": "length_m", "width": "width_m", "draught": "draft_m",
    },
    # common open-AIS CSV layout (e.g. MarineCadastre-style column names)
    "generic_csv": {
        "MMSI": "mmsi", "BaseDateTime": "timestamp", "LAT": "latitude", "LON": "longitude",
        "SOG": "sog", "COG": "cog", "Heading": "heading", "Status": "nav_status",
        "VesselType": "ship_type", "Length": "length_m", "Width": "width_m", "Draft": "draft_m",
    },
}


# --------------------------------------------------------------------------
# Vessel type normalisation (ITU-R M.1371 ship-type codes -> coarse classes)
# --------------------------------------------------------------------------
def ship_type_to_class(code) -> str:
    try:
        c = int(code)
    except (TypeError, ValueError):
        return "unknown"
    if c == 30:
        return "fishing"
    if c in (31, 32, 52):
        return "tug"
    if 50 <= c <= 59:
        return "service"
    if 60 <= c <= 69:
        return "passenger"
    if 70 <= c <= 79:
        return "cargo"
    if 80 <= c <= 89:
        return "tanker"
    if c in (36, 37):
        return "pleasure"
    if c == 35:
        return "military"
    return "other"


VESSEL_CLASSES = ["fishing", "reefer_carrier", "cargo", "tanker", "tug", "passenger",
                  "service", "pleasure", "military", "other", "unknown"]


def normalize_schema(raw: pd.DataFrame, source: str, column_map: Optional[Dict[str, str]] = None,
                     registry: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Rename provider columns to the canonical schema and normalise vessel type.

    `registry` (optional) is a vessel-identity table with columns `mmsi` and
    `vessel_class`; when present it overrides the coarse AIS ship-type class
    (e.g. to distinguish reefer/fish carriers from general cargo).
    The raw identifier is kept in column `mmsi` until `anonymize_ids` is called.
    """
    cmap = column_map or COLUMN_MAPS.get(source) or COLUMN_MAPS["generic_csv"]
    missing = [k for k in cmap if k not in raw.columns]
    if missing:
        raise KeyError(f"columns {missing} not found for source '{source}'")
    df = raw.rename(columns=cmap)[list(cmap.values())].copy()
    df["mmsi"] = df["mmsi"].astype(str)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    for c in ("latitude", "longitude", "sog", "cog", "heading", "length_m", "width_m", "draft_m"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["nav_status"] = pd.to_numeric(df["nav_status"], errors="coerce").fillna(15).astype(int)
    df["vessel_type"] = df["ship_type"].map(ship_type_to_class)
    if registry is not None and "vessel_class" in registry:
        reg = registry.assign(mmsi=registry.mmsi.astype(str)).set_index("mmsi")["vessel_class"]
        df["vessel_type"] = df["mmsi"].map(reg).fillna(df["vessel_type"])
    df = df.drop(columns=["ship_type"])
    df["source"] = source
    df["region_id"] = None  # assigned during preprocessing
    return df


# --------------------------------------------------------------------------
# Anonymisation
# --------------------------------------------------------------------------
DEFAULT_DEMO_SALT = "india-fishing-gnn-synthetic-demo"  # fine for synthetic data only


def anonymize_ids(df: pd.DataFrame, id_col: str = "mmsi", salt: Optional[str] = None,
                  prefix: str = "vessel_") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Replace raw identifiers with stable keyed-hash IDs (HMAC-SHA256).

    The same raw id + salt always gives the same `vessel_xxxxxxxx`, so IDs are
    stable across runs and datasets. For real data set a private salt via the
    environment variable AIS_ID_SALT and keep the returned mapping private
    (it is written under data/processed/_private/, which is git-ignored).
    """
    salt = salt or os.environ.get("AIS_ID_SALT") or DEFAULT_DEMO_SALT
    uniq = pd.Series(df[id_col].astype(str).unique())
    hashed = uniq.map(lambda x: prefix + hmac.new(salt.encode(), x.encode(), hashlib.sha256).hexdigest()[:8])
    if hashed.duplicated().any():  # astronomically unlikely; fail loudly
        raise RuntimeError("anonymised id collision; use a longer hash")
    mapping = pd.DataFrame({"raw_id": uniq.values, "vessel_id": hashed.values})
    out = df.copy()
    out["vessel_id"] = out[id_col].astype(str).map(dict(zip(mapping.raw_id, mapping.vessel_id)))
    out = out.drop(columns=[id_col])
    cols = ["vessel_id"] + [c for c in out.columns if c != "vessel_id"]
    return out[cols], mapping


# --------------------------------------------------------------------------
# Storage helpers (Parquet with CSV fallback)
# --------------------------------------------------------------------------
def save_table(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path.with_suffix(".parquet"), index=False)
        return path.with_suffix(".parquet")
    except (ImportError, ValueError):  # pyarrow missing
        df.to_csv(path.with_suffix(".csv"), index=False)
        return path.with_suffix(".csv")


def load_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.with_suffix(".parquet").exists():
        return pd.read_parquet(path.with_suffix(".parquet"))
    return pd.read_csv(path.with_suffix(".csv"))


# --------------------------------------------------------------------------
# Mode B / C adapters
# --------------------------------------------------------------------------
def load_local_ais_csv(path: Path, source: str, column_map: Optional[Dict[str, str]] = None,
                       registry: Optional[pd.DataFrame] = None, chunksize: int = 250_000) -> pd.DataFrame:
    """Load a locally downloaded, properly licensed AIS CSV in chunks and normalise it.

    Record the file's provenance in DATA_SOURCE.md before using it.
    """
    parts = [normalize_schema(ch, source, column_map, registry)
             for ch in pd.read_csv(path, chunksize=chunksize)]
    return pd.concat(parts, ignore_index=True)


class GFWAdapter:
    """Interface for Global Fishing Watch data (Mode B / Mode C).

    Not implemented in Milestone 1 - no GFW data are bundled or fabricated.
    To implement (Milestone 5): register at globalfishingwatch.org, accept the
    terms of use, create an API token, export it as GFW_API_TOKEN, and use the
    official `gfw-api-python-client` to request vessel presence / apparent
    fishing effort / encounters / SAR vessel detections for the IN-GOM bbox.
    Every download must be logged in DATA_SOURCE.md (access date, licence,
    extent, columns, restrictions) and attributed to Global Fishing Watch.
    """

    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("GFW_API_TOKEN")

    def available(self) -> bool:
        return bool(self.token)

    def fetch_ais_positions(self, bbox, start, end) -> pd.DataFrame:
        raise NotImplementedError(
            "GFW adapter is an interface only in Milestone 1. Provide GFW_API_TOKEN and "
            "implement with the official client in Milestone 5; synthetic mode is used meanwhile.")

    def fetch_sar_detections(self, bbox, start, end) -> pd.DataFrame:
        raise NotImplementedError("SAR detections adapter arrives in Milestone 5.")


SAR_SCHEMA = ["detection_id", "timestamp", "latitude", "longitude", "length_m_est",
              "fishing_score", "matched_vessel_id", "match_score", "footprint_id", "source"]


class SARDetectionsAdapter:
    """Interface for tabular Sentinel-1 vessel detections (verification evidence only)."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path

    def load(self) -> pd.DataFrame:
        if self.path is None or not Path(self.path).exists():
            raise FileNotFoundError(
                "No SAR detections file available. SAR is optional verification evidence; "
                "the pipeline sets sar_detection_flag = <NA> (unknown) when absent.")
        df = pd.read_csv(self.path) if str(self.path).endswith(".csv") else pd.read_parquet(self.path)
        missing = [c for c in SAR_SCHEMA if c not in df.columns]
        if missing:
            raise KeyError(f"SAR file lacks columns {missing}")
        return df[SAR_SCHEMA]
