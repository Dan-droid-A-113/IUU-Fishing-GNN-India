import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import PreprocessConfig, small_synthetic_config  # noqa: E402
from src.data_io import anonymize_ids, normalize_schema  # noqa: E402
from src.india_layers import get_layers  # noqa: E402
from src.preprocessing import run_preprocessing  # noqa: E402
from src.synthetic import generate_synthetic_dataset  # noqa: E402


@pytest.fixture(scope="session")
def layers():
    return get_layers("IN-GOM")


@pytest.fixture(scope="session")
def synth():
    return generate_synthetic_dataset(small_synthetic_config(), save=False)


@pytest.fixture(scope="session")
def processed(synth, layers):
    can = normalize_schema(synth["ais_raw"], "synthetic_demo", registry=synth["registry"])
    anon, idmap = anonymize_ids(can)
    res = run_preprocessing(anon, layers, PreprocessConfig())
    res["id_map"] = idmap
    return res


def make_track(vid, start, minutes, lat0=9.0, lon0=79.0, dlat=0.0005, sog=5.0, cog=0.0):
    """Tiny helper: regular track with given minute offsets."""
    ts = pd.Timestamp(start, tz="UTC") + pd.to_timedelta(minutes, unit="min")
    n = len(minutes)
    return pd.DataFrame({
        "vessel_id": vid, "timestamp": ts,
        "latitude": [lat0 + dlat * i for i in range(n)], "longitude": [lon0] * n,
        "sog": [sog] * n, "cog": [cog] * n, "heading": [cog] * n, "nav_status": 0,
        "vessel_type": "fishing", "length_m": 15.0, "width_m": 4.0, "draft_m": 2.0,
        "source": "unit_test", "region_id": None,
    })
