import numpy as np
import pandas as pd

from src.config import PreprocessConfig
from src.data_io import anonymize_ids, normalize_schema, ship_type_to_class, CANONICAL_COLUMNS
from src.preprocessing import (add_kinematics, angle_diff, clean_ais, detect_gaps, haversine_m,
                               resample_tracks)
from tests.conftest import make_track


def test_haversine_one_degree_lat():
    assert abs(haversine_m(9.0, 79.0, 10.0, 79.0) - 111_195) < 50


def test_angle_wrap():
    assert angle_diff(350, 10) == 20
    assert angle_diff(10, 350) == -20


def test_ship_type_normalisation():
    assert [ship_type_to_class(c) for c in (30, 52, 60, 70, 80, None)] == \
        ["fishing", "tug", "passenger", "cargo", "tanker", "unknown"]


def test_schema_and_anonymisation(synth):
    can = normalize_schema(synth["ais_raw"], "synthetic_demo", registry=synth["registry"])
    anon, mapping = anonymize_ids(can)
    assert set(CANONICAL_COLUMNS) <= set(anon.columns)
    assert "mmsi" not in anon.columns
    assert anon.vessel_id.str.match(r"^vessel_[0-9a-f]{8}$").all()
    assert not anon.astype(str).apply(lambda c: c.str.contains("SYNTH")).any().any()
    # stable: same input -> same ids
    anon2, _ = anonymize_ids(can)
    assert (anon.vessel_id.values == anon2.vessel_id.values).all()
    # registry distinguishes reefer carriers from generic cargo
    assert "reefer_carrier" in set(anon.vessel_type)


def test_cleaning_removes_injected_errors(synth, processed):
    rep = processed["report"]
    inj = synth["injected_errors"].error_type.value_counts()
    assert rep["invalid_coordinates_removed"] == inj.get("invalid_coordinate", 0)
    assert rep["duplicates_removed"] <= inj.get("duplicate", 0)
    assert rep["sog_not_available"] == inj.get("sog_not_available", 0)
    # every injected jump is flagged or pushed outside the region
    assert rep["impossible_speed_flagged"] + rep["outside_regions_removed"] >= inj.get("position_jump", 0)
    ais = processed["ais"]
    assert not ais.duplicated(["vessel_id", "timestamp"]).any()
    assert ais.groupby("vessel_id").timestamp.apply(lambda s: s.is_monotonic_increasing).all()
    assert ais.latitude.between(-90, 90).all() and (ais.sog.dropna() < 102).all()


def test_impossible_speed_flag():
    df = make_track("v1", "2024-01-01", [0, 5, 10, 15, 20])
    df.loc[2, "latitude"] += 1.0  # ~111 km jump in 5 min
    out, rep = clean_ais(df, PreprocessConfig(drop_impossible=False))
    assert out.impossible_speed_flag.tolist() == [False, False, True, False, False]


def test_gap_detection_basic():
    df = make_track("v1", "2024-01-01", [0, 5, 10, 190, 195])  # 180-min silence
    df["region_id"] = "IN-GOM"
    kin = add_kinematics(df)
    msgs, gaps = detect_gaps(kin, PreprocessConfig(gap_threshold_min=60))
    assert len(gaps) == 1
    g = gaps.iloc[0]
    assert g.gap_duration_min == 180 and g.long_gap
    # gap id sits on the reappearance fix only
    assert msgs.ais_gap_id.notna().sum() == 1
    assert msgs.loc[msgs.ais_gap_id.notna(), "timestamp"].iloc[0] == df.timestamp.iloc[3]


def test_gap_threshold_is_strict():
    df = make_track("v1", "2024-01-01", [0, 60, 121])
    df["region_id"] = "IN-GOM"
    _, gaps = detect_gaps(add_kinematics(df), PreprocessConfig(gap_threshold_min=60))
    assert len(gaps) == 1 and gaps.gap_duration_min.iloc[0] == 61


def test_concurrent_silence_indicator():
    # target v1 silent 100-300; neighbour v2 keeps reporting -> fraction 0
    v1 = make_track("v1", "2024-01-01", [0, 30, 60, 90, 300])
    v2 = make_track("v2", "2024-01-01", list(range(0, 400, 10)), lon0=79.01)
    v3 = make_track("v3", "2024-01-01", [0, 30, 60, 90, 310], lon0=79.02)  # also silent
    df = pd.concat([v1, v2, v3]).assign(region_id="IN-GOM")
    _, gaps = detect_gaps(add_kinematics(df), PreprocessConfig())
    g1 = gaps[gaps.vessel_id == "v1"].iloc[0]
    assert g1.n_neighbours_before == 2
    assert g1.concurrent_silence_frac == 0.5


def test_resampling_never_bridges_long_gaps():
    df = make_track("v1", "2024-01-01", [0, 7, 14, 21, 200, 207, 214])
    df["region_id"] = "IN-GOM"
    rs = resample_tracks(df, PreprocessConfig(resample_min=5, max_interp_gap_min=30))
    assert (rs.bracket_span_min <= 30).all()
    t = rs.timestamp - pd.Timestamp("2024-01-01", tz="UTC")
    mins = t.dt.total_seconds() / 60
    assert not ((mins > 21) & (mins < 200)).any()
    assert len(rs) > 0


def test_processed_output_columns(processed):
    ais = processed["ais"]
    need = {"dt_minutes", "distance_m", "acceleration", "turn_rate", "reporting_interval",
            "missingness_flag", "fishing_motion_score", "mpa_distance_m", "eez_distance_m",
            "port_distance_m", "ais_gap_id", "gap_duration_min", "sar_detection_flag", "segment_id"}
    assert need <= set(ais.columns)
    assert ais.sar_detection_flag.isna().all()  # unknown until SAR layer exists
    assert ais.fishing_motion_score.between(0, 1).all()


def test_jump_right_after_silence_is_flagged():
    # fix at minute 300 follows a 5-h silence and is ~55 km off; the next fix is back on track
    df = make_track("v1", "2024-01-01", [0, 5, 10, 300, 305, 310])
    df.loc[3, "latitude"] += 0.5
    out, _ = clean_ais(df, PreprocessConfig(drop_impossible=False))
    assert out.impossible_speed_flag.tolist() == [False, False, False, True, False, False]
