import pandas as pd

from src.graph_builder import proximity_pairs
from src.splits import (check_no_leakage, connected_groups, region_disjoint_split,
                        segment_table, time_disjoint_split, vessel_disjoint_split)
from tests.conftest import make_track


def test_proximity_pairs_sparse_and_correct():
    a = make_track("a", "2024-01-01", list(range(0, 60, 5)))
    b = make_track("b", "2024-01-01", list(range(0, 60, 5)), lon0=79.003)   # ~330 m east
    c = make_track("c", "2024-01-01", list(range(0, 60, 5)), lon0=79.5)     # far away
    p = proximity_pairs(pd.concat([a, b, c]), radius_m=500, min_bins=2, exclude_port_m=0)
    assert list(zip(p.vessel_a, p.vessel_b)) == [("a", "b")]
    assert 250 < p.min_dist_m.iloc[0] < 400


def test_connected_groups_transitive():
    g = connected_groups(["a", "b", "c", "d"], [("a", "b"), ("b", "c")])
    assert g["a"] == g["b"] == g["c"] != g["d"]


def test_vessel_split_keeps_groups_together():
    vessels = [f"v{i}" for i in range(30)]
    pairs = [("v0", "v1"), ("v1", "v2"), ("v10", "v11")]
    strata = {v: int(i % 5 == 0) for i, v in enumerate(vessels)}
    s = vessel_disjoint_split(vessels, pairs, strata, seed=1)
    assert s["v0"] == s["v1"] == s["v2"] and s["v10"] == s["v11"]
    assert set(s.values()) == {"train", "val", "test"}
    pos_splits = {s[v] for v in vessels if strata[v]}
    assert len(pos_splits) >= 2


def test_time_split_embargo():
    t = pd.Timestamp("2024-01-01", tz="UTC")
    seg = pd.DataFrame({"segment_id": ["s1", "s2", "s3", "s4"],
                        "start": [t, t + pd.Timedelta("47h"), t + pd.Timedelta("60h"), t + pd.Timedelta("103h")],
                        "end": [t + pd.Timedelta("10h"), t + pd.Timedelta("50h"), t + pd.Timedelta("70h"), t + pd.Timedelta("110h")]})
    out = time_disjoint_split(seg, [t + pd.Timedelta("48h"), t + pd.Timedelta("96h")], pd.Timedelta("6h"))
    assert out.to_dict() == {"s1": "train", "s2": "dropped", "s3": "val", "s4": "test"}


def test_region_split():
    assert region_disjoint_split(["IN-GOM", "IN-OD-WB", "IN-GUJ-MH"], ["IN-OD-WB"]) == \
        {"IN-GOM": "train", "IN-OD-WB": "test", "IN-GUJ-MH": "train"}


def test_no_segment_leakage(processed):
    ais = processed["ais"]
    seg = segment_table(ais)
    t0, t1 = ais.timestamp.min(), ais.timestamp.max()
    b1 = t0 + (t1 - t0) * 0.5
    b2 = t0 + (t1 - t0) * 0.75
    split = time_disjoint_split(seg, [b1, b2], pd.Timedelta("2h"))
    ais = ais.assign(split=ais.segment_id.map(split))
    assert check_no_leakage(ais)["segments_unique"]


def test_gap_prism_finds_partner_met_during_blackout():
    from src.graph_builder import gap_prism_pairs
    t0 = pd.Timestamp("2024-01-01", tz="UTC")
    # carrier drifts slowly at (9.0, 79.1) for 3 h; fisher silent 60-240 min, starts/ends ~5 km away
    carrier = make_track("carrier", "2024-01-01", list(range(0, 300, 5)), lat0=9.0, lon0=79.1, dlat=0.0, sog=0.5)
    far = make_track("far", "2024-01-01", list(range(0, 300, 5)), lat0=8.2, lon0=78.4, dlat=0.0, sog=0.5)
    gaps = pd.DataFrame([{"ais_gap_id": "g1", "vessel_id": "fisher", "gap_start": t0 + pd.Timedelta("60min"),
                          "gap_end": t0 + pd.Timedelta("240min"), "gap_duration_min": 180.0,
                          "start_lat": 9.0, "start_lon": 79.055, "end_lat": 9.0, "end_lon": 79.145,
                          "gap_context": "at_sea"}])
    ais = pd.concat([carrier, far])
    q = gap_prism_pairs(ais, gaps, vmax_kn=10, exclude_port_m=0)
    assert list(q.vessel_b) == ["carrier"]
