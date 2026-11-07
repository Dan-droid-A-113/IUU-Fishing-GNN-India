import asyncio
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from src.gfw_real import (anonymize_real, bbox_polygon, fetch_events, month_chunks, normalize_events,
                          sar_near_gaps, vessel_week_tables)


def _ev(i, kind, vid, start, lat=9.0, lon=79.0, hours=3, **extra):
    t0 = datetime.fromisoformat(start).replace(tzinfo=timezone.utc)
    d = {"id": f"{kind}{i}", "type": kind, "start": t0, "end": t0 + pd.Timedelta(hours=hours),
         "position": {"lat": lat, "lon": lon},
         "regions": {"mpa": ["555"] if extra.pop("mpa", False) else [], "eez": [extra.pop("eez", "8480")]},
         "distances": {"start_distance_from_shore_km": 12.0, "start_distance_from_port_km": 30.0},
         "vessel": {"id": vid, "name": "SHIP " + vid, "ssvid": "4190000" + vid[-2:], "flag": "IND", "type": "fishing"}}
    d.update(extra)
    return d


class _Res:
    def __init__(self, rows):
        self.rows = rows

    def data(self):
        class M(dict):
            def model_dump(self):
                return dict(self)
        return [M(r) for r in self.rows]


class FakeEvents:
    def __init__(self, rows):
        self.rows, self.calls = rows, []

    async def get_all_events(self, *, datasets, geometry, start_date, end_date, limit, offset):
        self.calls.append((start_date, offset))
        sel = [r for r in self.rows if start_date <= r["start"].strftime("%Y-%m-%d") < end_date]
        return _Res(sel[offset:offset + limit])


class FakeClient:
    def __init__(self, rows):
        self.events = FakeEvents(rows)


def test_month_chunks():
    assert month_chunks("2025-01-15", "2025-03-02") == [("2025-01-15", "2025-02-01"), ("2025-02-01", "2025-03-01"),
                                                       ("2025-03-01", "2025-03-03")]


def test_bbox_polygon_closed_lonlat():
    p = bbox_polygon((8.0, 10.2, 77.9, 80.2))
    ring = p["coordinates"][0]
    assert ring[0] == ring[-1] and ring[0] == [77.9, 8.0]


def test_fetch_paginates_and_caches(tmp_path):
    rows = [_ev(i, "fishing", "v01", f"2025-01-{1 + i % 28:02d}T0{i % 9}:00:00",
                fishing={"total_distance_km": 5.0, "average_speed_knots": 3.0}) for i in range(450)]
    c = FakeClient(rows)
    df = asyncio.run(fetch_events(c, "fishing", bbox_polygon((8, 10, 78, 80)), "2025-01-01", "2025-01-31",
                                  cache_dir=tmp_path, page=200, verbose=False))
    assert len(df) == 450 and df.event_id.is_unique
    assert [o for _, o in c.events.calls] == [0, 200, 400]
    n_calls = len(c.events.calls)
    asyncio.run(fetch_events(c, "fishing", {}, "2025-01-01", "2025-01-31", cache_dir=tmp_path, verbose=False))
    assert len(c.events.calls) == n_calls  # served from cache


def test_normalize_gap_and_encounter():
    g = normalize_events([_ev(1, "gap", "v01", "2025-02-01T00:00:00", hours=10,
                              gap={"intentional_disabling": True, "distance_km": 40.0, "duration_hours": 10.0,
                                   "implied_speed_knots": 2.2, "positions_per_day_sat_reception": 3.0,
                                   "positions_12_hours_before_sat": 1.0,
                                   "off_position": {"lat": 9.0, "lon": 79.0}, "on_position": {"lat": 9.3, "lon": 79.2}})],
                         "gap")
    r = g.iloc[0]
    assert r.gap_duration_h == 10 and r.intentional_disabling == True and r.on_lat == 9.3  # noqa: E712
    assert r.duration_h == 10 and r.eez_ids == "8480"
    e = normalize_events([_ev(2, "encounter", "v01", "2025-02-02T00:00:00", mpa=True,
                              encounter={"vessel": {"id": "c99", "ssvid": "1", "type": "carrier", "flag": "LKA"},
                                         "median_distance_kilometers": 0.05, "median_speed_knots": 0.4,
                                         "type": "fishing-carrier"})], "encounter")
    assert e.partner_type.iloc[0] == "carrier" and bool(e.in_mpa.iloc[0])


def test_missing_optional_fields_do_not_crash():
    df = normalize_events([{"id": "x", "start": "2025-01-01T00:00:00Z", "end": "2025-01-01T01:00:00Z"}], "gap")
    assert len(df) == 1 and np.isnan(df.gap_distance_km.iloc[0])
    assert normalize_events([], "loitering").empty


def _real_sample():
    rows = {
        "fishing": [_ev(1, "fishing", "v01", "2025-03-03T02:00:00", mpa=True),
                    _ev(2, "fishing", "v02", "2025-03-04T02:00:00", eez="8452")],
        "gap": [_ev(3, "gap", "v01", "2025-03-05T00:00:00", hours=12,
                    gap={"intentional_disabling": True, "distance_km": 30.0, "duration_hours": 12.0,
                         "off_position": {"lat": 9.0, "lon": 79.0}, "on_position": {"lat": 9.2, "lon": 79.1}})],
        "encounter": [_ev(4, "encounter", "v01", "2025-03-06T00:00:00",
                          encounter={"vessel": {"id": "c99", "ssvid": "5", "type": "carrier", "flag": "LKA"},
                                     "median_distance_kilometers": 0.1, "median_speed_knots": 0.5,
                                     "type": "fishing-carrier"})],
        "fishing2": [_ev(5, "fishing", "v01", "2025-03-12T02:00:00")],
    }
    frames = [normalize_events(v, k.rstrip("2")) for k, v in rows.items()]
    return pd.concat(frames, ignore_index=True)


def test_anonymize_removes_identifiers():
    ev, mapping = anonymize_real(_real_sample())
    assert not {"ssvid", "vessel_name", "vessel_gfw_id", "partner_gfw_id"} & set(ev.columns)
    assert ev.vessel_id.str.match(r"^vessel_[0-9a-f]{8}$").all()
    assert ev.partner_id.dropna().str.startswith("vessel_").all()


def test_sar_near_gaps():
    ev = _real_sample()
    gaps = ev[ev.event_type == "gap"]
    sar = pd.DataFrame({"date": pd.to_datetime(["2025-03-05", "2025-03-05", "2025-03-20"], utc=True),
                        "lat": [9.05, 12.0, 9.0], "lon": [79.02, 79.0, 79.0], "detections": [2, 5, 7]})
    assert sar_near_gaps(gaps, sar, 20).tolist() == [2]


def test_vessel_week_tables():
    ev, _ = anonymize_real(_real_sample())
    ev.loc[ev.event_type == "gap", "sar_dark_nearby"] = 2
    nodes, edges = vessel_week_tables(ev, india_eez_ids=["8480"])
    v1 = ev[ev.event_type == "fishing"].vessel_id.iloc[0]
    n1 = nodes[nodes.vessel_id == v1].sort_values("week")
    assert len(n1) == 2                                       # two active weeks
    first = n1.iloc[0]
    assert first.fishing_in_mpa_hours == 3 and first.n_intentional_gaps == 1 and first.n_carrier_encounters == 1
    assert first.sar_dark_near_gaps == 2
    v2 = ev[(ev.event_type == "fishing") & (ev.eez_ids == "8452")].vessel_id.iloc[0]
    assert nodes[nodes.vessel_id == v2].fishing_outside_india_eez_hours.iloc[0] == 3
    enc = edges[edges.edge_type == "encounter"]
    assert len(enc) == 2 and set(enc.src) == set(enc.dst)     # both directions
    carrier = enc.dst[enc.src == v1].iloc[0]
    assert carrier in set(nodes.vessel_id)                    # partner became a node
    tmp = edges[edges.edge_type == "temporal"]
    assert (pd.to_datetime(tmp.prev_week) < pd.to_datetime(tmp.week)).all()


def test_big_page_falls_back_and_months_run_in_parallel(tmp_path):
    rows = [_ev(i, "gap", "v01", f"2025-0{1 + i % 3}-{1 + i % 27:02d}T01:00:00") for i in range(900)]

    class Strict(FakeEvents):
        async def get_all_events(self, *, limit, **kw):
            if limit > 200:
                raise ValueError("422 Unprocessable: limit too large")
            return await super().get_all_events(limit=limit, **kw)

    c = FakeClient(rows)
    c.events = Strict(rows)
    df = asyncio.run(fetch_events(c, "gap", {}, "2025-01-01", "2025-03-31", cache_dir=tmp_path, verbose=False))
    assert len(df) == 900 and df.event_id.is_unique
    assert len(list(tmp_path.glob("events_gap_*.parquet"))) == 3
