import numpy as np
import pandas as pd

from src.config import small_synthetic_config
from src.synthetic import generate_synthetic_dataset


def test_deterministic():
    a = generate_synthetic_dataset(small_synthetic_config(n_days=1), save=False)["ais_raw"]
    b = generate_synthetic_dataset(small_synthetic_config(n_days=1), save=False)["ais_raw"]
    pd.testing.assert_frame_equal(a, b)


def test_labels_are_explicitly_synthetic(synth):
    assert synth["events"].synthetic_demo_label.str.startswith("synthetic_demo_").all()
    assert (synth["ais_raw"].source == "synthetic_demo").all()


def test_ids_cannot_be_real_mmsi(synth):
    assert synth["ais_raw"].mmsi.str.startswith("SYNTH").all()


def test_planted_events_present(synth):
    types = set(synth["events"].event_type)
    assert "low_speed_rendezvous" in types
    assert any(t.startswith("ais_blackout_") for t in types)
    assert "tug_assist" in types or "anchorage_departure_no_tug" in types
    assert (synth["events"].synthetic_demo_label == "synthetic_demo_suspicious").any()
    assert (synth["events"].synthetic_demo_label == "synthetic_demo_normal").any()


def test_hidden_truth_separate_and_consistent(synth):
    tr = synth["truth_reports"]
    assert "lat_true" in tr and "miss_reason" in tr
    # observed raw rows (before injected duplicates) == observed truth reports
    n_dup = (synth["injected_errors"].error_type == "duplicate").sum()
    assert len(synth["ais_raw"]) == tr.observed.sum() + n_dup
    # untransmitted reports are never observed
    assert not (tr.observed & ~tr.transmitted).any()
    # suspicious-label columns are absent from the observable AIS
    assert not {"suspicious", "behavior", "event_id"} & set(synth["ais_raw"].columns)


def test_tracks_at_sea_and_in_region(synth, layers):
    tr = synth["truth_reports"]
    on_land = layers.land.is_land(tr.lat_true.values, tr.lon_true.values).mean()
    assert on_land < 0.01
    lat0, lat1, lon0, lon1 = layers.bbox
    assert tr.lat_true.between(lat0, lat1).all() and tr.lon_true.between(lon0, lon1).all()


def test_blackouts_really_silent(synth):
    tr = synth["truth_reports"]
    dark = tr[tr.miss_reason == "dark_intentional"]
    assert len(dark) > 0 and not dark.observed.any()
