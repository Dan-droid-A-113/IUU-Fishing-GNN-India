import numpy as np

from src.india_layers import assign_region


def test_land_sea(layers):
    # Gulf of Mannar open water vs inland Tamil Nadu (Ramanathapuram district)
    assert not layers.land.is_land(9.0, 79.0)
    assert layers.land.is_land(9.5, 78.8)


def test_ports_are_at_sea(layers):
    assert not layers.land.is_land(layers.ports.lat.values, layers.ports.lon.values).any()


def test_boundary_side_and_sign(layers):
    f = layers.features(np.array([9.33, 9.10]), np.array([79.31, 79.69]))  # Rameswaram, Talaimannar
    assert f.india_side.tolist() == [True, False]
    assert f.eez_distance_m.iloc[0] > 0 > f.eez_distance_m.iloc[1]


def test_mpa_signed_distance(layers):
    f = layers.features(np.array([9.12, 8.60]), np.array([78.90, 78.60]))
    assert f.in_mpa.iloc[0] and f.mpa_distance_m.iloc[0] < 0
    assert not f.in_mpa.iloc[1] and f.mpa_distance_m.iloc[1] > 0


def test_distance_is_metric(layers):
    # port feature for a point ~11.1 km due south of Thoothukudi Port
    f = layers.features(np.array([8.75 - 0.1]), np.array([78.22]))
    assert abs(f.port_distance_m.iloc[0] - 11_000) < 1_000


def test_every_layer_has_metadata(layers):
    meta = layers.metadata_table()
    assert {"crs", "source", "uncertainty", "license"} <= set(meta.columns)
    assert meta.crs.notna().all() and (meta.uncertainty.str.len() > 0).all()


def test_region_assignment():
    r = assign_region(np.array([9.0, 20.0, 30.0]), np.array([79.0, 70.0, 79.0]))
    assert list(r) == ["IN-GOM", "IN-GUJ-MH", None]
