"""India-specific spatial layers for the Gulf of Mannar / Palk Bay pilot (IN-GOM).

Every layer carries a `LayerMeta` record (CRS, source, date, uncertainty).
IMPORTANT: the MPA, maritime-boundary and fishing-ground geometries below are
*approximate, hand-digitised research placeholders* so that the pipeline runs
offline. They must be replaced by authoritative layers (e.g. WDPA / Protected
Planet for MPAs, Marine Regions for the EEZ / maritime boundary) before any
real-data experiment. Nothing is labelled illegal because of these layers;
they only provide distance-to-boundary context features.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString, Point, Polygon, box
from pyproj import Transformer

from .config import GEO_CRS, METRIC_CRS, REGIONS, PILOT_REGION

try:  # offline 1-km land mask (GLOBE DEM derived), pip install global-land-mask
    from global_land_mask import globe as _globe
except ImportError:  # pragma: no cover
    _globe = None


# --------------------------------------------------------------------------
# Metadata
# --------------------------------------------------------------------------
@dataclass
class LayerMeta:
    name: str
    kind: str
    crs: str
    source: str
    date: str
    uncertainty: str
    license: str
    notes: str = ""


# --------------------------------------------------------------------------
# Raw geometry definitions (lon, lat order for shapely)
# --------------------------------------------------------------------------
# Approximate outline of the Gulf of Mannar Marine National Park reef/island
# band between Thoothukudi and Mandapam (islands lie 2-10 km offshore).
_GOM_MNP_APPROX = [
    # coastal side, west -> east
    (78.17, 8.82), (78.18, 8.95), (78.28, 9.05), (78.40, 9.12), (78.60, 9.16),
    (78.80, 9.22), (79.00, 9.28), (79.22, 9.30),
    # seaward side, east -> west
    (79.24, 9.22), (79.10, 9.18), (78.90, 9.10), (78.70, 9.07), (78.50, 9.02),
    (78.38, 8.95), (78.30, 8.87), (78.28, 8.82),
]

# India-Sri Lanka maritime boundary (IMBL), approximate turning points recalled
# from the 1974 (Palk Bay) and 1976 (Gulf of Mannar) agreements, north -> south.
# The first point is an extension to close the polygon outside the pilot box.
_IMBL_APPROX_LATLON = [
    (10.600, 80.800),  # extension (not a treaty point)
    (10.083, 80.050), (9.950, 79.583), (9.669, 79.377), (9.363, 79.512),
    (9.217, 79.533), (9.100, 79.533),
    (9.000, 79.522), (8.897, 79.488), (8.667, 79.303), (8.620, 79.217),
    (8.520, 79.078), (8.370, 78.923), (8.203, 78.895), (7.588, 78.773),
]

# Sea-side reference points for ports / harbours / anchorages (lat, lon).
PORTS = [
    # name, lat, lon, kind, country
    ("Thoothukudi Port", 8.75, 78.22, "commercial_port", "IN"),
    ("Thoothukudi Anchorage", 8.70, 78.30, "anchorage", "IN"),
    ("Tharuvaikulam FH", 8.88, 78.20, "fishing_harbour", "IN"),
    ("Kilakarai", 9.19, 78.79, "fishing_harbour", "IN"),
    ("Mandapam (south)", 9.25, 79.12, "fishing_harbour", "IN"),
    ("Rameswaram (north)", 9.33, 79.31, "fishing_harbour", "IN"),
    ("Jegathapattinam", 9.88, 79.24, "fishing_harbour", "IN"),
    ("Talaimannar", 9.10, 79.69, "ferry_pier", "LK"),
]

# Illustrative fishing grounds (synthetic; not official zones).
FISHING_GROUNDS = {
    "GOM_SHELF": (8.45, 8.75, 78.35, 78.75),
    "GOM_MID": (8.85, 9.00, 78.65, 79.05),
    "PALK_IN": (9.50, 9.85, 79.20, 79.33),
}


def _bbox_poly(lat0, lat1, lon0, lon1) -> Polygon:
    return box(lon0, lat0, lon1, lat1)


# --------------------------------------------------------------------------
# Land mask with distance-to-coast lookup
# --------------------------------------------------------------------------
class LandMask:
    """Offline land/sea mask on a regular grid with a distance-to-coast raster."""

    def __init__(self, bbox, res_deg: float = 0.01, pad_deg: float = 0.3):
        if _globe is None:
            raise ImportError("global-land-mask is required: pip install global-land-mask")
        lat0, lat1, lon0, lon1 = bbox
        self.res = res_deg
        self.lats = np.arange(lat0 - pad_deg, lat1 + pad_deg + 1e-9, res_deg)
        self.lons = np.arange(lon0 - pad_deg, lon1 + pad_deg + 1e-9, res_deg)
        LO, LA = np.meshgrid(self.lons, self.lats)
        self.grid = _globe.is_land(LA, LO)
        from scipy.ndimage import distance_transform_edt
        dy = res_deg * 111_320.0
        dx = res_deg * 111_320.0 * np.cos(np.deg2rad(np.mean(self.lats)))
        # distance from every sea cell to the nearest land cell (0 on land)
        self.coast_dist_m = distance_transform_edt(~self.grid, sampling=(dy, dx))
        # distance from every land cell to the nearest sea cell (0 at sea)
        self.inland_dist_m = distance_transform_edt(self.grid, sampling=(dy, dx))
        self.meta = LayerMeta(
            "land_mask", "raster", GEO_CRS,
            "global-land-mask Python package (GLOBE 30 arc-second DEM, NOAA)",
            "package data; accessed at runtime",
            f"~1 km cells; grid resampled at {res_deg} deg; small islets and "
            "harbour structures may be missed or merged",
            "Package MIT; GLOBE data public domain (verify before redistribution)")

    def is_land(self, lat, lon):
        return _globe.is_land(np.asarray(lat, float), np.asarray(lon, float))

    def _idx(self, lat, lon):
        i = np.clip(np.round((np.asarray(lat) - self.lats[0]) / self.res).astype(int), 0, len(self.lats) - 1)
        j = np.clip(np.round((np.asarray(lon) - self.lons[0]) / self.res).astype(int), 0, len(self.lons) - 1)
        return i, j

    def inland_distance_m(self, lat, lon):
        i, j = self._idx(lat, lon)
        return self.inland_dist_m[i, j]

    def coast_distance_m(self, lat, lon):
        i, j = self._idx(lat, lon)
        return self.coast_dist_m[i, j]


# --------------------------------------------------------------------------
# Layer bundle
# --------------------------------------------------------------------------
class IndiaLayers:
    """All spatial context for one region, in both geographic and metric CRS."""

    def __init__(self, region_id: str = PILOT_REGION, land_res_deg: float = 0.01):
        if region_id != PILOT_REGION:
            raise NotImplementedError(
                f"Layers for {region_id} are not digitised yet; only {PILOT_REGION} "
                "(pilot) is available. Add authoritative layers before enabling it.")
        self.region = REGIONS[region_id]
        self.bbox = self.region.bbox
        lat0, lat1, lon0, lon1 = self.bbox
        self.region_poly = _bbox_poly(*self.bbox)
        self._to_m = Transformer.from_crs(GEO_CRS, METRIC_CRS, always_xy=True)

        self.mpa = {"Gulf of Mannar MNP (approx.)": Polygon(_GOM_MNP_APPROX)}
        imbl_lonlat = [(lon, lat) for lat, lon in _IMBL_APPROX_LATLON]
        self.imbl = LineString(imbl_lonlat)
        # India side of the boundary = polygon closed to the west/north
        self.india_side = Polygon(imbl_lonlat + [(78.7, 7.0), (76.0, 7.0), (76.0, 11.0), (80.8, 11.0)])
        self.ports = pd.DataFrame(PORTS, columns=["name", "lat", "lon", "kind", "country"])
        self.fishing_grounds = {k: _bbox_poly(*b) for k, b in FISHING_GROUNDS.items()}
        self.land = LandMask(self.bbox, res_deg=land_res_deg)

        # metric versions
        self.mpa_m = {k: self.project(g) for k, g in self.mpa.items()}
        self.imbl_m = self.project(self.imbl)
        self.india_side_m = self.project(self.india_side)
        self.region_boundary_m = self.project(self.region_poly.exterior)
        self.ports_m = shapely.points(*self._to_m.transform(self.ports.lon.values, self.ports.lat.values))
        self.grounds_m = {k: self.project(g) for k, g in self.fishing_grounds.items()}

        approx = ("APPROXIMATE hand-digitised placeholder for offline prototyping; "
                  "error of several km possible")
        self.meta: List[LayerMeta] = [
            self.land.meta,
            LayerMeta("Gulf of Mannar Marine National Park (approx.)", "polygon", GEO_CRS,
                      "Hand-digitised from general knowledge of the island/reef chain "
                      "(Thoothukudi-Mandapam)", "2026-09 (digitised)", approx,
                      "n/a (replace with WDPA/Protected Planet polygon; check licence)",
                      "Biosphere-reserve extent is larger; dugong reserve in Palk Bay not yet included"),
            LayerMeta("India-Sri Lanka maritime boundary (approx. IMBL/EEZ)", "line", GEO_CRS,
                      "Turning points recalled from the 1974 and 1976 bilateral agreements",
                      "2026-09 (digitised)", approx + "; northern point is an extension",
                      "n/a (replace with Marine Regions EEZ boundaries v12, CC-BY)",
                      "Used as eez_distance_m in this region (positive = Indian side)"),
            LayerMeta("Ports, harbours, anchorage (sea-side points)", "points", GEO_CRS,
                      "Approximate public locations, snapped seaward", "2026-09",
                      "~1-3 km", "n/a (replace with World Port Index, public domain)"),
            LayerMeta("Fishing grounds (illustrative)", "polygons", GEO_CRS,
                      "SYNTHETIC zones created for the demo", "2026-09",
                      "Not real fishing zones", "n/a"),
        ]

    # ---------------------------------------------------------------- utils
    def project(self, geom):
        return shapely.transform(geom, lambda xy: np.column_stack(self._to_m.transform(xy[:, 0], xy[:, 1])))

    def to_metric(self, lat, lon):
        x, y = self._to_m.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(x), np.asarray(y)

    def metadata_table(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(m) for m in self.meta])

    # ------------------------------------------------------- feature engine
    def features(self, lat, lon) -> pd.DataFrame:
        """Vectorised geospatial context features for arrays of positions."""
        lat = np.asarray(lat, float)
        lon = np.asarray(lon, float)
        x, y = self.to_metric(lat, lon)
        pts = shapely.points(x, y)
        out = {}
        # MPA: signed distance, negative inside
        mpa_d = np.full(len(lat), np.inf)
        in_mpa = np.zeros(len(lat), bool)
        for g in self.mpa_m.values():
            inside = shapely.contains_xy(g, x, y)
            d = shapely.distance(g.exterior, pts)
            mpa_d = np.minimum(mpa_d, np.where(inside, -d, d))
            in_mpa |= inside
        out["in_mpa"] = in_mpa
        out["mpa_distance_m"] = mpa_d
        # maritime boundary: signed, positive on Indian side
        india = shapely.contains_xy(self.india_side_m, x, y)
        d_b = shapely.distance(self.imbl_m, pts)
        out["india_side"] = india
        out["eez_distance_m"] = np.where(india, d_b, -d_b)
        # ports
        dmat = np.stack([shapely.distance(p, pts) for p in self.ports_m], axis=1)
        k = dmat.argmin(axis=1)
        out["port_distance_m"] = dmat[np.arange(len(lat)), k]
        out["nearest_port"] = self.ports.name.values[k]
        # coast / land
        out["coast_distance_m"] = self.land.coast_distance_m(lat, lon)
        out["on_land"] = self.land.is_land(lat, lon)
        # fishing grounds
        ground = np.full(len(lat), None, dtype=object)
        for name, g in self.grounds_m.items():
            ground[shapely.contains_xy(g, x, y)] = name
        out["fishing_ground"] = ground
        # distance to region edge (for region entry/exit context)
        out["region_edge_distance_m"] = shapely.distance(self.region_boundary_m, pts)
        out["in_region"] = shapely.contains_xy(self.region_poly, lon, lat)
        return pd.DataFrame(out)

    def geojson(self) -> Dict:
        """Layers as a GeoJSON FeatureCollection (EPSG:4326) for Folium."""
        feats = []
        for name, g in self.mpa.items():
            feats.append({"type": "Feature", "geometry": shapely.geometry.mapping(g),
                          "properties": {"name": name, "layer": "mpa"}})
        feats.append({"type": "Feature", "geometry": shapely.geometry.mapping(self.imbl),
                      "properties": {"name": "India-Sri Lanka boundary (approx.)", "layer": "imbl"}})
        for name, g in self.fishing_grounds.items():
            feats.append({"type": "Feature", "geometry": shapely.geometry.mapping(g),
                          "properties": {"name": f"{name} (synthetic ground)", "layer": "ground"}})
        return {"type": "FeatureCollection", "features": feats}


@lru_cache(maxsize=4)
def get_layers(region_id: str = PILOT_REGION) -> IndiaLayers:
    """Cached accessor (building the land-mask raster takes ~1 s)."""
    return IndiaLayers(region_id)


def assign_region(lat, lon) -> np.ndarray:
    """Return the region_id for each position (None if outside all regions)."""
    lat = np.asarray(lat, float)
    lon = np.asarray(lon, float)
    out = np.full(len(lat), None, dtype=object)
    for rid, r in REGIONS.items():
        a0, a1, o0, o1 = r.bbox
        m = (lat >= a0) & (lat <= a1) & (lon >= o0) & (lon <= o1) & (out == None)  # noqa: E711
        out[m] = rid
    return out
