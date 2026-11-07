"""Mode A - synthetic AIS generator for the Gulf of Mannar / Palk Bay pilot.

Produces *clearly labelled* synthetic data (`synthetic_demo_*`) so the whole
pipeline can be validated without any real dataset. Nothing here describes
real vessels; identifiers are `SYNTH...` strings that cannot collide with real
MMSI numbers.

Simulated behaviours (brief section 5, Mode A):
  benign : coastal/through transit, fishing-like zig-zag in legal grounds,
           port-to-port ferry (incl. a boundary-crossing ferry), anchorage,
           tug assist (legal encounter), parallel fleet fishing (legal encounter),
           carrier loitering, legal MPA transit while steaming, AIS off in port,
           random coverage gaps, transmitter glitches, one receiver outage.
  suspicious (planted): low-speed rendezvous fishing-vessel <-> carrier
           (often with AIS blackout), AIS blackout with fishing inside the MPA
           or across the maritime boundary followed by reappearance, and
           AIS-on fishing inside the MPA / across the boundary.

Outputs are split into:
  * observed provider-like AIS (`ais_raw_synthetic.parquet`) + vessel registry
  * HIDDEN TRUTH (`hidden_truth/`): every scheduled report incl. untransmitted
    and unreceived ones with true positions, per-minute behaviour, planted
    events and injected data errors. Hidden truth is for evaluation only and
    must never be used as a model input.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import shapely
from shapely.geometry import Polygon, box

from .config import KNOT_MS, SYNTHETIC_CAVEAT, SYNTHETIC_DIR, SyntheticConfig
from .india_layers import IndiaLayers, get_layers

M_PER_DEG = 111_320.0

# Synthetic terrestrial AIS receivers (coverage model only; not real stations)
RECEIVERS = {
    "Thoothukudi": (8.80, 78.14),
    "Vembar": (9.08, 78.36),
    "Mandapam": (9.28, 79.12),
    "Rameswaram": (9.29, 79.30),
    "Thondi": (9.74, 79.02),
    "Kanyakumari": (8.08, 77.54),
    "Kodiyakkarai": (10.28, 79.85),
}

# Scenario zones (lat0, lat1, lon0, lon1) - internal to the simulator
ACROSS_BOUNDARY_ZONES = {"SL_PALK": (9.50, 9.65, 79.52, 79.68), "SL_GOM": (8.80, 8.98, 79.56, 79.72)}
RDV_ZONES = {"RDV_GOM": (8.55, 8.85, 78.95, 79.20), "RDV_PALK": (9.60, 9.80, 79.30, 79.45)}
LOITER_ZONE = (8.60, 8.95, 78.70, 79.15)

ENTRY_SOUTH = [(8.02, 78.60), (8.02, 79.00), (8.02, 79.30)]
ENTRY_PALK_NORTH = (10.15, 79.90)

LANES = {
    "A": [(8.02, 78.60), (8.55, 78.35), (8.70, 78.30)],   # south <-> Thoothukudi anchorage
    "B": [(8.02, 79.30), (8.40, 78.80), (8.70, 78.30)],   # south-east <-> anchorage
    "C": [(8.15, 77.95), (8.10, 78.60), (8.02, 79.45)],   # coastal through-transit
}
FERRY_ROUTES = {
    "F1": [(8.75, 78.22), (8.90, 78.45), (9.05, 78.90), (9.25, 79.12)],               # Thoothukudi - Mandapam (touches MPA)
    "F2": [(9.33, 79.31), (9.40, 79.45), (9.28, 79.62), (9.13, 79.70), (9.10, 79.69)],  # Rameswaram - Talaimannar (crosses boundary)
}
PORT_XY = {
    "Thoothukudi Port": (8.75, 78.22), "Thoothukudi Anchorage": (8.70, 78.30),
    "Tharuvaikulam FH": (8.88, 78.20), "Kilakarai": (9.19, 78.79),
    "Mandapam (south)": (9.25, 79.12), "Rameswaram (north)": (9.33, 79.31),
    "Jegathapattinam": (9.88, 79.24),
}
FISHING_PORT_WEIGHTS = {
    "Thoothukudi Port": 6, "Tharuvaikulam FH": 4, "Kilakarai": 6,
    "Mandapam (south)": 6, "Rameswaram (north)": 6, "Jegathapattinam": 4,
}
PORT_GROUNDS = {
    "Thoothukudi Port": ["GOM_SHELF", "GOM_MID"], "Tharuvaikulam FH": ["GOM_SHELF", "GOM_MID"],
    "Kilakarai": ["GOM_MID", "GOM_SHELF"], "Mandapam (south)": ["GOM_MID"],
    "Rameswaram (north)": ["PALK_IN"], "Jegathapattinam": ["PALK_IN"],
}
PALK_PORTS = {"Rameswaram (north)", "Jegathapattinam"}

SHIP_TYPE_CODE = {"fishing": 30, "tug": 52, "passenger": 60, "cargo": 70,
                  "reefer_carrier": 70, "tanker": 80}


# ==========================================================================
# Small geometry helpers (local flat-earth approximation, fine at <200 km)
# ==========================================================================
def _flat_dist_m(lat1, lon1, lat2, lon2):
    dn = (lat2 - lat1) * M_PER_DEG
    de = (lon2 - lon1) * M_PER_DEG * math.cos(math.radians((lat1 + lat2) / 2))
    return math.hypot(dn, de)


def _bearing(lat1, lon1, lat2, lon2):
    dn = (lat2 - lat1) * M_PER_DEG
    de = (lon2 - lon1) * M_PER_DEG * math.cos(math.radians((lat1 + lat2) / 2))
    return math.degrees(math.atan2(de, dn)) % 360


def _offset(lat, lon, east_m, north_m):
    return (lat + north_m / M_PER_DEG,
            lon + east_m / (M_PER_DEG * math.cos(math.radians(lat))))


def _rand_in_box(rng, b):
    lat0, lat1, lon0, lon1 = b
    return float(rng.uniform(lat0, lat1)), float(rng.uniform(lon0, lon1))


def _box_poly(b):
    return box(b[2], b[0], b[3], b[1])


# ==========================================================================
# Voyage builder: 1-minute kinematic simulation with land avoidance
# ==========================================================================
class VoyageBuilder:
    COLS = ["t_min", "lat", "lon", "sog", "cog", "nav_status", "behavior",
            "suspicious", "event_id"]

    def __init__(self, rng, layers: IndiaLayers, t0: int, lat: float, lon: float,
                 p_nav_fishing: float = 0.5):
        self.rng = rng
        self.land = layers.land
        self.t = int(t0)
        self.lat, self.lon = float(lat), float(lon)
        self.cog = float(rng.uniform(0, 360))
        self.rows: List[tuple] = []
        self.p_nav_fishing = p_nav_fishing
        self.markers: Dict[str, int] = {}

    # -------------------------------------------------------------- basics
    def _is_land(self, lat, lon):
        i, j = self.land._idx(lat, lon)
        return bool(self.land.grid[i, j])

    def _emit(self, sog, cog, nav, behavior, suspicious, event_id):
        self.rows.append((self.t, self.lat, self.lon, max(sog, 0.0), cog % 360, nav,
                          behavior, suspicious, event_id))
        self.t += 1

    def _move(self, heading, speed_kn):
        """Advance one minute; steer around land if needed. Returns heading used."""
        d = speed_kn * KNOT_MS * 60.0
        for dh in (0, 25, -25, 50, -50, 80, -80, 120, -120, 160, -160):
            h = heading + dh
            lat2, lon2 = _offset(self.lat, self.lon, d * math.sin(math.radians(h)),
                                 d * math.cos(math.radians(h)))
            if not self._is_land(lat2, lon2):
                self.lat, self.lon = lat2, lon2
                return h
        return heading  # boxed in: hold position this minute

    # ---------------------------------------------------------- behaviours
    def goto(self, lat, lon, speed_kn, behavior="steaming", suspicious=False,
             event_id=None, nav=0, max_min=3000):
        for _ in range(max_min):
            dist = _flat_dist_m(self.lat, self.lon, lat, lon)
            spd = speed_kn * (1 + self.rng.normal(0, 0.03))
            step = spd * KNOT_MS * 60
            if dist <= step:
                self.lat, self.lon = lat, lon
                self._emit(dist / (KNOT_MS * 60), self.cog, nav, behavior, suspicious, event_id)
                return self
            head = _bearing(self.lat, self.lon, lat, lon) + self.rng.normal(0, 2)
            self.cog = self._move(head, spd)
            self._emit(spd, self.cog, nav, behavior, suspicious, event_id)
        return self

    def follow_waypoints(self, wps, speed_kn, **kw):
        for la, lo in wps:
            self.goto(la, lo, speed_kn, **kw)
        return self

    def zigzag(self, zone: Polygon, duration_min, speed_kn, behavior="fishing",
               suspicious=False, event_id=None):
        """Fishing-like motion: short legs, sharp turns, low speed, kept inside `zone`."""
        cx, cy = zone.centroid.x, zone.centroid.y
        leg_left = 0
        heading = self.rng.uniform(0, 360)
        nav = 7 if self.rng.random() < self.p_nav_fishing else 0
        for _ in range(int(duration_min)):
            if leg_left <= 0:
                heading += self.rng.choice([-1, 1]) * self.rng.uniform(60, 165)
                leg_left = int(self.rng.uniform(10, 40))
            spd = max(0.5, speed_kn * (1 + self.rng.normal(0, 0.12)))
            d = spd * KNOT_MS * 60
            lat2, lon2 = _offset(self.lat, self.lon, d * math.sin(math.radians(heading)),
                                 d * math.cos(math.radians(heading)))
            if not shapely.contains_xy(zone, lon2, lat2):
                heading = _bearing(self.lat, self.lon, cy, cx) + self.rng.uniform(-45, 45)
            heading += self.rng.normal(0, 6)
            self.cog = self._move(heading, spd)
            self._emit(spd, self.cog, nav, behavior, suspicious, event_id)
            leg_left -= 1
        return self

    def drift(self, duration_min, speed_kn, direction, behavior="drifting",
              suspicious=False, event_id=None, nav=0):
        for _ in range(int(duration_min)):
            spd = max(0.0, speed_kn * (1 + self.rng.normal(0, 0.25)))
            self.cog = self._move(direction + self.rng.normal(0, 12), spd)
            self._emit(spd, self.cog, nav, behavior, suspicious, event_id)
        return self

    def anchor(self, duration_min, behavior="anchored", event_id=None):
        alat, alon = self.lat, self.lon
        phase = self.rng.uniform(0, 2 * np.pi)
        for k in range(int(duration_min)):
            ang = phase + k * 2 * np.pi / 720  # swing around anchor over ~12 h
            self.lat, self.lon = _offset(alat, alon, 60 * math.sin(ang), 60 * math.cos(ang))
            self._emit(abs(self.rng.normal(0.1, 0.08)), math.degrees(ang) + 90, 1,
                       behavior, False, event_id)
        self.lat, self.lon = alat, alon
        return self

    def wait_until(self, t_target, behavior="waiting", suspicious=False, event_id=None):
        if t_target > self.t:
            self.drift(t_target - self.t, 0.4, self.rng.uniform(0, 360), behavior,
                       suspicious, event_id)
        return self

    def follow(self, leader: pd.DataFrame, t_from, t_to, east_m, north_m,
               behavior=None, suspicious=None, event_id=None, jitter_m=12.0):
        """Shadow another vessel's simulated track with an offset (encounters, pairs)."""
        # never step backwards in time if this vessel arrived late
        seg = leader[(leader.t_min >= max(t_from, self.t)) & (leader.t_min <= t_to)]
        for r in seg.itertuples(index=False):
            e = east_m + self.rng.normal(0, jitter_m)
            n = north_m + self.rng.normal(0, jitter_m)
            self.lat, self.lon = _offset(r.lat, r.lon, e, n)
            self.t = int(r.t_min)
            self.rows.append((self.t, self.lat, self.lon,
                              max(0.0, r.sog + self.rng.normal(0, 0.1)),
                              (r.cog + self.rng.normal(0, 4)) % 360,
                              r.nav_status if r.nav_status != 1 else 0,
                              behavior or r.behavior,
                              r.suspicious if suspicious is None else suspicious,
                              event_id if event_id is not None else r.event_id))
            self.t += 1
        return self

    def mark(self, name):
        self.markers[name] = self.t
        return self

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.COLS)


# ==========================================================================
# Generator
# ==========================================================================
@dataclass
class _Vessel:
    raw_id: str
    vessel_class: str
    ais_class: str
    home_port: Optional[str]
    length_m: float
    width_m: float
    draft_m: float
    actor: bool = False            # hidden: may perform suspicious scenarios
    network: Optional[int] = None  # hidden: which suspicious carrier an actor works with
    route: Optional[str] = None
    t_free: int = 0
    at_port: bool = True
    p_nav_fishing: float = 0.5
    voyages: List[pd.DataFrame] = field(default_factory=list)
    dark_windows: List[Tuple[int, int, str]] = field(default_factory=list)


class SyntheticAISGenerator:
    def __init__(self, cfg: SyntheticConfig = SyntheticConfig(), layers: Optional[IndiaLayers] = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.layers = layers or get_layers(cfg.region_id)
        self.t_start = pd.Timestamp(cfg.start)
        self.H = int(cfg.n_days * 1440)
        self.events: List[dict] = []
        self.vessels: List[_Vessel] = []
        self.mpa_poly = list(self.layers.mpa.values())[0]
        self.ground_polys = {k: _box_poly(b) for k, b in _grounds().items()}
        self._ev = 0
        self._build_fleet()

    # ------------------------------------------------------------- fleet
    def _new_id(self, k):
        return f"SYNTH{k:04d}{int(self.rng.integers(1000, 9999))}"

    def _build_fleet(self):
        c, r = self.cfg, self.rng
        k = 0
        ports = list(FISHING_PORT_WEIGHTS)
        w = np.array([FISHING_PORT_WEIGHTS[p] for p in ports], float)
        homes = r.choice(ports, size=c.n_fishing, p=w / w.sum())
        actors = set(r.choice(c.n_fishing, size=c.n_fishing_suspicious, replace=False))
        for i in range(c.n_fishing):
            L = float(r.uniform(11, 24))
            self.vessels.append(_Vessel(self._new_id(k), "fishing", "B", str(homes[i]), L,
                                        round(L / 3.6, 1), round(r.uniform(1.5, 3.0), 1),
                                        actor=i in actors, p_nav_fishing=float(r.uniform(0.2, 0.9))))
            k += 1
        sus_c = set(r.choice(c.n_carriers, size=c.n_carriers_suspicious, replace=False))
        for i in range(c.n_carriers):
            L = float(r.uniform(60, 110))
            self.vessels.append(_Vessel(self._new_id(k), "reefer_carrier", "A", None, L,
                                        round(L / 6, 1), round(r.uniform(4, 6.5), 1),
                                        actor=i in sus_c, at_port=False))
            k += 1
        # each suspicious carrier works with its own small network of actor fishing vessels
        act_c = [v for v in self.vessels if v.vessel_class == "reefer_carrier" and v.actor]
        act_f = [v for v in self.vessels if v.vessel_class == "fishing" and v.actor]
        for j, v in enumerate(act_c):
            v.network = j
        for j, v in enumerate(act_f):
            v.network = j % max(1, len(act_c))
        for i in range(c.n_transit):
            cls = "tanker" if r.random() < 0.35 else "cargo"
            L = float(r.uniform(120, 240))
            self.vessels.append(_Vessel(self._new_id(k), cls, "A", "Thoothukudi Port", L,
                                        round(L / 6.5, 1), round(r.uniform(7, 12), 1), at_port=False))
            k += 1
        for i in range(c.n_tugs):
            self.vessels.append(_Vessel(self._new_id(k), "tug", "A", "Thoothukudi Port",
                                        float(r.uniform(25, 35)), 10.0, 4.0))
            k += 1
        for i in range(c.n_ferries):
            route = "F1" if i % 2 == 0 else "F2"
            home = "Thoothukudi Port" if route == "F1" else "Rameswaram (north)"
            self.vessels.append(_Vessel(self._new_id(k), "passenger", "A", home,
                                        float(r.uniform(40, 70)), 12.0, 3.0, route=route))
            k += 1

    def _event(self, etype, label, vessels, t0, t1, lat, lon, **extra):
        self._ev += 1
        eid = f"EV{self._ev:04d}"
        self.events.append(dict(event_id=eid, event_type=etype,
                                synthetic_demo_label=f"synthetic_demo_{label}",
                                vessels=",".join(v.raw_id for v in vessels),
                                t0_min=int(t0), t1_min=int(t1), lat=float(lat), lon=float(lon), **extra))
        return eid

    def _patch_event(self, eid, **kw):
        for e in self.events:
            if e["event_id"] == eid:
                e.update(kw)

    def _builder(self, v, t0, lat, lon):
        return VoyageBuilder(self.rng, self.layers, t0, lat, lon, v.p_nav_fishing)

    def _commit(self, v: _Vessel, b: VoyageBuilder, port_stay=(60, 180), allow_glitch=True):
        df = b.frame()
        if df.empty:
            return df
        df = df[df.t_min < self.H]
        if df.empty:
            return df
        df = df.assign(voyage_id=f"{v.raw_id}_v{len(v.voyages):03d}")
        v.voyages.append(df)
        v.t_free = int(df.t_min.max()) + int(self.rng.uniform(*port_stay))
        if allow_glitch and len(df) > 240 and self.rng.random() < self.cfg.p_equipment_glitch \
                and not df.suspicious.any():
            dur = int(self.rng.uniform(60, 240))
            s = int(self.rng.uniform(df.t_min.min() + 30, max(df.t_min.min() + 31, df.t_min.max() - dur - 30)))
            v.dark_windows.append((s, s + dur, "equipment_glitch"))
            r = df.iloc[np.argmin(np.abs(df.t_min.values - s))]
            self._event("equipment_glitch", "normal", [v], s, s + dur, r.lat, r.lon)
        return df

    def _day_start(self, d, lo_h, hi_h):
        return max(0, int(d * 1440 + self.rng.uniform(lo_h, hi_h) * 60))

    # ------------------------------------------------------ fishing trips
    def _sample_sea(self, b, min_coast_m=2000.0, tries=200):
        """Random point in box b that is at sea on the simulator's land grid and at
        least `min_coast_m` from any coast/islet (prevents vessels being trapped)."""
        land = self.layers.land
        for _ in range(tries):
            la, lo = _rand_in_box(self.rng, b)
            if land.coast_distance_m(la, lo) >= min_coast_m:
                return la, lo
        return _rand_in_box(self.rng, b)

    def _pick_ground(self, v):
        opts = PORT_GROUNDS[v.home_port]
        return opts[0] if len(opts) == 1 or self.rng.random() < 0.75 else opts[1]

    def _ground_point(self, gname):
        return self._sample_sea(_grounds()[gname])

    def _normal_fishing(self, v, t0):
        r = self.rng
        hl, ho = PORT_XY[v.home_port]
        b = self._builder(v, t0, hl, ho)
        for _ in range(int(r.integers(1, 3))):
            g = self._pick_ground(v)
            b.goto(*self._ground_point(g), r.uniform(7, 9), "steaming")
            b.zigzag(self.ground_polys[g], r.uniform(200, 420), r.uniform(2.5, 4))
        b.goto(hl, ho, r.uniform(7, 9), "steaming")
        return b

    def _mpa_local_zone(self, near_lat, near_lon):
        """A ~12 km patch of the MPA close to a home port (for MPA scenarios)."""
        r = self.rng
        minx, miny, maxx, maxy = self.mpa_poly.bounds
        best = None
        for _ in range(400):
            lo, la = r.uniform(minx, maxx), r.uniform(miny, maxy)
            if shapely.contains_xy(self.mpa_poly, lo, la) and self.layers.land.coast_distance_m(la, lo) >= 2000:
                d = _flat_dist_m(la, lo, near_lat, near_lon)
                if 3000 < d < 30000:
                    best = (la, lo)
                    break
        if best is None:
            best = (9.10, 78.80)
        la, lo = best
        patch = box(lo - 0.055, la - 0.04, lo + 0.055, la + 0.04).intersection(self.mpa_poly)
        return patch, best

    def _suspicious_target(self, v):
        """Where an actor goes for a planted suspicious fishing episode."""
        r = self.rng
        if v.home_port in PALK_PORTS:
            z = ACROSS_BOUNDARY_ZONES["SL_PALK"]
            return "cross_boundary", _box_poly(z), self._sample_sea(z)
        if v.home_port == "Mandapam (south)" and r.random() < 0.5:
            z = ACROSS_BOUNDARY_ZONES["SL_GOM"]
            return "cross_boundary", _box_poly(z), self._sample_sea(z)
        hl, ho = PORT_XY[v.home_port]
        patch, pt = self._mpa_local_zone(hl, ho)
        return "mpa", patch, pt

    def _dark_fishing(self, v, t0):
        """Legal fishing -> AIS off -> fishing in MPA / across boundary -> reappear elsewhere."""
        r = self.rng
        hl, ho = PORT_XY[v.home_port]
        g = self._pick_ground(v)
        kind, zone, (tl, to) = self._suspicious_target(v)
        b = self._builder(v, t0, hl, ho)
        b.goto(*self._ground_point(g), r.uniform(7, 9), "steaming")
        b.zigzag(self.ground_polys[g], r.uniform(60, 150), r.uniform(2.5, 4))
        off_lat, off_lon = b.lat, b.lon
        t_off = b.t
        eid = self._event(f"ais_blackout_{kind}_fishing", "suspicious", [v], t_off, t_off, tl, to,
                          target=kind)
        b.goto(tl, to, r.uniform(7, 9), "steaming", True, eid)
        b.zigzag(zone, r.uniform(150, 330), r.uniform(2.5, 4), "fishing", True, eid)
        # reappear at a legal-ground point far from the switch-off point
        for _ in range(50):
            rl, ro = self._ground_point(g)
            if _flat_dist_m(rl, ro, off_lat, off_lon) > 6000:
                break
        b.goto(rl, ro, r.uniform(7, 9), "steaming", True, eid)
        t_on = b.t
        b.zigzag(self.ground_polys[g], r.uniform(60, 150), r.uniform(2.5, 4))
        b.goto(hl, ho, r.uniform(7, 9), "steaming")
        v.dark_windows.append((t_off, t_on, "dark_intentional"))
        self._patch_event(eid, t1_min=t_on, off_lat=off_lat, off_lon=off_lon,
                          on_lat=rl, on_lon=ro, dark_min=t_on - t_off)
        return b

    def _visible_suspicious_fishing(self, v, t0):
        """AIS stays on while fishing inside the MPA / across the boundary."""
        r = self.rng
        hl, ho = PORT_XY[v.home_port]
        kind, zone, (tl, to) = self._suspicious_target(v)
        b = self._builder(v, t0, hl, ho)
        b.goto(tl, to, r.uniform(7, 9), "steaming")
        ts = b.t
        eid = self._event(f"{kind}_fishing_ais_on", "suspicious", [v], ts, ts, tl, to, target=kind)
        b.zigzag(zone, r.uniform(120, 240), r.uniform(2.5, 4), "fishing", True, eid)
        self._patch_event(eid, t1_min=b.t)
        g = self._pick_ground(v)
        b.goto(*self._ground_point(g), r.uniform(7, 9), "steaming")
        b.zigzag(self.ground_polys[g], r.uniform(60, 150), r.uniform(2.5, 4))
        b.goto(hl, ho, r.uniform(7, 9), "steaming")
        return b

    def _fleet_pair(self, lead, foll, t0):
        r = self.rng
        b = self._normal_fishing(lead, t0)
        ldf = b.frame()
        fish = ldf[ldf.behavior == "fishing"]
        eid = self._event("fleet_parallel_fishing", "normal", [lead, foll],
                          fish.t_min.min(), fish.t_min.max(), fish.lat.mean(), fish.lon.mean())
        ldf.loc[ldf.behavior == "fishing", "event_id"] = eid
        b.rows = list(ldf.itertuples(index=False, name=None))
        ang = r.uniform(0, 2 * np.pi)
        dist = r.uniform(250, 600)
        fb = self._builder(foll, int(ldf.t_min.min()), ldf.lat.iloc[0], ldf.lon.iloc[0])
        fb.follow(ldf, ldf.t_min.min(), ldf.t_min.max(), dist * math.cos(ang), dist * math.sin(ang))
        return b, fb

    # ------------------------------------------------------- rendezvous
    def _rendezvous(self, carrier: _Vessel, fisher: _Vessel, day: int):
        r = self.rng
        palk = fisher.home_port in PALK_PORTS
        zone = RDV_ZONES["RDV_PALK" if palk else "RDV_GOM"]
        ml, mo = self._sample_sea(zone)
        entry = ENTRY_PALK_NORTH if palk else ENTRY_SOUTH[int(r.integers(0, 3))]
        hl, ho = PORT_XY[fisher.home_port]
        c_speed, f_speed = r.uniform(9, 12), r.uniform(7, 9)
        c_travel = _flat_dist_m(*entry, ml, mo) / (c_speed * KNOT_MS * 60) * 1.1
        f_travel = _flat_dist_m(hl, ho, ml, mo) / (f_speed * KNOT_MS * 60) * 1.15
        t_meet = max(self._day_start(day, 2, 14),
                     carrier.t_free + c_travel + 30, fisher.t_free + f_travel + 30)
        t_meet = int(t_meet)
        dur = int(r.uniform(90, 240))
        eid = self._event("low_speed_rendezvous", "suspicious", [fisher, carrier],
                          t_meet, t_meet + dur, ml, mo, zone="RDV_PALK" if palk else "RDV_GOM")
        # approaches (both vessels), then synchronise on the later arrival
        cb = self._builder(carrier, int(t_meet - c_travel), *entry)
        cb.goto(ml, mo, c_speed, "transit")
        ang = r.uniform(0, 2 * np.pi)
        sep = r.uniform(40, 150)
        e_off, n_off = sep * math.cos(ang), sep * math.sin(ang)
        fb = self._builder(fisher, int(t_meet - f_travel), hl, ho)
        al, ao = _offset(ml, mo, e_off, n_off)
        fb.goto(al, ao, f_speed, "steaming")
        t_arr_f = fb.t
        t_meet = int(max(t_meet, cb.t, fb.t))
        self._patch_event(eid, t0_min=t_meet, t1_min=t_meet + dur)
        # joint low-speed drift
        cb.wait_until(t_meet, "waiting", True, eid)
        cb.drift(dur, r.uniform(0.3, 1.2), r.uniform(0, 360), "rendezvous", True, eid)
        cdf_meet = cb.frame()
        exit_pt = entry if r.random() < 0.5 else ENTRY_SOUTH[int(r.integers(0, 3))] if not palk else ENTRY_PALK_NORTH
        cb.goto(*exit_pt, c_speed, "transit")
        fb.wait_until(t_meet, "waiting", True, eid)
        fb.follow(cdf_meet, t_meet, t_meet + dur - 1, e_off, n_off, "rendezvous", True, eid)
        t_part = fb.t
        g = self._pick_ground(fisher)
        fb.goto(*self._ground_point(g), f_speed, "steaming")
        fb.zigzag(self.ground_polys[g], r.uniform(60, 150), r.uniform(2.5, 4))
        fb.goto(hl, ho, f_speed, "steaming")
        # AIS behaviour around the meeting
        dark_f = r.random() < 0.6
        dark_c = r.random() < 0.25
        if dark_f:
            s = int(min(t_arr_f, t_meet) - r.uniform(20, 60))
            e = int(t_part + r.uniform(20, 90))
            fisher.dark_windows.append((s, e, "dark_intentional"))
        if dark_c:
            s = int(t_meet - r.uniform(10, 40))
            e = int(t_meet + dur + r.uniform(10, 60))
            carrier.dark_windows.append((s, e, "dark_intentional"))
        self._patch_event(eid, fisher_dark=bool(dark_f), carrier_dark=bool(dark_c))
        return cb, fb

    # ------------------------------------------------------ large vessels
    def _transit_day(self, v, day, assists):
        r = self.rng
        t0 = max(self._day_start(day, 0, 16), v.t_free)
        if v.at_port:  # departure from Thoothukudi
            lane = LANES[str(r.choice(["A", "B"]))]
            b = self._builder(v, t0, *PORT_XY["Thoothukudi Port"])
            b.follow_waypoints(lane[::-1], r.uniform(10, 13), behavior="transit")
            v.at_port = False
            return b
        if r.random() < self.cfg.p_transit_arrival:
            lane = LANES[str(r.choice(["A", "B"]))]
            b = self._builder(v, t0, *lane[0])
            b.follow_waypoints(lane[1:-1], r.uniform(10, 13), behavior="transit")
            al, ao = _offset(*PORT_XY["Thoothukudi Anchorage"], r.uniform(-2500, 2500), r.uniform(-2500, 2500))
            b.goto(al, ao, r.uniform(6, 9), "transit")
            b.anchor(r.uniform(240, 900))
            t_a = b.t
            dur = int(r.uniform(30, 60))
            eid = self._event("tug_assist", "normal", [v], t_a, t_a + dur, b.lat, b.lon)
            b.drift(dur, r.uniform(0.2, 1.0), r.uniform(0, 360), "tug_assist", False, eid)
            b.goto(*PORT_XY["Thoothukudi Port"], r.uniform(4, 6), "transit", event_id=eid)
            assists.append((v, eid, t_a, b))
            v.at_port = True
            return b
        lane = LANES["C"] if r.random() < 0.5 else LANES["C"][::-1]
        b = self._builder(v, t0, *lane[0])
        b.follow_waypoints(lane[1:], r.uniform(11, 14), behavior="transit")
        return b

    def _tug_assist(self, tugs, cargo, eid, t_a, cargo_builder):
        r = self.rng
        cdf = cargo_builder.frame()
        pl, po = PORT_XY["Thoothukudi Port"]
        al, ao = cdf.loc[cdf.t_min == t_a, ["lat", "lon"]].iloc[0]
        travel = _flat_dist_m(pl, po, al, ao) / (9 * KNOT_MS * 60) * 1.2
        free = [t for t in tugs if t.t_free <= t_a - travel - 10]
        if not free:
            self._patch_event(eid, event_type="anchorage_departure_no_tug")
            cdf.loc[cdf.event_id == eid, "event_id"] = None
            return None
        tug = free[0]
        self._patch_event(eid, vessels=f"{cargo.raw_id},{tug.raw_id}")
        tb = self._builder(tug, int(t_a - travel - r.uniform(5, 20)), pl, po)
        ang = r.uniform(0, 2 * np.pi)
        e, n = 45 * math.cos(ang), 45 * math.sin(ang)
        tb.goto(*_offset(al, ao, e, n), 9, "transit")
        tb.wait_until(t_a, "waiting", False, eid)
        tb.follow(cdf, t_a, cdf.t_min.max(), e, n, None, False, eid)
        return tug, tb

    def _carrier_normal(self, v, day):
        r = self.rng
        t0 = max(self._day_start(day, 0, 14), v.t_free)
        entry = ENTRY_SOUTH[int(r.integers(0, 3))]
        b = self._builder(v, t0, *entry)
        if r.random() < 0.7:
            b.goto(*self._sample_sea(LOITER_ZONE), r.uniform(9, 12), "transit")
            eid = self._event("carrier_loitering", "normal", [v], b.t, b.t, b.lat, b.lon)
            b.drift(r.uniform(240, 600), r.uniform(0.3, 1.2), r.uniform(0, 360), "loitering", False, eid)
            self._patch_event(eid, t1_min=b.t)
            b.goto(*ENTRY_SOUTH[int(r.integers(0, 3))], r.uniform(9, 12), "transit")
        else:
            lane = LANES["C"] if r.random() < 0.5 else LANES["C"][::-1]
            b = self._builder(v, t0, *lane[0])
            b.follow_waypoints(lane[1:], r.uniform(10, 12), behavior="transit")
        return b

    def _ferry_day(self, v, day):
        r = self.rng
        route = FERRY_ROUTES[v.route]
        out = []
        t = max(self._day_start(day, 1, 3), v.t_free)
        for leg in (route, route[::-1]):
            b = self._builder(v, t, *leg[0])
            b.follow_waypoints(leg[1:], r.uniform(10, 13), behavior="ferry")
            out.append(b)
            t = b.t + int(r.uniform(90, 180))
        return out

    # ---------------------------------------------------------- schedule
    def simulate(self):
        c, r = self.cfg, self.rng
        fishers = [v for v in self.vessels if v.vessel_class == "fishing"]
        carriers = [v for v in self.vessels if v.vessel_class == "reefer_carrier"]
        transits = [v for v in self.vessels if v.vessel_class in ("cargo", "tanker")]
        tugs = [v for v in self.vessels if v.vessel_class == "tug"]
        ferries = [v for v in self.vessels if v.vessel_class == "passenger"]
        actors = [v for v in fishers if v.actor]

        for day in range(c.n_days):
            busy = set()
            # 1) planted rendezvous (joint scheduling)
            for cv in carriers:
                if cv.actor and r.random() < c.p_carrier_rendezvous:
                    cand = [a for a in actors if a.raw_id not in busy and a.network == cv.network]
                    if not cand:
                        break
                    fv = cand[int(r.integers(0, len(cand)))]
                    cb, fb = self._rendezvous(cv, fv, day)
                    self._commit(cv, cb, (600, 1400), False)
                    self._commit(fv, fb, (180, 600), False)
                    busy |= {cv.raw_id, fv.raw_id}
            # 2) carriers (normal days)
            for cv in carriers:
                if cv.raw_id not in busy and r.random() < 0.6:
                    self._commit(cv, self._carrier_normal(cv, day), (600, 1400))
            # 3) fishing vessels
            free = [v for v in fishers if v.raw_id not in busy]
            r.shuffle(free)
            normal_pool = []
            for v in free:
                if r.random() < c.p_fishing_stay_in_port:
                    continue
                t0 = max(self._day_start(day, -1.5, 8), v.t_free)
                if v.actor:
                    u = r.random()
                    if u < c.p_actor_dark:
                        self._commit(v, self._dark_fishing(v, t0), (180, 600), False)
                        continue
                    if u < c.p_actor_dark + c.p_actor_mpa:
                        self._commit(v, self._visible_suspicious_fishing(v, t0), (180, 600), False)
                        continue
                normal_pool.append(v)
            used = set()
            for v in normal_pool:
                if v.raw_id in used:
                    continue
                mates = [m for m in normal_pool if m.raw_id not in used and m is not v
                         and m.home_port == v.home_port and not m.actor]
                t0 = max(self._day_start(day, -1.5, 8), v.t_free)
                if mates and not v.actor and r.random() < c.p_fleet_pair:
                    m = mates[0]
                    t0 = max(t0, m.t_free)
                    lb, fb = self._fleet_pair(v, m, t0)
                    self._commit(v, lb, (180, 600), False)
                    self._commit(m, fb, (180, 600), False)
                    used |= {v.raw_id, m.raw_id}
                else:
                    self._commit(v, self._normal_fishing(v, t0), (180, 600))
                    used.add(v.raw_id)
            # 4) transits + tug assists
            assists = []
            for v in transits:
                if r.random() < 0.8 or v.at_port:
                    self._commit(v, self._transit_day(v, day, assists), (300, 1200))
            for cargo, eid, t_a, cb in assists:
                res = self._tug_assist(tugs, cargo, eid, t_a, cb)
                if res:
                    self._commit(res[0], res[1], (30, 120))
            # 5) ferries
            for v in ferries:
                for b in self._ferry_day(v, day):
                    self._commit(v, b, (90, 180))
        return self

    # --------------------------------------------------- AIS observation
    def _reception_prob(self, lat, lon, t_min, ais_class, outage=True):
        c = self.cfg
        scale = c.class_b_range_factor if ais_class == "B" else 1.0
        full, mx = c.receiver_full_range_km * scale * 1000, c.receiver_max_range_km * scale * 1000
        o0 = c.outage_start_offset_h * 60
        o1 = o0 + c.outage_duration_h * 60
        p_miss = np.ones(len(lat))
        for name, (rl, ro) in RECEIVERS.items():
            dn = (lat - rl) * M_PER_DEG
            de = (lon - ro) * M_PER_DEG * np.cos(np.deg2rad(lat))
            d = np.hypot(dn, de)
            p = np.clip((mx - d) / (mx - full), 0, 1) * 0.98
            if outage and name in c.outage_receivers:
                if c.outage_every_h > 0:
                    phase = np.mod(t_min - o0, c.outage_every_h * 60)
                    active = (t_min >= o0) & (phase < c.outage_duration_h * 60)
                else:
                    active = (t_min >= o0) & (t_min < o1)
                p = np.where(active, 0.0, p)
            p_miss *= 1 - p
        p_sat = c.p_satellite_class_b if ais_class == "B" else c.p_satellite
        return 1 - p_miss * (1 - p_sat)

    def observe(self):
        c, r = self.cfg, self.rng
        rep_frames = []
        for v in self.vessels:
            if not v.voyages:
                continue
            for vdf in v.voyages:
                t = vdf.t_min.values.astype(float)
                if len(t) < 2:
                    continue
                times = []
                cur = t[0] + r.uniform(0, 1)
                anchored = vdf.nav_status.values == 1
                while cur <= t[-1]:
                    times.append(cur)
                    k = min(int(cur - t[0]), len(t) - 1)
                    if v.ais_class == "B":
                        base = c.report_min_class_b
                    else:
                        base = c.report_min_class_a_anchored if anchored[k] else c.report_min_class_a
                    cur += base * r.uniform(0.85, 1.15)
                times = np.array(times)
                idx = np.clip((times - t[0]).astype(int), 0, len(t) - 1)
                rep = pd.DataFrame({
                    "raw_id": v.raw_id, "voyage_id": vdf.voyage_id.iloc[0], "t_min": times,
                    "lat_true": np.interp(times, t, vdf.lat.values),
                    "lon_true": np.interp(times, t, vdf.lon.values),
                    "sog_true": vdf.sog.values[idx], "cog_true": vdf.cog.values[idx],
                    "nav_status": vdf.nav_status.values[idx], "behavior": vdf.behavior.values[idx],
                    "suspicious": vdf.suspicious.values[idx].astype(bool),
                    "event_id": vdf.event_id.values[idx],
                })
                rep_frames.append(rep)
        rep = pd.concat(rep_frames, ignore_index=True)
        cls = {v.raw_id: v for v in self.vessels}
        # transmission windows (intentional dark / glitches)
        transmit = np.ones(len(rep), bool)
        reason = np.full(len(rep), "ok", dtype=object)
        for v in self.vessels:
            m = rep.raw_id.values == v.raw_id
            for s, e, why in v.dark_windows:
                w = m & (rep.t_min.values >= s) & (rep.t_min.values <= e)
                transmit[w] = False
                reason[w] = why
        # reception
        ais_class = rep.raw_id.map(lambda x: cls[x].ais_class).values
        p_act = np.empty(len(rep))
        p_nom = np.empty(len(rep))
        for k in ("A", "B"):
            m = ais_class == k
            p_act[m] = self._reception_prob(rep.lat_true.values[m], rep.lon_true.values[m], rep.t_min.values[m], k, True)
            p_nom[m] = self._reception_prob(rep.lat_true.values[m], rep.lon_true.values[m], rep.t_min.values[m], k, False)
        received = r.random(len(rep)) < p_act
        lost = transmit & ~received
        reason[lost & (p_nom - p_act > 0.3)] = "receiver_outage"
        reason[lost & (reason == "ok")] = "coverage"
        rep["transmitted"] = transmit
        rep["received"] = received
        rep["observed"] = transmit & received
        rep["miss_reason"] = reason
        rep["reception_prob_model"] = p_act
        rep["timestamp"] = self.t_start + pd.to_timedelta(rep.t_min, unit="min")
        self.truth_reports = rep
        return rep

    def to_raw_ais(self):
        """Provider-like raw AIS with sensor noise and injected data-quality errors."""
        c, r = self.cfg, self.rng
        obs = self.truth_reports[self.truth_reports.observed].copy()
        vmap = {v.raw_id: v for v in self.vessels}
        n = len(obs)
        lat = obs.lat_true.values
        e = r.normal(0, c.pos_noise_m, n)
        nn = r.normal(0, c.pos_noise_m, n)
        sog = np.maximum(0, obs.sog_true.values + r.normal(0, c.sog_noise_kn, n))
        cog_noise = np.where(obs.sog_true.values < 0.5, r.normal(0, 40, n), r.normal(0, c.cog_noise_deg, n))
        cog = (obs.cog_true.values + cog_noise) % 360
        heading = (obs.cog_true.values + r.normal(0, 3, n)) % 360
        is_b = obs.raw_id.map(lambda x: vmap[x].ais_class == "B").values
        heading = np.where(is_b & (r.random(n) < 0.7), 511, np.round(heading))
        heading = np.where(obs.nav_status.values == 1, np.round(r.uniform(0, 360, n)), heading)
        raw = pd.DataFrame({
            "mmsi": obs.raw_id.values,
            "timestamp": obs.timestamp.dt.strftime("%Y-%m-%dT%H:%M:%SZ").values,
            "lat": np.round(lat + nn / M_PER_DEG, 6),
            "lon": np.round(obs.lon_true.values + e / (M_PER_DEG * np.cos(np.deg2rad(lat))), 6),
            "sog": np.round(sog, 1), "cog": np.round(cog, 1), "heading": heading,
            "nav_status": np.where(obs.nav_status.values == 7, 7, obs.nav_status.values).astype(int),
            "ship_type": obs.raw_id.map(lambda x: SHIP_TYPE_CODE[vmap[x].vessel_class]).values,
            "length": obs.raw_id.map(lambda x: round(vmap[x].length_m)).values,
            "width": obs.raw_id.map(lambda x: vmap[x].width_m).values,
            "draught": obs.raw_id.map(lambda x: vmap[x].draft_m).values,
            "source": "synthetic_demo",
        })
        # --- inject data-quality problems (recorded in hidden truth)
        errs = []
        k = r.choice(n, size=int(n * c.p_invalid_coord), replace=False)
        for i, j in enumerate(k):
            if i % 2 == 0:
                raw.loc[raw.index[j], ["lat", "lon"]] = [91.0, raw.lon.iloc[j]]
            else:
                raw.loc[raw.index[j], ["lat", "lon"]] = [0.0, 0.0]
            errs.append(("invalid_coordinate", raw.mmsi.iloc[j], raw.timestamp.iloc[j]))
        pool = np.setdiff1d(np.arange(n), k)
        kj = r.choice(pool, size=max(1, int(n * c.p_position_jump)), replace=False)
        for j in kj:
            raw.loc[raw.index[j], "lat"] = raw.lat.iloc[j] + 0.5
            errs.append(("position_jump", raw.mmsi.iloc[j], raw.timestamp.iloc[j]))
        ks = r.choice(n, size=int(n * c.p_sog_unavailable), replace=False)
        raw.loc[raw.index[ks], "sog"] = 102.3
        for j in ks:
            errs.append(("sog_not_available", raw.mmsi.iloc[j], raw.timestamp.iloc[j]))
        kd = r.choice(n, size=int(n * c.p_duplicate), replace=False)
        dup = raw.iloc[kd]
        for j in kd:
            errs.append(("duplicate", raw.mmsi.iloc[j], raw.timestamp.iloc[j]))
        raw = pd.concat([raw, dup], ignore_index=True)
        raw = raw.sample(frac=1.0, random_state=int(c.seed)).reset_index(drop=True)  # unsorted like a feed
        self.injected_errors = pd.DataFrame(errs, columns=["error_type", "mmsi", "timestamp"])
        self.raw = raw
        return raw

    # ---------------------------------------------------------- outputs
    def registry(self) -> pd.DataFrame:
        return pd.DataFrame([dict(mmsi=v.raw_id, vessel_class=v.vessel_class, ais_class=v.ais_class,
                                  ship_type=SHIP_TYPE_CODE[v.vessel_class], length_m=round(v.length_m, 1),
                                  width_m=v.width_m, home_port=v.home_port, source="synthetic_demo")
                             for v in self.vessels])

    def vessel_truth(self) -> pd.DataFrame:
        return pd.DataFrame([dict(mmsi=v.raw_id, vessel_class=v.vessel_class, actor=v.actor, network=v.network,
                                  n_voyages=len(v.voyages)) for v in self.vessels])

    def events_frame(self) -> pd.DataFrame:
        ev = pd.DataFrame(self.events)
        if ev.empty:
            return ev
        ev = ev[ev.t0_min < self.H].reset_index(drop=True)  # drop events beyond the horizon
        ev["start"] = self.t_start + pd.to_timedelta(ev.t0_min, unit="min")
        ev["end"] = self.t_start + pd.to_timedelta(ev.t1_min, unit="min")
        return ev

    def minute_truth(self) -> pd.DataFrame:
        out = []
        for v in self.vessels:
            for vdf in v.voyages:
                out.append(vdf.assign(raw_id=v.raw_id))
        return pd.concat(out, ignore_index=True)


def _grounds():
    from .india_layers import FISHING_GROUNDS
    return FISHING_GROUNDS


def generate_synthetic_dataset(cfg: SyntheticConfig = SyntheticConfig(), out_dir: Optional[Path] = None,
                               save: bool = True) -> Dict[str, pd.DataFrame]:
    """Run the simulator and (optionally) write Parquet files. Returns all tables."""
    gen = SyntheticAISGenerator(cfg).simulate()
    gen.observe()
    raw = gen.to_raw_ais()
    tables = {
        "ais_raw": raw,
        "registry": gen.registry(),
        "truth_reports": gen.truth_reports,
        "events": gen.events_frame(),
        "vessel_truth": gen.vessel_truth(),
        "injected_errors": gen.injected_errors,
    }
    if save:
        out_dir = Path(out_dir or SYNTHETIC_DIR)
        hid = out_dir / "hidden_truth"
        hid.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(out_dir / "ais_raw_synthetic.parquet", index=False)
        tables["registry"].to_parquet(out_dir / "vessel_registry_synthetic.parquet", index=False)
        for k in ("truth_reports", "events", "vessel_truth", "injected_errors"):
            tables[k].to_parquet(hid / f"{k}.parquet", index=False)
        meta = dict(caveat=SYNTHETIC_CAVEAT, config=cfg.to_dict(), n_raw_rows=len(raw),
                    n_vessels=len(gen.vessels), n_events=len(gen.events),
                    receivers_synthetic=RECEIVERS,
                    note="hidden_truth/ is for evaluation only - never use it as model input")
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    return tables
