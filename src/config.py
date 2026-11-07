"""Central configuration for the India-focused explainable fishing-risk prototype.

Everything that affects results (regions, thresholds, seeds, synthetic-scenario
mix) lives here so that experiments are reproducible and auditable.

All timestamps in this project are UTC. India Standard Time is UTC+05:30.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Tuple

# --------------------------------------------------------------------------
# Paths (resolved relative to the repository root so notebooks work anywhere)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRIC_DIR = OUTPUT_DIR / "metrics"
ALERT_DIR = OUTPUT_DIR / "alerts"

SEED = 42

# Projected CRS used for all metric distance calculations in the pilot.
# UTM zone 44N covers 78E-84E; the pilot box starts at 77.9E, so distortion at
# the western edge is small (<0.1%) and acceptable for a research prototype.
GEO_CRS = "EPSG:4326"
METRIC_CRS = "EPSG:32644"

KNOT_MS = 0.514444  # metres per second in one knot

CAVEAT = (
    "Research alert requiring analyst verification. This score indicates "
    "investigation priority for suspicious or anomalous behaviour; it is not "
    "evidence that any vessel acted illegally."
)
SYNTHETIC_CAVEAT = (
    "SYNTHETIC DEMO DATA: all vessels, tracks and labels are simulated "
    "(label prefix 'synthetic_demo'). They are not observations of real vessels "
    "and must never be presented as evidence of real illegal fishing."
)


# --------------------------------------------------------------------------
# Regions. Region IDs allow region-disjoint evaluation later on.
# bbox = (lat_min, lat_max, lon_min, lon_max) in EPSG:4326.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Region:
    region_id: str
    name: str
    bbox: Tuple[float, float, float, float]
    status: str  # pilot | secondary | optional_separate
    notes: str = ""


REGIONS: Dict[str, Region] = {
    "IN-GOM": Region(
        "IN-GOM",
        "Gulf of Mannar - Palk Bay (south Tamil Nadu coast)",
        (8.0, 10.2, 77.9, 80.2),
        "pilot",
        "Primary pilot corridor: Thoothukudi to Rameswaram, Palk Bay, and the "
        "India-Sri Lanka maritime boundary.",
    ),
    "IN-GUJ-MH": Region(
        "IN-GUJ-MH",
        "Gujarat - Maharashtra (Gulf of Kutch to Konkan)",
        (17.5, 23.5, 67.5, 73.3),
        "secondary",
        "West-coast comparison region. Not used until the pilot works.",
    ),
    "IN-OD-WB": Region(
        "IN-OD-WB",
        "Odisha - West Bengal (northern Bay of Bengal, Sundarbans approaches)",
        (19.0, 22.2, 84.5, 89.2),
        "secondary",
        "East-coast comparison region. Not used until the pilot works.",
    ),
    "IN-ANI": Region(
        "IN-ANI",
        "Andaman and Nicobar Islands",
        (6.5, 14.0, 92.0, 94.5),
        "optional_separate",
        "Island-region experiment; never pooled blindly with mainland traffic.",
    ),
}
PILOT_REGION = "IN-GOM"


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
@dataclass
class PreprocessConfig:
    gap_threshold_min: float = 60.0      # silence longer than this is an AIS-gap event
    long_gap_min: float = 180.0          # "long" gap, used for summaries only
    max_speed_kn: float = 40.0           # implied speed above this is flagged impossible
    drop_impossible: bool = True         # drop the offending (jumped) fixes
    resample_min: int = 10               # optional regular grid for visualisation / models
    max_interp_gap_min: float = 30.0     # never interpolate across longer silences
    port_radius_m: float = 5_000.0       # gap starting/ending this close to a port = port context
    region_edge_buffer_m: float = 10_000.0
    coverage_radius_m: float = 25_000.0  # neighbourhood for concurrent-silence indicator
    coverage_lookback_min: float = 60.0
    fishing_speed_band_kn: Tuple[float, float] = (1.5, 5.0)
    fishing_window: int = 7              # messages in rolling fishing-motion window

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------
# Synthetic data (Mode A)
# --------------------------------------------------------------------------
@dataclass
class SyntheticConfig:
    seed: int = SEED
    region_id: str = PILOT_REGION
    start: str = "2024-02-01T00:00:00Z"
    n_days: int = 7
    # fleet composition
    n_fishing: int = 32
    n_fishing_suspicious: int = 8       # "actors" who sometimes behave suspiciously
    n_carriers: int = 4                 # reefer / fish carriers
    n_carriers_suspicious: int = 2
    n_transit: int = 10                 # cargo / tanker
    n_tugs: int = 2
    n_ferries: int = 2
    # behaviour probabilities (per vessel-day)
    p_fishing_stay_in_port: float = 0.15
    p_fleet_pair: float = 0.12           # benign parallel fishing pair
    p_actor_dark: float = 0.18
    p_actor_mpa: float = 0.12
    p_carrier_rendezvous: float = 0.5    # per suspicious carrier-day
    p_equipment_glitch: float = 0.04     # benign transmitter fault per voyage
    p_transit_arrival: float = 0.35      # transit vessel calls at Thoothukudi via anchorage
    # AIS reporting (after provider down-sampling)
    report_min_class_a: float = 3.0
    report_min_class_a_anchored: float = 6.0
    report_min_class_b: float = 5.0
    # coverage model
    receiver_full_range_km: float = 40.0
    receiver_max_range_km: float = 90.0
    class_b_range_factor: float = 0.8
    p_satellite: float = 0.30          # per-report satellite pick-up, Class A
    p_satellite_class_b: float = 0.10  # Class B is weak and rarely seen from space
    # one scheduled outage of a receiver cluster (benign, affects every vessel near it)
    outage_receivers: Tuple[str, ...] = ("Mandapam", "Rameswaram", "Thondi")
    outage_start_offset_h: float = 30.0  # hours after `start`
    outage_duration_h: float = 6.0
    outage_every_h: float = 0.0          # >0: the outage repeats with this period
    # sensor noise
    pos_noise_m: float = 8.0
    sog_noise_kn: float = 0.15
    cog_noise_deg: float = 2.0
    # data-quality error injection (exercises the cleaning code)
    p_duplicate: float = 0.01
    p_invalid_coord: float = 0.002
    p_position_jump: float = 0.001
    p_sog_unavailable: float = 0.005

    def to_dict(self):
        return asdict(self)


def small_synthetic_config(**overrides) -> SyntheticConfig:
    """A fast configuration for unit tests."""
    base = dict(n_days=2, n_fishing=10, n_fishing_suspicious=4, n_carriers=2,
                n_carriers_suspicious=1, n_transit=4, n_tugs=1, n_ferries=1,
                p_actor_dark=0.5, p_carrier_rendezvous=1.0, outage_start_offset_h=20.0)
    base.update(overrides)
    return SyntheticConfig(**base)


def m2_synthetic_config(**overrides) -> SyntheticConfig:
    """Larger simulation for Milestone 2+ (60 days) so that held-out splits contain
    enough planted suspicious episodes. Four suspicious carrier networks allow
    whole networks to be held out (2 train / 1 validation / 1 test)."""
    base = dict(n_days=60, n_fishing=40, n_fishing_suspicious=12, n_carriers=6,
                n_carriers_suspicious=4, n_transit=12, n_tugs=2, n_ferries=2,
                outage_start_offset_h=30.0, outage_every_h=233.0)
    base.update(overrides)
    return SyntheticConfig(**base)


@dataclass
class WindowConfig:
    """Vessel-window samples shared by all Milestone 2+ models."""
    window_h: int = 6            # one sample = one vessel x 6 h window (UTC-aligned)
    step_min: int = 10           # sequence resolution inside a window (36 steps)
    nn_radius_m: float = 2000.0  # neighbour count radius
    close_m: float = 500.0       # "close encounter" distance
    slow_kn: float = 2.0         # "low speed" threshold

    @property
    def n_steps(self) -> int:
        return self.window_h * 60 // self.step_min


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10           # early stopping on validation PR-AUC
    seeds: Tuple[int, ...] = (0, 1, 2)
    hidden: int = 32
    dropout: float = 0.2
