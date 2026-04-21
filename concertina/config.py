"""Configuration dataclasses for the concertina solver.

All physical constants, constraints, and tuning parameters live here.
Every module receives its relevant config as a parameter rather than
importing global constants.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class InstrumentSpec:
    """Physical dimensions that define the instrument."""

    button_diameter: float = 6.0        # mm
    h_pitch: float = 16.5              # mm, horizontal center-to-center
    v_pitch: float = 16.5 * math.sqrt(3) / 2  # mm, hexagonal vertical offset (~14.3)
    button_travel: float = 2.5          # mm

    lever_thickness: float = 1.5        # mm (material thickness)
    lever_width_min: float = 3.0        # mm (near button end)
    lever_width_max: float = 5.0        # mm (near pivot)
    lever_material: str = "304 SS"      # label only

    keys_per_side_lh: int = 23
    keys_per_side_rh: int = 29
    min_lever_length: float = 30.0      # mm, prevents steep pallet angle

    @property
    def button_radius(self) -> float:
        return self.button_diameter / 2


@dataclass
class RatioSpec:
    """Leverage ratio rules."""

    target_ratio: float = 2.0          # default lever ratio
    ratio_min: float = 1.8             # hard floor (4.5mm lift / 2.5mm travel)
    ratio_max: float = 2.3             # hard ceiling

    graduated: bool = True             # taper ratio across pitch range
    bass_ratio: float = 2.2            # ratio for lowest notes
    treble_ratio: float = 1.8          # ratio for highest notes

    grace_zone: float = 0.05           # deviation within this has zero penalty


@dataclass
class ClearanceSpec:
    """Clearance rules between components."""

    static_floor: float = 1.2          # mm, hard min: lever to fixed object (button hole)
    dynamic_gap: float = 1.8           # mm, soft target: between moving levers
    pivot_buffer: float = 2.0          # mm, clearance around pivot posts
    min_lever_lever_distance: float = 1.8  # mm, minimum lever-to-lever gap


@dataclass
class CostWeights:
    """Penalty weights for the objective function.

    Set a weight to 0 to disable that penalty term.
    Organized in tiers -- Tier 2/3 default to 0 for incremental development.
    """

    # --- Tier 1: Core (enable from start) ---
    w_reed_collision: float = 1e6      # hard: overlapping reed plates
    w_lever_collision: float = 1e6     # hard: lever intersects button hole
    w_lever_length: float = 0.1        # sum of L^2, penalizes long levers
    w_hex_area: float = 0.01           # convex hull area, pulls reeds inward
    w_ratio_deviation: float = 200.0   # (actual - target)^2 per lever

    # --- Tier 2: Add once Tier 1 produces valid layouts ---
    w_bend: float = 0.0               # per dogleg bend (activate: ~50)
    w_uniformity: float = 0.0         # neighbor ratio variance (activate: ~10)
    w_lever_proximity: float = 0.0    # lever-lever too close (activate: ~2000)
    w_min_length: float = 0.0         # lever below min length (activate: ~1e6)

    # --- Tier 3: Refinements ---
    w_pivot_accessibility: float = 0.0  # pivot under another lever (activate: ~500)
    w_center_of_gravity: float = 0.0    # mass imbalance (activate: ~5)
    w_chamber_proportionality: float = 0.0  # air space vs reed area (activate: ~10)


@dataclass
class SolverBounds:
    """Optimization parameter bounds."""

    r_min: float = 30.0                # mm, minimum reed distance from origin
    r_max: float = 120.0               # mm, maximum reed distance
    theta_min: float = -math.pi        # radians
    theta_max: float = math.pi         # radians
    phi_min: float = -math.pi           # radians, reed plate rotation
    phi_max: float = math.pi            # radians


@dataclass
class ReedDimensions:
    """Reed plate size range for interpolation."""

    bass_length: float = 55.0          # mm, longest reed plate
    bass_width: float = 18.0           # mm
    treble_length: float = 24.0        # mm, shortest reed plate
    treble_width: float = 15.0         # mm
    pallet_offset_ratio: float = 0.2   # pallet at this fraction from plate tip
    bank_depth: float = 20.0           # mm, reed block depth (footprint on reed pan)
    bank_wall_thickness: float = 2.0   # mm, wall between reed slots on a bank


@dataclass
class HexBoundary:
    """Hexagonal reed pan boundary."""

    across_flats: float = 200.0    # mm (~7.9", sized for accordion reeds)
    wall_thickness: float = 3.0    # mm, wood wall around the edge

    @property
    def inner_radius(self) -> float:
        """Usable radius (across flats / 2 minus wall)."""
        return self.across_flats / 2 - self.wall_thickness

    @property
    def outer_radius(self) -> float:
        """Circumradius of the hexagon (across corners / 2)."""
        return self.across_flats / 2 / math.cos(math.pi / 6)


@dataclass
class ConcertinaConfig:
    """Top-level configuration combining all sub-configs."""

    instrument: InstrumentSpec = field(default_factory=InstrumentSpec)
    ratio: RatioSpec = field(default_factory=RatioSpec)
    clearance: ClearanceSpec = field(default_factory=ClearanceSpec)
    weights: CostWeights = field(default_factory=CostWeights)
    bounds: SolverBounds = field(default_factory=SolverBounds)
    reeds: ReedDimensions = field(default_factory=ReedDimensions)
    hex_boundary: HexBoundary = field(default_factory=HexBoundary)

    @classmethod
    def defaults(cls) -> ConcertinaConfig:
        """Create a config with all Beaumont/Holden defaults."""
        return cls()

    def save(self, path: str | Path) -> None:
        """Serialize config to JSON file."""
        path = Path(path)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> ConcertinaConfig:
        """Load config from JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)
        return cls(
            instrument=InstrumentSpec(**data.get("instrument", {})),
            ratio=RatioSpec(**data.get("ratio", {})),
            clearance=ClearanceSpec(**data.get("clearance", {})),
            weights=CostWeights(**data.get("weights", {})),
            bounds=SolverBounds(**data.get("bounds", {})),
            reeds=ReedDimensions(**data.get("reeds", {})),
        )
