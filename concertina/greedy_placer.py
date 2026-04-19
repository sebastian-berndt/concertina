"""Greedy sequential reed placement.

Places reeds one at a time (largest first), finding the closest
collision-free position for each given already-placed reeds.
Much faster and more reliable than global optimization for the
initial layout.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString

from concertina.config import ConcertinaConfig
from concertina.hayden_layout import HaydenLayout, HaydenButton
from concertina.reed_specs import ReedSpec, ReedPlate


@dataclass
class PlacementResult:
    """Result of greedy placement."""

    plates: list[ReedPlate]
    feasible_count: int         # levers with clear straight paths
    relaxed_count: int          # placed without lever clearance check
    fallback_count: int         # couldn't find any valid position


def greedy_place(
    layout: HaydenLayout,
    reed_specs: list[ReedSpec],
    config: ConcertinaConfig | None = None,
    min_lever_length: float = 35.0,
    r_range: tuple[float, float] = (45, 130),
    r_step: float = 2.5,
    theta_step_deg: float = 3.0,
    verbose: bool = False,
) -> PlacementResult:
    """Place all reeds using a greedy sequential strategy.

    Algorithm:
    1. Sort reeds by area (largest/bass first)
    2. For each reed, sweep (r, theta) candidates
    3. Check: no reed-reed overlap, lever clears other reeds, lever >= min length
    4. Pick the candidate with the shortest lever
    5. If no fully-clear candidate, relax the lever check

    Args:
        layout: Button layout (positions are fixed).
        reed_specs: Reed specifications sorted by MIDI.
        config: Configuration for clearances and dimensions.
        min_lever_length: Minimum button-to-pallet distance in mm.
        r_range: (min, max) radius to search.
        r_step: Radius increment in mm.
        theta_step_deg: Angle increment in degrees.
        verbose: Print placement progress.

    Returns:
        PlacementResult with positioned plates and stats.
    """
    config = config or ConcertinaConfig.defaults()
    clearance = config.clearance.static_floor
    lever_hw = config.instrument.lever_width_min / 2
    n = len(reed_specs)

    # Button positions by note
    buttons = sorted(layout.get_all_enabled(), key=lambda b: b.midi)
    btn_xy = {spec.note: (btn.x, btn.y) for btn, spec in zip(buttons, reed_specs)}

    # Place largest first
    order = sorted(range(n), key=lambda i: -reed_specs[i].length * reed_specs[i].width)

    placed_polys = []       # buffered reed polygons for collision checking
    placement_map = [None] * n
    feasible_count = 0
    relaxed_count = 0
    fallback_count = 0

    r_min, r_max = r_range
    theta_step = math.radians(theta_step_deg)
    thetas = np.arange(0, 2 * math.pi, theta_step)
    radii = np.arange(r_min, r_max + r_step, r_step)

    for idx in order:
        spec = reed_specs[idx]
        bx, by = btn_xy[spec.note]

        plate, tag = _find_best_position(
            spec, bx, by, placed_polys, radii, thetas,
            clearance, lever_hw, min_lever_length,
            check_lever=True,
        )

        if plate is None:
            # Relaxed: skip lever-reed check
            plate, tag = _find_best_position(
                spec, bx, by, placed_polys, radii, thetas,
                clearance, lever_hw, min_lever_length,
                check_lever=False,
            )
            if plate is not None:
                tag = "relaxed"

        if plate is None:
            # Fallback: place far out
            theta = 2 * math.pi * idx / n
            plate = ReedPlate(spec=spec, r=r_max, theta=theta, phi=theta + math.pi)
            tag = "FALLBACK"
            fallback_count += 1

        if tag == "OK":
            feasible_count += 1
        elif tag == "relaxed":
            relaxed_count += 1

        if verbose:
            px, py = plate.pallet_position
            lever_len = math.sqrt((bx - px)**2 + (by - py)**2)
            print(f"  {spec.note:4s}: r={plate.r:5.1f} "
                  f"θ={math.degrees(plate.theta):6.1f}° "
                  f"lever={lever_len:5.1f}mm [{tag}]")

        placement_map[idx] = plate
        placed_polys.append(plate.get_polygon(clearance=clearance))

    return PlacementResult(
        plates=placement_map,
        feasible_count=feasible_count,
        relaxed_count=relaxed_count,
        fallback_count=fallback_count,
    )


def _find_best_position(
    spec: ReedSpec,
    bx: float,
    by: float,
    placed_polys: list,
    radii: np.ndarray,
    thetas: np.ndarray,
    clearance: float,
    lever_hw: float,
    min_lever_length: float,
    check_lever: bool,
) -> tuple[ReedPlate | None, str]:
    """Search for the best position for one reed plate.

    Returns (plate, tag) or (None, "") if nothing found.
    """
    best_plate = None
    best_lever_len = float("inf")

    for r in radii:
        for theta in thetas:
            cx = r * math.cos(theta)
            cy = r * math.sin(theta)
            phi = math.atan2(by - cy, bx - cx)

            plate = ReedPlate(spec=spec, r=r, theta=theta, phi=phi)

            # Check lever length first (fast reject)
            px, py = plate.pallet_position
            lever_len = math.sqrt((bx - px)**2 + (by - py)**2)
            if lever_len < min_lever_length:
                continue
            if lever_len >= best_lever_len:
                continue  # already have a shorter option

            # Check reed-reed collision
            poly = plate.get_polygon(clearance=clearance)
            if any(poly.intersects(ep) for ep in placed_polys):
                continue

            # Check lever-reed collision (optional)
            if check_lever and placed_polys:
                lever_buf = LineString([(bx, by), (px, py)]).buffer(lever_hw)
                if any(lever_buf.intersects(ep) for ep in placed_polys):
                    continue

            best_lever_len = lever_len
            best_plate = plate

    tag = "OK" if best_plate is not None else ""
    return best_plate, tag
