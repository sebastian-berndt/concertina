"""Sector-based reed placement.

Assigns each button an angular "lane" and places its reed in that lane,
so levers fan out radially without crossing each other.

Strategy:
1. Create N angular slots evenly distributed around the circle
2. Use linear_sum_assignment to optimally match buttons to slots
   (minimizing angular deviation from each button's natural direction)
3. For each slot, find the optimal radius (closest to button without
   overlapping neighbors)

All geometry uses numpy (geometry.py). No shapely in hot paths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from concertina.config import ConcertinaConfig
from concertina.geometry import (
    rect_corners_buffered,
    rects_overlap,
    segment_to_rect_dist,
    pallet_position,
    lever_obstacle_corners,
)
from concertina.hayden_layout import HaydenLayout
from concertina.reed_specs import ReedSpec, ReedPlate

# Max bend angle for lever feasibility (must match lever_router)
MAX_BEND_ANGLE = math.radians(30)


def _lever_clears_buttons(
    start: tuple[float, float],
    end: tuple[float, float],
    button_circles: list[tuple[tuple[float, float], float]],
    skip_index: int,
    lever_hw: float,
) -> bool:
    """Check if a lever segment clears all button holes (except own)."""
    from concertina.geometry import segment_to_circle_dist
    for i, (center, radius) in enumerate(button_circles):
        if i == skip_index:
            continue
        dist = segment_to_circle_dist(start, end, center, radius)
        if dist < lever_hw:
            return False
    return True


@dataclass
class SectorResult:
    """Result of sector-based placement."""

    plates: list[ReedPlate]
    assignments: list[tuple[str, float]]  # (note, assigned_angle) pairs
    feasible_count: int
    infeasible_notes: list[str]


def sector_place(
    layout: HaydenLayout,
    reed_specs: list[ReedSpec],
    config: ConcertinaConfig | None = None,
    min_lever_length: float = 35.0,
    r_range: tuple[float, float] = (40, 130),
    r_step: float = 2.0,
    angular_margin_deg: float = 2.0,
    verbose: bool = False,
) -> SectorResult:
    """Place reeds using sector-based angular assignment.

    Algorithm:
    1. Compute each button's natural outward angle from grid center
    2. Create 26 angular slots spread around the circle
    3. Assign buttons to slots using linear_sum_assignment (Hungarian algorithm)
       to minimize total angular deviation
    4. For each button-slot pair, sweep radius to find optimal placement
       (shortest lever, no overlap with neighbors)

    Args:
        layout: Button layout.
        reed_specs: Reed specs sorted by MIDI.
        config: Configuration.
        min_lever_length: Minimum lever length in mm.
        r_range: (min, max) radius search range.
        r_step: Radius increment.
        angular_margin_deg: Extra angular spread allowed within a slot.
        verbose: Print progress.

    Returns:
        SectorResult with placed reeds and stats.
    """
    config = config or ConcertinaConfig.defaults()
    clearance = config.clearance.static_floor
    lever_hw = config.instrument.lever_width_min / 2
    n = len(reed_specs)

    buttons = sorted(layout.get_all_enabled(), key=lambda b: b.midi)
    btn_xy = [(b.x, b.y) for b in buttons]

    # --- Step 1: Compute natural angles ---
    btn_angles = np.array([math.atan2(b.y, b.x) for b in buttons])

    # --- Step 2: Create angular slots ---
    # Spread slots evenly, but offset so they align well with the button distribution
    slots = np.linspace(-math.pi, math.pi, n, endpoint=False)
    # Rotate slots to best match the button distribution
    best_offset = _find_best_rotation(btn_angles, slots)
    slots = slots + best_offset
    # Normalize to [-pi, pi]
    slots = (slots + math.pi) % (2 * math.pi) - math.pi

    # --- Step 3: Assign buttons to slots (Hungarian algorithm) ---
    # Cost matrix: angular distance between each button and each slot
    cost_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # Angular distance (wrapping around)
            diff = btn_angles[i] - slots[j]
            diff = (diff + math.pi) % (2 * math.pi) - math.pi
            cost_matrix[i, j] = abs(diff)

            # Penalize assigning large reeds to slots near other buttons
            # (large reeds need more angular space)
            reed_area = reed_specs[i].length * reed_specs[i].width
            cost_matrix[i, j] *= 1.0 + reed_area / 2000.0

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Build assignment: button index -> slot angle
    assigned_angles = np.zeros(n)
    for i, j in zip(row_ind, col_ind):
        assigned_angles[i] = slots[j]

    if verbose:
        print("Angular assignments:")
        for i in range(n):
            btn_deg = math.degrees(btn_angles[i])
            slot_deg = math.degrees(assigned_angles[i])
            diff = abs(btn_deg - slot_deg)
            if diff > 180:
                diff = 360 - diff
            print(f"  {reed_specs[i].note:4s}: button={btn_deg:7.1f}° → slot={slot_deg:7.1f}° (Δ={diff:5.1f}°)")

    # --- Step 4: Place each reed at its assigned angle, sweep radius ---
    # Process largest reeds first for radius search (they need more space)
    order = sorted(range(n), key=lambda i: -reed_specs[i].length * reed_specs[i].width)

    placed_reed_corners: list[tuple[int, np.ndarray]] = []  # (index, reed corners)
    plates: list[ReedPlate | None] = [None] * n
    feasible_count = 0
    infeasible_notes = []

    # Button holes as circular obstacles for lever routing
    btn_radius = config.instrument.button_radius + config.clearance.static_floor
    button_circles = [((b.x, b.y), btn_radius) for b in buttons]

    r_min, r_max = r_range
    radii = np.arange(r_min, r_max + r_step, r_step)
    margin = math.radians(angular_margin_deg)

    for idx in order:
        spec = reed_specs[idx]
        bx, by = btn_xy[idx]
        base_theta = assigned_angles[idx]

        best_plate = None
        best_lever_len = float("inf")

        # Search within the angular margin of the assigned slot
        theta_offsets = np.linspace(-margin, margin, max(1, int(2 * margin / math.radians(1)) + 1))

        for r in radii:
            for dt in theta_offsets:
                theta = base_theta + dt
                cx = r * math.cos(theta)
                cy = r * math.sin(theta)
                phi = math.atan2(by - cy, bx - cx)

                px, py = pallet_position(cx, cy, spec.length, phi)
                lever_len = math.sqrt((bx - px)**2 + (by - py)**2)

                if lever_len < min_lever_length:
                    continue
                if lever_len >= best_lever_len:
                    continue

                # Check reed overlap with placed reeds
                candidate = rect_corners_buffered(
                    cx, cy, spec.length, spec.width, phi, clearance,
                )
                overlaps = False
                for _, existing in placed_reed_corners:
                    if rects_overlap(candidate, existing):
                        overlaps = True
                        break
                if overlaps:
                    continue

                # Check lever clears button holes (not reed plates!)
                if not _lever_clears_buttons(
                    (bx, by), (px, py), button_circles, idx, lever_hw,
                ):
                    continue

                best_lever_len = lever_len
                best_plate = ReedPlate(spec=spec, r=r, theta=theta, phi=phi)

        if best_plate is None:
            # Widen angular search
            wider_thetas = np.linspace(
                base_theta - math.radians(30),
                base_theta + math.radians(30),
                61,
            )
            for r in radii:
                for theta in wider_thetas:
                    cx = r * math.cos(theta)
                    cy = r * math.sin(theta)
                    phi = math.atan2(by - cy, bx - cx)

                    px, py = pallet_position(cx, cy, spec.length, phi)
                    lever_len = math.sqrt((bx - px)**2 + (by - py)**2)

                    if lever_len < min_lever_length:
                        continue
                    if lever_len >= best_lever_len:
                        continue

                    candidate = rect_corners_buffered(
                        cx, cy, spec.length, spec.width, phi, clearance,
                    )
                    if any(rects_overlap(candidate, ec) for _, ec in placed_reed_corners):
                        continue
                    if not _lever_clears_buttons(
                        (bx, by), (px, py), button_circles, idx, lever_hw,
                    ):
                        continue

                    best_lever_len = lever_len
                    best_plate = ReedPlate(spec=spec, r=r, theta=theta, phi=phi)

            if best_plate is not None:
                tag = "wide"
            else:
                theta = base_theta
                best_plate = ReedPlate(spec=spec, r=r_max, theta=theta, phi=theta + math.pi)
                tag = "FALLBACK"
                infeasible_notes.append(spec.note)
        else:
            tag = "OK"
            feasible_count += 1

        if verbose:
            px, py = pallet_position(
                best_plate.r * math.cos(best_plate.theta),
                best_plate.r * math.sin(best_plate.theta),
                spec.length, best_plate.phi,
            )
            lever_len = math.sqrt((bx - px)**2 + (by - py)**2)
            print(f"  {spec.note:4s}: r={best_plate.r:5.1f} "
                  f"θ={math.degrees(best_plate.theta):6.1f}° "
                  f"lever={lever_len:5.1f}mm [{tag}]")

        plates[idx] = best_plate
        cx = best_plate.r * math.cos(best_plate.theta)
        cy = best_plate.r * math.sin(best_plate.theta)
        corners = rect_corners_buffered(
            cx, cy, spec.length, spec.width, best_plate.phi, clearance,
        )
        placed_reed_corners.append((idx, corners))

        # Note: lever-lever collision is handled during routing, not placement.
        # The placer ensures reed-reed clearance and lever-button clearance.
        # The router handles lever-lever clearance incrementally.

    assignments = [(reed_specs[i].note, float(assigned_angles[i])) for i in range(n)]

    return SectorResult(
        plates=plates,
        assignments=assignments,
        feasible_count=feasible_count,
        infeasible_notes=infeasible_notes,
    )


def _find_best_rotation(btn_angles: np.ndarray, slots: np.ndarray) -> float:
    """Find the rotation offset that best aligns slots with button angles."""
    best_offset = 0.0
    best_cost = float("inf")

    for offset_deg in range(0, 360, 5):
        offset = math.radians(offset_deg)
        rotated = (slots + offset + math.pi) % (2 * math.pi) - math.pi

        # Quick cost: sum of min angular distances
        cost = 0.0
        for ba in btn_angles:
            diffs = np.abs((ba - rotated + math.pi) % (2 * math.pi) - math.pi)
            cost += diffs.min()

        if cost < best_cost:
            best_cost = cost
            best_offset = offset

    return best_offset
