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


def _is_segment_clear(
    start: tuple[float, float],
    end: tuple[float, float],
    obstacles: list,
    lever_hw: float,
) -> bool:
    """Check if a lever segment clears all obstacles."""
    for obs in obstacles:
        if isinstance(obs, tuple):
            _, corners = obs
        else:
            corners = obs
        dist = segment_to_rect_dist(start, end, corners)
        if dist < lever_hw:
            return False
    return True


def _angle_deviation(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    """Bend angle deviation from straight at point b."""
    dx1 = b[0] - a[0]
    dy1 = b[1] - a[1]
    dx2 = c[0] - b[0]
    dy2 = c[1] - b[1]
    len1 = math.sqrt(dx1*dx1 + dy1*dy1)
    len2 = math.sqrt(dx2*dx2 + dy2*dy2)
    if len1 < 1e-10 or len2 < 1e-10:
        return 0.0
    cos_a = (dx1*dx2 + dy1*dy2) / (len1 * len2)
    cos_a = max(-1.0, min(1.0, cos_a))
    return math.pi - math.acos(cos_a)


def _find_lever_path(
    btn: tuple[float, float],
    pallet: tuple[float, float],
    reed_obstacles: list,
    lever_obstacles: list,
    lever_hw: float,
    max_bend_angle: float = MAX_BEND_ANGLE,
) -> list[tuple[float, float]] | None:
    """Check if a feasible lever path exists (straight or gentle dogleg).

    Returns the path points if feasible, None otherwise.
    This is a lightweight version of the full router, used during placement.
    """
    all_obs = list(reed_obstacles) + list(lever_obstacles)

    # 1. Try straight
    if _is_segment_clear(btn, pallet, all_obs, lever_hw):
        return [btn, pallet]

    # 2. Try single gentle bend
    # Find blocking obstacles
    blocking_centers = []
    for obs in all_obs:
        corners = obs[1] if isinstance(obs, tuple) else obs
        dist = segment_to_rect_dist(btn, pallet, corners)
        if dist < lever_hw:
            cx = float(corners[:, 0].mean())
            cy = float(corners[:, 1].mean())
            half_diag = max(
                math.sqrt((corners[i, 0] - cx)**2 + (corners[i, 1] - cy)**2)
                for i in range(4)
            )
            blocking_centers.append((cx, cy, half_diag))

    if not blocking_centers:
        return [btn, pallet]

    dx = pallet[0] - btn[0]
    dy = pallet[1] - btn[1]
    seg_len = math.sqrt(dx*dx + dy*dy)
    if seg_len < 1e-6:
        return None

    nx = -dy / seg_len
    ny = dx / seg_len

    best_path = None
    best_length = float("inf")

    for bcx, bcy, half_diag in blocking_centers:
        offset = half_diag + lever_hw * 2 + 1.5

        # Try perpendicular + diagonal waypoints
        candidates = [
            (bcx + nx * offset, bcy + ny * offset),
            (bcx - nx * offset, bcy - ny * offset),
        ]
        for angle_deg in range(0, 360, 30):
            angle = math.radians(angle_deg)
            candidates.append((
                bcx + offset * math.cos(angle),
                bcy + offset * math.sin(angle),
            ))

        for wp in candidates:
            path = [btn, wp, pallet]
            # Check angle constraint
            dev = _angle_deviation(btn, wp, pallet)
            if dev > max_bend_angle:
                continue
            # Check both segments clear
            if not _is_segment_clear(btn, wp, all_obs, lever_hw):
                continue
            if not _is_segment_clear(wp, pallet, all_obs, lever_hw):
                continue
            length = math.sqrt((btn[0]-wp[0])**2 + (btn[1]-wp[1])**2) + \
                     math.sqrt((wp[0]-pallet[0])**2 + (wp[1]-pallet[1])**2)
            if length < best_length:
                best_length = length
                best_path = path

    return best_path


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
    placed_lever_corners: list[np.ndarray] = []  # lever path rectangles
    plates: list[ReedPlate | None] = [None] * n
    feasible_count = 0
    infeasible_notes = []

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

                # Check reed doesn't overlap placed levers
                for lc in placed_lever_corners:
                    if rects_overlap(candidate, lc):
                        overlaps = True
                        break
                if overlaps:
                    continue

                # Check lever feasibility (straight or gentle dogleg)
                lever_path = _find_lever_path(
                    (bx, by), (px, py),
                    placed_reed_corners, placed_lever_corners,
                    lever_hw,
                )
                if lever_path is None:
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
                    if any(rects_overlap(candidate, lc) for lc in placed_lever_corners):
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

        # Add this lever's physical footprint as an obstacle for future placements
        if tag in ("OK", "wide"):
            px, py = pallet_position(cx, cy, spec.length, best_plate.phi)
            # Find the actual lever path for this placement
            lpath = _find_lever_path(
                (bx, by), (px, py),
                placed_reed_corners, placed_lever_corners,
                lever_hw,
            )
            if lpath:
                for si in range(len(lpath) - 1):
                    lc = lever_obstacle_corners(lpath[si], lpath[si + 1], lever_hw + clearance)
                    placed_lever_corners.append(lc)

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
