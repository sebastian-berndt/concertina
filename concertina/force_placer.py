"""Global reed placement via smooth optimization.

Models all reed positions as variables and minimizes a smooth objective:
- Reed-reed overlap penalty (squared penetration depth)
- Hex boundary penalty (squared distance outside)
- Lever length consistency (squared deviation from target)
- Compactness (pull toward center)

Uses scipy.optimize.minimize (L-BFGS-B) with analytical-ish gradients
approximated by the optimizer. Starts from sector_placer initial guess.

All geometry uses numpy. No shapely.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from concertina.config import ConcertinaConfig
from concertina.geometry import (
    rect_corners,
    rect_corners_buffered,
    rects_overlap,
    rect_in_hexagon,
    point_in_hexagon,
    pallet_position,
)
from concertina.hayden_layout import HaydenLayout
from concertina.reed_specs import ReedSpec, ReedPlate


@dataclass
class ForceResult:
    """Result of optimized placement."""

    plates: list[ReedPlate]
    iterations: int
    final_cost: float
    lever_lengths: list[float]


def force_place(
    layout: HaydenLayout,
    reed_specs: list[ReedSpec],
    initial_plates: list[ReedPlate],
    config: ConcertinaConfig | None = None,
    target_lever_length: float = 60.0,
    w_overlap: float = 1000.0,
    w_boundary: float = 500.0,
    w_lever: float = 1.0,
    w_compact: float = 0.01,
    verbose: bool = False,
) -> ForceResult:
    """Optimize reed positions using smooth global objective.

    Args:
        layout: Button positions (fixed).
        reed_specs: Reed dimensions.
        initial_plates: Starting positions (e.g. from sector_placer).
        config: Configuration.
        target_lever_length: Target lever length for consistency.
        w_overlap: Weight for reed-reed overlap penalty.
        w_boundary: Weight for hex boundary violation.
        w_lever: Weight for lever length deviation.
        w_compact: Weight for pulling reeds toward center.
        verbose: Print progress.

    Returns:
        ForceResult with optimized positions.
    """
    config = config or ConcertinaConfig.defaults()
    clearance = config.clearance.static_floor
    n = len(reed_specs)

    buttons = sorted(layout.get_all_enabled(), key=lambda b: b.midi)
    btn_xy = np.array([[b.x, b.y] for b in buttons])

    hex_af = config.hex_boundary.across_flats - 2 * config.hex_boundary.wall_thickness
    dims = np.array([[s.length, s.width] for s in reed_specs])

    # Initial state: (cx, cy, phi) per reed, flattened
    x0 = np.zeros(n * 3)
    for i, plate in enumerate(initial_plates):
        cx, cy = plate.center
        x0[i * 3] = cx
        x0[i * 3 + 1] = cy
        x0[i * 3 + 2] = plate.phi

    # Bounds
    r_max = hex_af / 2 + 20  # allow slight overshoot, penalty handles it
    bounds = []
    for i in range(n):
        bounds.append((-r_max, r_max))  # cx
        bounds.append((-r_max, r_max))  # cy
        bounds.append((-math.pi, math.pi))  # phi

    iter_count = [0]

    def objective(x):
        state = x.reshape(n, 3)
        cost = 0.0

        # 1. Reed-reed overlap penalty (smooth projected overlap)
        for i in range(n):
            for j in range(i + 1, n):
                dx = state[j, 0] - state[i, 0]
                dy = state[j, 1] - state[i, 1]
                dist = math.sqrt(dx * dx + dy * dy + 1e-6)

                # Project both rects onto the axis connecting their centers
                # to get a smooth, direction-aware overlap estimate
                if dist > 1e-6:
                    ax, ay = dx / dist, dy / dist
                else:
                    ax, ay = 1.0, 0.0

                # Half-extent of each rect along the connecting axis
                # For a rotated rect, the projection is |cos(a)*L/2| + |sin(a)*W/2|
                # where a is the angle between the axis and the rect's orientation
                def _half_extent(phi, length, width):
                    ca = abs(math.cos(phi) * ax + math.sin(phi) * ay)
                    sa = abs(-math.sin(phi) * ax + math.cos(phi) * ay)
                    return ca * length / 2 + sa * width / 2

                he_i = _half_extent(state[i, 2], dims[i, 0] + 2 * clearance, dims[i, 1] + 2 * clearance)
                he_j = _half_extent(state[j, 2], dims[j, 0] + 2 * clearance, dims[j, 1] + 2 * clearance)

                min_sep = he_i + he_j
                penetration = max(0, min_sep - dist)
                cost += penetration ** 2 * w_overlap

        # 2. Hex boundary penalty
        for i in range(n):
            ci = rect_corners_buffered(
                state[i, 0], state[i, 1],
                dims[i, 0], dims[i, 1],
                state[i, 2], clearance,
            )
            for corner in ci:
                if not point_in_hexagon(corner[0], corner[1], hex_af):
                    # Distance outside: approximate as radial overshoot
                    r = math.sqrt(corner[0] ** 2 + corner[1] ** 2)
                    overshoot = max(0, r - hex_af / 2)
                    cost += overshoot ** 2 * w_boundary

        # 3. Lever length consistency
        for i in range(n):
            px, py = pallet_position(
                state[i, 0], state[i, 1],
                dims[i, 0], state[i, 2],
            )
            lever_len = math.sqrt(
                (btn_xy[i, 0] - px) ** 2 + (btn_xy[i, 1] - py) ** 2 + 1e-6
            )
            cost += (lever_len - target_lever_length) ** 2 * w_lever

        # 4. Compactness: pull toward center
        for i in range(n):
            r_sq = state[i, 0] ** 2 + state[i, 1] ** 2
            cost += r_sq * w_compact

        iter_count[0] += 1
        if verbose and iter_count[0] % 50 == 0:
            overlaps = _count_overlaps_fast(state, dims, clearance)
            inside = _count_inside_hex(state, dims, clearance, hex_af)
            levers = _lever_lengths(state, dims, btn_xy)
            print(f"  Iter {iter_count[0]:4d}: cost={cost:.1f}, "
                  f"overlaps={overlaps}, inside={inside}/{n}, "
                  f"lever={np.mean(levers):.0f}±{np.std(levers):.0f}mm")

        return cost

    if verbose:
        print("Optimizing reed placement...")

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": 500,
            "ftol": 1e-8,
        },
    )

    if verbose:
        print(f"  Done: {result.nit} iterations, cost={result.fun:.1f}, success={result.success}")

    # Build result
    state = result.x.reshape(n, 3)
    plates = []
    lever_lengths = []
    for i, spec in enumerate(reed_specs):
        cx, cy, phi = state[i]
        r = math.sqrt(cx * cx + cy * cy)
        theta = math.atan2(cy, cx)
        plates.append(ReedPlate(spec=spec, r=max(0.1, r), theta=theta, phi=phi))

        px, py = pallet_position(cx, cy, dims[i, 0], phi)
        lever_lengths.append(
            math.sqrt((btn_xy[i, 0] - px) ** 2 + (btn_xy[i, 1] - py) ** 2)
        )

    return ForceResult(
        plates=plates,
        iterations=result.nit,
        final_cost=float(result.fun),
        lever_lengths=lever_lengths,
    )


def _count_overlaps_fast(state, dims, clearance):
    n = len(state)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = state[j, 0] - state[i, 0]
            dy = state[j, 1] - state[i, 1]
            dist = math.sqrt(dx * dx + dy * dy)
            half_diag_i = math.sqrt(dims[i, 0] ** 2 + dims[i, 1] ** 2) / 2 + clearance
            half_diag_j = math.sqrt(dims[j, 0] ** 2 + dims[j, 1] ** 2) / 2 + clearance
            if dist < (half_diag_i + half_diag_j) * 0.7:
                count += 1
    return count


def _count_inside_hex(state, dims, clearance, hex_af):
    count = 0
    for i in range(len(state)):
        ci = rect_corners_buffered(state[i, 0], state[i, 1], dims[i, 0], dims[i, 1], state[i, 2], clearance)
        if rect_in_hexagon(ci, hex_af):
            count += 1
    return count


def _lever_lengths(state, dims, btn_xy):
    lengths = []
    for i in range(len(state)):
        px, py = pallet_position(state[i, 0], state[i, 1], dims[i, 0], state[i, 2])
        lengths.append(math.sqrt((btn_xy[i, 0] - px) ** 2 + (btn_xy[i, 1] - py) ** 2))
    return np.array(lengths)
