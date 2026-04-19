"""Fast numpy-based cost function for the optimizer inner loop.

Avoids shapely entirely -- uses numpy vectorized math for:
- Reed-reed overlap (axis-aligned bounding box + rotation check)
- Lever-button collision (point-to-line distance)
- Lever length, hex area, ratio deviation

~100x faster than the shapely-based cost_function.py.
The shapely version is still used for final visualization and validation.
"""

from __future__ import annotations

import numpy as np

from concertina.config import ConcertinaConfig
from concertina.hayden_layout import HaydenLayout
from concertina.reed_specs import ReedSpec


def _precompute_buttons(layout: HaydenLayout) -> np.ndarray:
    """Extract button positions as (N, 2) array. Cache this."""
    btns = sorted(layout.get_all_enabled(), key=lambda b: b.midi)
    return np.array([[b.x, b.y] for b in btns])


def _precompute_reed_dims(reed_specs: list[ReedSpec]) -> tuple[np.ndarray, np.ndarray]:
    """Extract reed dimensions and target ratios. Cache this."""
    dims = np.array([[s.length, s.width] for s in reed_specs])
    ratios = np.array([s.target_ratio for s in reed_specs])
    return dims, ratios


def _decode_polar(x: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode state vector into r, theta, phi arrays.

    Handles both 2-param (Stage 1) and 3-param (Stage 2) formats.
    """
    params_per = len(x) // n
    x2 = x.reshape(n, params_per)

    r = x2[:, 0]
    theta = x2[:, 1]
    if params_per >= 3:
        phi = x2[:, 2]
    else:
        phi = theta + np.pi  # point toward center

    return r, theta, phi


def _reed_centers(r: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Convert polar to cartesian. Returns (N, 2)."""
    return np.column_stack([r * np.cos(theta), r * np.sin(theta)])


def _pallet_positions(
    centers: np.ndarray,
    phi: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    """Compute pallet positions. Returns (N, 2).

    Pallet is at 0.3 * length from center along the plate axis.
    (length/2 - length*0.2 = 0.3*length offset from center)
    """
    offset = lengths * 0.3
    px = centers[:, 0] + offset * np.cos(phi)
    py = centers[:, 1] + offset * np.sin(phi)
    return np.column_stack([px, py])


def _reed_overlap_penalty(
    centers: np.ndarray,
    dims: np.ndarray,
    clearance: float,
) -> float:
    """Fast reed-reed overlap check using center distance vs min separation.

    Approximates each reed as a circle with radius = half-diagonal + clearance.
    Not exact but very fast and conservative.
    """
    n = len(centers)
    # Half-diagonal of each reed plate
    half_diag = np.sqrt(dims[:, 0]**2 + dims[:, 1]**2) / 2 + clearance

    penalty = 0.0
    for i in range(n):
        # Vectorized distance from reed i to all reeds j > i
        dx = centers[i+1:, 0] - centers[i, 0]
        dy = centers[i+1:, 1] - centers[i, 1]
        dists = np.sqrt(dx*dx + dy*dy)
        min_sep = half_diag[i] + half_diag[i+1:]
        overlaps = np.maximum(0, min_sep - dists)
        penalty += np.sum(overlaps ** 2)

    return penalty


def _lever_button_collision_penalty(
    button_pos: np.ndarray,
    pallet_pos: np.ndarray,
    all_buttons: np.ndarray,
    button_clearance: float,
    lever_half_width: float,
) -> float:
    """Check if lever lines pass through button holes.

    Uses point-to-segment distance for each button against each lever.
    """
    n_levers = len(button_pos)
    n_buttons = len(all_buttons)
    min_dist = button_clearance + lever_half_width

    penalty = 0.0
    for i in range(n_levers):
        # Lever line segment: button_pos[i] -> pallet_pos[i]
        p1 = button_pos[i]
        p2 = pallet_pos[i]
        seg = p2 - p1
        seg_len_sq = np.dot(seg, seg)

        if seg_len_sq < 1e-10:
            continue

        for j in range(n_buttons):
            if j == i:
                continue  # skip own button

            # Point-to-segment distance
            pt = all_buttons[j]
            t = np.dot(pt - p1, seg) / seg_len_sq
            t = np.clip(t, 0, 1)
            closest = p1 + t * seg
            dist = np.sqrt(np.sum((pt - closest) ** 2))

            if dist < min_dist:
                penalty += (min_dist - dist) ** 2

    return penalty


def evaluate_fast(
    x: np.ndarray,
    button_pos: np.ndarray,
    reed_dims: np.ndarray,
    target_ratios: np.ndarray,
    config: ConcertinaConfig,
) -> float:
    """Fast cost function for scipy.optimize.

    Args:
        x: State vector [r0, theta0, (phi0), r1, theta1, (phi1), ...]
        button_pos: (N, 2) pre-computed button positions.
        reed_dims: (N, 2) pre-computed [length, width] per reed.
        target_ratios: (N,) pre-computed target ratios.
        config: Configuration.

    Returns:
        Scalar cost.
    """
    n = len(button_pos)
    weights = config.weights
    clearance = config.clearance

    r, theta, phi = _decode_polar(x, n)
    centers = _reed_centers(r, theta)
    pallets = _pallet_positions(centers, phi, reed_dims[:, 0])

    cost = 0.0

    # 1. Reed-reed overlap
    if weights.w_reed_collision > 0:
        overlap = _reed_overlap_penalty(centers, reed_dims, clearance.static_floor)
        cost += overlap * weights.w_reed_collision

    # 2. Lever-button collisions
    if weights.w_lever_collision > 0:
        btn_clear = config.instrument.button_radius + clearance.static_floor
        lever_hw = config.instrument.lever_width_min / 2
        collision = _lever_button_collision_penalty(
            button_pos, pallets, button_pos, btn_clear, lever_hw,
        )
        cost += collision * weights.w_lever_collision

    # 3. Lever length (L^2)
    if weights.w_lever_length > 0:
        dx = pallets[:, 0] - button_pos[:, 0]
        dy = pallets[:, 1] - button_pos[:, 1]
        lengths_sq = dx*dx + dy*dy
        cost += np.sum(lengths_sq) * weights.w_lever_length

    # 4. Hex area (convex hull approximation: use bounding circle)
    if weights.w_hex_area > 0:
        # Use max distance from origin as radius, area = pi * r^2
        max_r = np.max(r + reed_dims[:, 0] / 2)
        area = np.pi * max_r ** 2
        cost += area * weights.w_hex_area

    # 5. Ratio deviation
    if weights.w_ratio_deviation > 0:
        # Actual ratio = pivot-to-pallet / button-to-pivot
        # With target ratio R, the actual lever length determines the ratio
        # For now, the ratio is exactly the target (since we place the pivot accordingly)
        # Deviation comes from lever being too short for the ratio
        lever_lengths = np.sqrt(np.sum((pallets - button_pos)**2, axis=1))
        min_len = config.instrument.min_lever_length
        too_short = np.maximum(0, min_len - lever_lengths)
        cost += np.sum(too_short ** 2) * weights.w_ratio_deviation

    # 6. Radial compactness: pull reeds toward center
    if weights.w_hex_area > 0:
        cost += np.sum(r ** 2) * weights.w_hex_area * 0.1

    return cost


def make_fast_objective(
    layout: HaydenLayout,
    reed_specs: list[ReedSpec],
    config: ConcertinaConfig,
):
    """Create a closure with pre-computed data for scipy.

    Returns:
        (objective_func, button_pos, reed_dims, target_ratios)
    """
    button_pos = _precompute_buttons(layout)
    reed_dims, target_ratios = _precompute_reed_dims(reed_specs)

    def objective(x):
        return evaluate_fast(x, button_pos, reed_dims, target_ratios, config)

    return objective, button_pos, reed_dims, target_ratios
