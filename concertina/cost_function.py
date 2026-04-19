"""Objective function for the concertina layout optimizer.

Evaluates a state vector X (reed positions in polar coordinates)
and returns a scalar cost. Lower is better.

Organized in tiers:
- Tier 1: Core penalties (always active)
- Tier 2: Refinements (activate via config weights)
- Tier 3: Production polish (activate via config weights)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from shapely.ops import unary_union

from concertina.config import ConcertinaConfig
from concertina.hayden_layout import HaydenLayout
from concertina.reed_specs import ReedSpec, ReedPlate
from concertina.obstacles import ObstacleField
from concertina.lever_router import LeverRouter, LeverPath, route_all_levers


@dataclass
class CostBreakdown:
    """Detailed breakdown of all cost terms."""

    # Tier 1
    reed_collision: float = 0.0
    lever_collision: float = 0.0
    lever_length: float = 0.0
    hex_area: float = 0.0
    ratio_deviation: float = 0.0

    # Tier 2
    bend: float = 0.0
    uniformity: float = 0.0
    lever_proximity: float = 0.0
    min_length: float = 0.0

    # Tier 3
    pivot_accessibility: float = 0.0
    center_of_gravity: float = 0.0
    chamber_proportionality: float = 0.0

    @property
    def total(self) -> float:
        return (
            self.reed_collision
            + self.lever_collision
            + self.lever_length
            + self.hex_area
            + self.ratio_deviation
            + self.bend
            + self.uniformity
            + self.lever_proximity
            + self.min_length
            + self.pivot_accessibility
            + self.center_of_gravity
            + self.chamber_proportionality
        )


def decode_state(
    x: np.ndarray,
    reed_specs: list[ReedSpec],
) -> list[ReedPlate]:
    """Unpack the flat state vector into ReedPlate objects.

    Args:
        x: Flat array. For Stage 1 (2 params/reed): [r0, theta0, r1, theta1, ...].
            For Stage 2 (3 params/reed): [r0, theta0, phi0, r1, theta1, phi1, ...].
        reed_specs: Reed specifications (determines count and target ratios).

    Returns:
        List of positioned ReedPlate objects.
    """
    n = len(reed_specs)
    params_per_reed = len(x) // n

    plates = []
    for i, spec in enumerate(reed_specs):
        offset = i * params_per_reed
        r = x[offset]
        theta = x[offset + 1]
        if params_per_reed >= 3:
            phi = x[offset + 2]
        else:
            # Stage 1: phi defaults to "point toward center"
            phi = theta + np.pi
        plates.append(ReedPlate(spec=spec, r=r, theta=theta, phi=phi))

    return plates


def evaluate(
    x: np.ndarray,
    layout: HaydenLayout,
    reed_specs: list[ReedSpec],
    config: ConcertinaConfig,
) -> float:
    """The objective function J(X) for scipy.optimize.

    Args:
        x: Flat state vector of reed positions.
        layout: Fixed button layout.
        reed_specs: Reed plate specifications.
        config: Full configuration (weights, clearances, etc.).

    Returns:
        Scalar cost value. Lower is better.
    """
    return evaluate_detailed(x, layout, reed_specs, config).total


def evaluate_detailed(
    x: np.ndarray,
    layout: HaydenLayout,
    reed_specs: list[ReedSpec],
    config: ConcertinaConfig,
) -> CostBreakdown:
    """Evaluate with full cost breakdown for diagnostics.

    Same inputs as evaluate(), returns CostBreakdown instead of scalar.
    """
    weights = config.weights
    breakdown = CostBreakdown()

    # --- Decode state into geometry ---
    plates = decode_state(x, reed_specs)

    # --- Build obstacle field and route levers ---
    obstacle_field = ObstacleField(layout, plates, config)
    lever_paths = route_all_levers(layout, plates, reed_specs, obstacle_field, config)

    # =======================================================
    # TIER 1: Core penalties (always active)
    # =======================================================

    # 1. Reed-reed collisions
    if weights.w_reed_collision > 0:
        collisions = obstacle_field.check_reed_reed_collisions()
        total_overlap = sum(area for _, _, area in collisions)
        breakdown.reed_collision = total_overlap * weights.w_reed_collision

    # 2. Lever-obstacle collisions (infeasible levers)
    if weights.w_lever_collision > 0:
        infeasible_count = sum(1 for lp in lever_paths if not lp.is_feasible)
        breakdown.lever_collision = infeasible_count * weights.w_lever_collision

    # 3. Lever length (sum of L^2)
    if weights.w_lever_length > 0:
        length_cost = sum(lp.total_length ** 2 for lp in lever_paths)
        breakdown.lever_length = length_cost * weights.w_lever_length

    # 4. Hex area (convex hull of all reed plates)
    if weights.w_hex_area > 0:
        all_polys = [p.get_polygon() for p in plates]
        hull = unary_union(all_polys).convex_hull
        breakdown.hex_area = hull.area * weights.w_hex_area

    # 5. Ratio deviation
    if weights.w_ratio_deviation > 0:
        ratio_cost = 0.0
        for lp, spec in zip(lever_paths, reed_specs):
            deviation = lp.actual_ratio - spec.target_ratio
            # Grace zone: no penalty for small deviations
            grace = config.ratio.grace_zone
            if abs(deviation) > grace:
                ratio_cost += (abs(deviation) - grace) ** 2
        breakdown.ratio_deviation = ratio_cost * weights.w_ratio_deviation

    # =======================================================
    # TIER 2: Refinements (weights default to 0)
    # =======================================================

    # 6. Bend penalty
    if weights.w_bend > 0:
        total_bends = sum(max(0, lp.segments - 1) for lp in lever_paths)
        breakdown.bend = total_bends * weights.w_bend

    # 7. Uniformity (neighbor ratio consistency)
    if weights.w_uniformity > 0:
        buttons_sorted = sorted(layout.get_all_enabled(), key=lambda b: b.midi)
        uniformity_cost = 0.0
        for i, btn in enumerate(buttons_sorted):
            neighbors = layout.get_neighbors(btn)
            for nbr in neighbors:
                # Find the neighbor's lever path index
                for j, other_btn in enumerate(buttons_sorted):
                    if other_btn.note == nbr.note:
                        diff = lever_paths[i].actual_ratio - lever_paths[j].actual_ratio
                        uniformity_cost += diff ** 2
                        break
        breakdown.uniformity = uniformity_cost * weights.w_uniformity

    # 8. Lever-lever proximity
    if weights.w_lever_proximity > 0:
        lever_lines = [lp.path for lp in lever_paths]
        violations = obstacle_field.check_lever_lever_distance(lever_lines)
        breakdown.lever_proximity = len(violations) * weights.w_lever_proximity

    # 9. Minimum lever length
    if weights.w_min_length > 0:
        min_len = config.instrument.min_lever_length
        for lp in lever_paths:
            if lp.total_length < min_len:
                breakdown.min_length += (min_len - lp.total_length) ** 2 * weights.w_min_length

    # =======================================================
    # TIER 3: Production polish (weights default to 0)
    # =======================================================

    # 10. Pivot accessibility
    if weights.w_pivot_accessibility > 0:
        for i, lp_a in enumerate(lever_paths):
            for j, lp_b in enumerate(lever_paths):
                if i == j:
                    continue
                from shapely.geometry import Point
                pivot_pt = Point(lp_a.pivot_pos)
                lever_shape = lp_b.path.buffer(config.instrument.lever_width_min / 2)
                if lever_shape.contains(pivot_pt):
                    breakdown.pivot_accessibility += weights.w_pivot_accessibility

    # 11. Center of gravity
    if weights.w_center_of_gravity > 0:
        total_mass = 0.0
        cx, cy = 0.0, 0.0
        for plate in plates:
            mass = plate.spec.length * plate.spec.width  # proportional to area
            pcx, pcy = plate.center
            cx += pcx * mass
            cy += pcy * mass
            total_mass += mass
        if total_mass > 0:
            cx /= total_mass
            cy /= total_mass
            dist_sq = cx ** 2 + cy ** 2
            breakdown.center_of_gravity = dist_sq * weights.w_center_of_gravity

    # 12. Chamber proportionality (placeholder -- needs more thought)
    # Skipped for now; requires defining "air space" around each reed.

    return breakdown
