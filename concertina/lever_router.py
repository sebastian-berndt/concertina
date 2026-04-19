"""Lever routing from buttons to pallets.

v1: Try straight line first, then single-bend dogleg.
v2 (future): Full visibility graph + Dijkstra for complex routing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely.geometry import Point, LineString, MultiPoint

from concertina.config import ConcertinaConfig
from concertina.obstacles import ObstacleField


@dataclass
class LeverPath:
    """Result of routing one lever."""

    button_pos: tuple[float, float]
    pallet_pos: tuple[float, float]
    pivot_pos: tuple[float, float]
    path: LineString               # centerline of the lever
    segments: int                  # 1 = straight, 2+ = dogleg
    total_length: float            # mm
    actual_ratio: float            # pallet-side / button-side length
    is_feasible: bool              # no hard-constraint violations


class LeverRouter:
    """Routes levers from buttons to pallets, avoiding obstacles.

    v1 strategy:
    1. Try straight line -- if it clears all obstacles, use it.
    2. If straight fails, try single-bend doglegs via tangent points
       on the blocking obstacle.
    3. If no dogleg works, mark as infeasible.
    """

    def __init__(
        self,
        obstacle_field: ObstacleField,
        config: ConcertinaConfig | None = None,
    ):
        self.obstacles = obstacle_field
        self.config = config or ConcertinaConfig.defaults()
        self._lever_width = self.config.instrument.lever_width_min

    def route(
        self,
        button_pos: tuple[float, float],
        pallet_pos: tuple[float, float],
        target_ratio: float,
        lever_index: int,
    ) -> LeverPath:
        """Find a lever path from button to pallet.

        Args:
            button_pos: (x, y) of the button center.
            pallet_pos: (x, y) of the pallet hole.
            target_ratio: Desired leverage ratio.
            lever_index: Index of this lever's button in the layout.

        Returns:
            LeverPath with the best route found.
        """
        # Try straight line first
        straight = self._try_straight(button_pos, pallet_pos, lever_index)
        if straight is not None:
            pivot = self._compute_pivot(straight, target_ratio)
            length = straight.length
            actual_ratio = self._actual_ratio(straight, pivot)
            feasible = length >= self.config.instrument.min_lever_length
            return LeverPath(
                button_pos=button_pos,
                pallet_pos=pallet_pos,
                pivot_pos=pivot,
                path=straight,
                segments=1,
                total_length=length,
                actual_ratio=actual_ratio,
                is_feasible=feasible,
            )

        # Try single dogleg
        dogleg = self._try_single_dogleg(button_pos, pallet_pos, lever_index)
        if dogleg is not None:
            pivot = self._compute_pivot(dogleg, target_ratio)
            length = dogleg.length
            actual_ratio = self._actual_ratio(dogleg, pivot)
            feasible = length >= self.config.instrument.min_lever_length
            return LeverPath(
                button_pos=button_pos,
                pallet_pos=pallet_pos,
                pivot_pos=pivot,
                path=dogleg,
                segments=len(dogleg.coords) - 1,
                total_length=length,
                actual_ratio=actual_ratio,
                is_feasible=feasible,
            )

        # Infeasible: return straight line anyway for visualization
        straight_line = LineString([button_pos, pallet_pos])
        pivot = self._compute_pivot(straight_line, target_ratio)
        return LeverPath(
            button_pos=button_pos,
            pallet_pos=pallet_pos,
            pivot_pos=pivot,
            path=straight_line,
            segments=1,
            total_length=straight_line.length,
            actual_ratio=self._actual_ratio(straight_line, pivot),
            is_feasible=False,
        )

    def _try_straight(
        self,
        button_pos: tuple[float, float],
        pallet_pos: tuple[float, float],
        lever_index: int,
    ) -> LineString | None:
        """Return straight line if it clears routing obstacles, else None."""
        line = LineString([button_pos, pallet_pos])
        buffered = line.buffer(self._lever_width / 2)
        merged = self.obstacles.get_merged_routing(exclude_reed_index=lever_index)

        if not buffered.intersects(merged):
            return line
        return None

    def _try_single_dogleg(
        self,
        button_pos: tuple[float, float],
        pallet_pos: tuple[float, float],
        lever_index: int,
    ) -> LineString | None:
        """Try routing around blocking obstacles with a single bend.

        For each obstacle that blocks the straight path, compute tangent
        waypoints and test 2-segment paths through them.
        """
        straight = LineString([button_pos, pallet_pos])
        buffered_straight = straight.buffer(self._lever_width / 2)
        merged = self.obstacles.get_merged_routing(exclude_reed_index=lever_index)

        # Find which individual obstacles block the path
        blocking = []
        for obs in self.obstacles.get_routing_obstacles(
            exclude_reed_index=lever_index,
        ):
            if buffered_straight.intersects(obs):
                blocking.append(obs)

        if not blocking:
            return straight  # shouldn't happen if _try_straight failed, but be safe

        best_path = None
        best_length = float("inf")

        for obs in blocking:
            # Generate waypoints around the obstacle
            waypoints = self._tangent_waypoints(button_pos, pallet_pos, obs)

            for wp in waypoints:
                candidate = LineString([button_pos, wp, pallet_pos])
                buffered = candidate.buffer(self._lever_width / 2)

                if not buffered.intersects(merged):
                    if candidate.length < best_length:
                        best_length = candidate.length
                        best_path = candidate

        return best_path

    def _tangent_waypoints(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        obstacle,
    ) -> list[tuple[float, float]]:
        """Generate candidate waypoints around an obstacle.

        Uses points on the obstacle boundary offset by the lever width,
        perpendicular to the start-end line.
        """
        centroid = obstacle.centroid
        cx, cy = centroid.x, centroid.y

        # Direction from start to end
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            return []

        # Perpendicular direction (both sides)
        nx = -dy / length
        ny = dx / length

        # Obstacle "radius" -- approximate as distance from centroid to boundary
        boundary_pt = obstacle.boundary.interpolate(0)
        obs_radius = centroid.distance(boundary_pt)

        # Offset distance: obstacle radius + lever width + small margin
        offset = obs_radius + self._lever_width + 1.0

        waypoints = [
            (cx + nx * offset, cy + ny * offset),
            (cx - nx * offset, cy - ny * offset),
        ]

        # Also try diagonal offsets for rounder obstacles
        for angle in [math.pi / 4, -math.pi / 4, 3 * math.pi / 4, -3 * math.pi / 4]:
            wx = cx + offset * math.cos(angle)
            wy = cy + offset * math.sin(angle)
            waypoints.append((wx, wy))

        return waypoints

    def _compute_pivot(
        self,
        path: LineString,
        target_ratio: float,
    ) -> tuple[float, float]:
        """Find the pivot point along the path at the correct ratio.

        For a ratio R, the pivot divides the lever into:
        - Button-to-pivot distance = L / (R + 1)
        - Pivot-to-pallet distance = L * R / (R + 1)

        The pivot is placed along the path from the button end.
        """
        total_length = path.length
        pivot_distance = total_length / (target_ratio + 1.0)
        pivot_point = path.interpolate(pivot_distance)
        return (pivot_point.x, pivot_point.y)

    def _actual_ratio(
        self,
        path: LineString,
        pivot_pos: tuple[float, float],
    ) -> float:
        """Compute the actual leverage ratio from the pivot position.

        ratio = pivot-to-pallet / button-to-pivot
        """
        pivot = Point(pivot_pos)
        # Distance along path from start (button) to pivot
        btn_to_pivot = path.project(pivot)
        pivot_to_pallet = path.length - btn_to_pivot

        if btn_to_pivot < 1e-6:
            return float("inf")
        return pivot_to_pallet / btn_to_pivot


def route_all_levers(
    layout,
    reed_plates: list,
    reed_specs: list,
    obstacle_field: ObstacleField,
    config: ConcertinaConfig | None = None,
) -> list[LeverPath]:
    """Route all levers for a complete layout.

    Args:
        layout: HaydenLayout with button positions.
        reed_plates: Positioned ReedPlate objects.
        reed_specs: ReedSpec objects (for target ratios).
        obstacle_field: Pre-built obstacle field.
        config: Configuration.

    Returns:
        List of LeverPath, one per enabled button.
    """
    config = config or ConcertinaConfig.defaults()
    router = LeverRouter(obstacle_field, config)

    buttons = layout.get_all_enabled()
    # Sort both by MIDI for consistent pairing
    buttons_sorted = sorted(buttons, key=lambda b: b.midi)

    paths = []
    for i, (btn, plate) in enumerate(zip(buttons_sorted, reed_plates)):
        path = router.route(
            button_pos=btn.pos,
            pallet_pos=plate.pallet_position,
            target_ratio=plate.spec.target_ratio,
            lever_index=i,
        )
        paths.append(path)

    return paths
