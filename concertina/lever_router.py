"""Lever routing from buttons to pallets.

Physical constraints for laser-cut levers:
- Maximum 2 bends (3 segments). A flat piece of metal can't zigzag.
- Maximum bend angle of 30° per bend. Sharp turns weaken the lever.
- Prefer straight > 1-bend > 2-bend.

Hot-path checks use numpy geometry (geometry.py). Shapely is only used
for the LineString in LeverPath (needed by visualize.py for plotting).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.geometry import LineString

from concertina.config import ConcertinaConfig
from concertina.geometry import (
    rect_corners_buffered,
    segment_to_rect_dist,
)
from concertina.reed_specs import ReedPlate

# Maximum bend angle in radians (30°)
MAX_BEND_ANGLE = math.radians(30)


@dataclass
class LeverPath:
    """Result of routing one lever."""

    button_pos: tuple[float, float]
    pallet_pos: tuple[float, float]
    pivot_pos: tuple[float, float]
    path: LineString               # centerline (shapely, for visualization only)
    segments: int                  # 1 = straight, 2 = single bend, 3 = double bend
    total_length: float            # mm
    actual_ratio: float            # pallet-side / button-side length
    is_feasible: bool              # no hard-constraint violations


def _seg_length(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)


def _polyline_length(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for i in range(len(points) - 1):
        total += _seg_length(points[i], points[i + 1])
    return total


def _interpolate_along_segments(
    points: list[tuple[float, float]],
    distance: float,
) -> tuple[float, float]:
    """Walk along a polyline and return the point at the given distance."""
    remaining = distance
    for i in range(len(points) - 1):
        seg_len = _seg_length(points[i], points[i + 1])
        if remaining <= seg_len or i == len(points) - 2:
            t = remaining / seg_len if seg_len > 1e-10 else 0.0
            t = min(1.0, max(0.0, t))
            return (
                points[i][0] + t * (points[i + 1][0] - points[i][0]),
                points[i][1] + t * (points[i + 1][1] - points[i][1]),
            )
        remaining -= seg_len
    return points[-1]


def _angle_between_segments(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> float:
    """Angle of the bend at point b, between segments a→b and b→c.

    Returns the deviation from straight (0 = straight, pi = U-turn).
    """
    dx1 = b[0] - a[0]
    dy1 = b[1] - a[1]
    dx2 = c[0] - b[0]
    dy2 = c[1] - b[1]

    len1 = math.sqrt(dx1*dx1 + dy1*dy1)
    len2 = math.sqrt(dx2*dx2 + dy2*dy2)
    if len1 < 1e-10 or len2 < 1e-10:
        return 0.0

    # Dot product gives cos of angle between directions
    cos_angle = (dx1*dx2 + dy1*dy2) / (len1 * len2)
    cos_angle = max(-1.0, min(1.0, cos_angle))

    # Deviation from straight: 0 means straight, pi means U-turn
    return math.acos(cos_angle)


def _check_bend_angles(
    points: list[tuple[float, float]],
    max_angle: float,
) -> bool:
    """Check that all bends in a polyline are within the angle limit.

    Returns True if all bends are <= max_angle deviation from straight.
    """
    for i in range(1, len(points) - 1):
        deviation = math.pi - _angle_between_segments(
            points[i - 1], points[i], points[i + 1],
        )
        if deviation > max_angle:
            return False
    return True


class LeverRouter:
    """Routes levers from buttons to pallets, avoiding obstacles.

    Physical constraints:
    - Max 2 bends (3 segments)
    - Max 30° bend angle per bend
    - Levers pass under buttons (only reed plates are obstacles)
    """

    def __init__(
        self,
        reed_plates: list[ReedPlate],
        config: ConcertinaConfig | None = None,
        max_bends: int = 2,
        max_bend_angle: float = MAX_BEND_ANGLE,
    ):
        self.config = config or ConcertinaConfig.defaults()
        self._lever_hw = self.config.instrument.lever_width_min / 2
        self._clearance = self.config.clearance.static_floor
        self._max_bends = max_bends
        self._max_bend_angle = max_bend_angle

        # Pre-compute buffered corners for all reed plates
        self._reed_corners: list[np.ndarray] = []
        for plate in reed_plates:
            cx, cy = plate.center
            corners = rect_corners_buffered(
                cx, cy, plate.spec.length, plate.spec.width,
                plate.phi, self._clearance,
            )
            self._reed_corners.append(corners)

    def route(
        self,
        button_pos: tuple[float, float],
        pallet_pos: tuple[float, float],
        target_ratio: float,
        lever_index: int,
    ) -> LeverPath:
        """Find a lever path from button to pallet.

        Tries in order: straight → single bend → double bend.
        All bends must be ≤ max_bend_angle.
        """
        obstacles = [c for i, c in enumerate(self._reed_corners) if i != lever_index]

        # 1. Try straight
        if self._is_clear(button_pos, pallet_pos, obstacles):
            return self._build_path([button_pos, pallet_pos], target_ratio, feasible=True)

        # 2. Try single bend (2 segments)
        if self._max_bends >= 1:
            single = self._try_dogleg(button_pos, pallet_pos, obstacles, max_waypoints=1)
            if single is not None:
                return self._build_path(single, target_ratio, feasible=True)

        # 3. Try double bend (3 segments)
        if self._max_bends >= 2:
            double = self._try_dogleg(button_pos, pallet_pos, obstacles, max_waypoints=2)
            if double is not None:
                return self._build_path(double, target_ratio, feasible=True)

        # Infeasible
        return self._build_path([button_pos, pallet_pos], target_ratio, feasible=False)

    def _is_clear(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        obstacles: list[np.ndarray],
    ) -> bool:
        """Check if a lever segment clears all obstacles."""
        for corners in obstacles:
            dist = segment_to_rect_dist(start, end, corners)
            if dist < self._lever_hw:
                return False
        return True

    def _try_dogleg(
        self,
        button_pos: tuple[float, float],
        pallet_pos: tuple[float, float],
        obstacles: list[np.ndarray],
        max_waypoints: int,
    ) -> list[tuple[float, float]] | None:
        """Try routing with up to max_waypoints bend points.

        For single bend: try waypoints around each blocking obstacle.
        For double bend: try pairs of waypoints from different obstacles.
        All bends must satisfy the angle constraint.
        """
        # Generate candidate waypoints from blocking obstacles
        blocking = []
        for corners in obstacles:
            dist = segment_to_rect_dist(button_pos, pallet_pos, corners)
            if dist < self._lever_hw:
                blocking.append(corners)

        if not blocking:
            return [button_pos, pallet_pos]

        # Generate waypoints around all blocking obstacles
        all_waypoints = []
        for corners in blocking:
            all_waypoints.extend(self._waypoints_around(button_pos, pallet_pos, corners))

        best_path = None
        best_length = float("inf")

        if max_waypoints == 1:
            # Single bend: button → wp → pallet
            for wp in all_waypoints:
                path = [button_pos, wp, pallet_pos]
                if not _check_bend_angles(path, self._max_bend_angle):
                    continue
                if not self._all_segments_clear(path, obstacles):
                    continue
                length = _polyline_length(path)
                if length < best_length:
                    best_length = length
                    best_path = path

        elif max_waypoints == 2:
            # Double bend: button → wp1 → wp2 → pallet
            # Try pairs of waypoints (from same or different obstacles)
            for i, wp1 in enumerate(all_waypoints):
                # Quick check: is button→wp1 clear?
                if not self._is_clear(button_pos, wp1, obstacles):
                    continue
                for wp2 in all_waypoints[i + 1:]:
                    # Quick check: is wp2→pallet clear?
                    if not self._is_clear(wp2, pallet_pos, obstacles):
                        continue
                    path = [button_pos, wp1, wp2, pallet_pos]
                    if not _check_bend_angles(path, self._max_bend_angle):
                        continue
                    if not self._all_segments_clear(path, obstacles):
                        continue
                    length = _polyline_length(path)
                    if length < best_length:
                        best_length = length
                        best_path = path

                    # Also try reversed order
                    path_rev = [button_pos, wp2, wp1, pallet_pos]
                    if not _check_bend_angles(path_rev, self._max_bend_angle):
                        continue
                    if not self._all_segments_clear(path_rev, obstacles):
                        continue
                    length_rev = _polyline_length(path_rev)
                    if length_rev < best_length:
                        best_length = length_rev
                        best_path = path_rev

        return best_path

    def _all_segments_clear(
        self,
        points: list[tuple[float, float]],
        obstacles: list[np.ndarray],
    ) -> bool:
        """Check that every segment in a polyline clears all obstacles."""
        for i in range(len(points) - 1):
            if not self._is_clear(points[i], points[i + 1], obstacles):
                return False
        return True

    def _waypoints_around(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        corners: np.ndarray,
    ) -> list[tuple[float, float]]:
        """Generate candidate waypoints around a blocking rectangle.

        Uses offset corner points and perpendicular offsets.
        """
        cx = float(corners[:, 0].mean())
        cy = float(corners[:, 1].mean())

        half_diag = max(
            math.sqrt((corners[i, 0] - cx)**2 + (corners[i, 1] - cy)**2)
            for i in range(4)
        )
        offset = half_diag + self._lever_hw * 2 + 1.0

        # Direction from start to end
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx * dx + dy * dy)

        waypoints = []

        # Perpendicular offsets (both sides)
        if length > 1e-6:
            nx = -dy / length
            ny = dx / length
            waypoints.append((cx + nx * offset, cy + ny * offset))
            waypoints.append((cx - nx * offset, cy - ny * offset))

        # Corner-based offsets (8 directions)
        for angle_deg in range(0, 360, 45):
            angle = math.radians(angle_deg)
            waypoints.append((
                cx + offset * math.cos(angle),
                cy + offset * math.sin(angle),
            ))

        return waypoints

    def _build_path(
        self,
        points: list[tuple[float, float]],
        target_ratio: float,
        feasible: bool,
    ) -> LeverPath:
        """Construct a LeverPath from a list of waypoints."""
        total_length = _polyline_length(points)

        if total_length < self.config.instrument.min_lever_length:
            feasible = False

        pivot_distance = total_length / (target_ratio + 1.0)
        pivot_pos = _interpolate_along_segments(points, pivot_distance)

        btn_to_pivot = pivot_distance
        pivot_to_pallet = total_length - pivot_distance
        if btn_to_pivot < 1e-6:
            actual_ratio = float("inf")
        else:
            actual_ratio = pivot_to_pallet / btn_to_pivot

        path = LineString(points)

        return LeverPath(
            button_pos=points[0],
            pallet_pos=points[-1],
            pivot_pos=pivot_pos,
            path=path,
            segments=len(points) - 1,
            total_length=total_length,
            actual_ratio=actual_ratio,
            is_feasible=feasible,
        )


def route_all_levers(
    layout,
    reed_plates: list[ReedPlate],
    reed_specs: list,
    obstacle_field=None,
    config: ConcertinaConfig | None = None,
) -> list[LeverPath]:
    """Route all levers for a complete layout.

    Args:
        layout: HaydenLayout with button positions.
        reed_plates: Positioned ReedPlate objects.
        reed_specs: ReedSpec objects (for target ratios).
        obstacle_field: Ignored (backwards compat).
        config: Configuration.

    Returns:
        List of LeverPath, one per enabled button.
    """
    config = config or ConcertinaConfig.defaults()
    router = LeverRouter(reed_plates, config)

    buttons = sorted(layout.get_all_enabled(), key=lambda b: b.midi)

    paths = []
    for i, (btn, plate) in enumerate(zip(buttons, reed_plates)):
        path = router.route(
            button_pos=btn.pos,
            pallet_pos=plate.pallet_position,
            target_ratio=plate.spec.target_ratio,
            lever_index=i,
        )
        paths.append(path)

    return paths
