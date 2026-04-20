"""Lever routing from buttons to pallets.

Physical reality:
- Levers are slots cut through the action board
- They collide with: button holes, other levers, pivot posts, pallet holes
- They do NOT collide with reed plates (those are on the reed pan side)

Constraints for laser-cut levers:
- Maximum 2 bends (3 segments)
- Maximum 30° deviation from straight per bend
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
    segment_to_circle_dist,
)
from concertina.hayden_layout import HaydenLayout, HaydenButton
from concertina.reed_specs import ReedPlate

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


def _check_bend_angles(
    points: list[tuple[float, float]],
    max_angle: float,
) -> bool:
    for i in range(1, len(points) - 1):
        dx1 = points[i][0] - points[i-1][0]
        dy1 = points[i][1] - points[i-1][1]
        dx2 = points[i+1][0] - points[i][0]
        dy2 = points[i+1][1] - points[i][1]
        len1 = math.sqrt(dx1*dx1 + dy1*dy1)
        len2 = math.sqrt(dx2*dx2 + dy2*dy2)
        if len1 < 1e-10 or len2 < 1e-10:
            continue
        cos_a = (dx1*dx2 + dy1*dy2) / (len1 * len2)
        cos_a = max(-1.0, min(1.0, cos_a))
        deviation = math.pi - math.acos(cos_a)
        if deviation > max_angle:
            return False
    return True


class LeverRouter:
    """Routes levers from buttons to pallets.

    Obstacles are button holes and other levers — NOT reed plates.
    Reed plates are on the reed pan side of the board.
    """

    def __init__(
        self,
        buttons: list[HaydenButton],
        reed_plates: list[ReedPlate],
        config: ConcertinaConfig | None = None,
        max_bends: int = 2,
        max_bend_angle: float = MAX_BEND_ANGLE,
    ):
        self.config = config or ConcertinaConfig.defaults()
        self._lever_hw = self.config.instrument.lever_width_min / 2
        self._max_bends = max_bends
        self._max_bend_angle = max_bend_angle

        # Button holes are circular obstacles
        btn_radius = self.config.instrument.button_radius + self.config.clearance.static_floor
        self._button_centers = [(b.x, b.y) for b in buttons]
        self._button_radius = btn_radius

        # Pallet holes are circular obstacles (from other levers' reed plates)
        self._pallet_centers = [p.pallet_position for p in reed_plates]
        pallet_radius = 3.0 + self.config.clearance.static_floor  # approximate pallet hole radius
        self._pallet_radius = pallet_radius

        # Placed lever paths (built up incrementally during routing)
        self._placed_lever_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []

    def route(
        self,
        button_pos: tuple[float, float],
        pallet_pos: tuple[float, float],
        target_ratio: float,
        lever_index: int,
    ) -> LeverPath:
        """Find a lever path from button to pallet.

        Obstacles: all button holes (except own), all pallet holes (except own),
        all previously placed levers.
        """
        # 1. Try straight
        if self._is_clear(button_pos, pallet_pos, lever_index):
            path = [button_pos, pallet_pos]
            result = self._build_path(path, target_ratio, feasible=True)
            self._register_lever(path)
            return result

        # 2. Try single bend
        if self._max_bends >= 1:
            single = self._try_dogleg(button_pos, pallet_pos, lever_index, max_wp=1)
            if single is not None:
                result = self._build_path(single, target_ratio, feasible=True)
                self._register_lever(single)
                return result

        # 3. Try double bend
        if self._max_bends >= 2:
            double = self._try_dogleg(button_pos, pallet_pos, lever_index, max_wp=2)
            if double is not None:
                result = self._build_path(double, target_ratio, feasible=True)
                self._register_lever(double)
                return result

        # Infeasible
        path = [button_pos, pallet_pos]
        return self._build_path(path, target_ratio, feasible=False)

    def _is_clear(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        lever_index: int,
    ) -> bool:
        """Check if a segment clears all obstacles (buttons, pallets, placed levers)."""
        # Check button holes (skip own button)
        for i, center in enumerate(self._button_centers):
            if i == lever_index:
                continue
            dist = segment_to_circle_dist(start, end, center, self._button_radius)
            if dist < self._lever_hw:
                return False

        # Check pallet holes (skip own pallet)
        for i, center in enumerate(self._pallet_centers):
            if i == lever_index:
                continue
            dist = segment_to_circle_dist(start, end, center, self._pallet_radius)
            if dist < self._lever_hw:
                return False

        # Check placed levers
        for seg_start, seg_end in self._placed_lever_segments:
            # Segment-to-segment distance check
            d = _segment_segment_dist(start, end, seg_start, seg_end)
            if d < self._lever_hw * 2 + self.config.clearance.dynamic_gap:
                return False

        return True

    def _try_dogleg(
        self,
        button_pos: tuple[float, float],
        pallet_pos: tuple[float, float],
        lever_index: int,
        max_wp: int,
    ) -> list[tuple[float, float]] | None:
        """Try routing with gentle bends around blocking obstacles."""
        # Collect blocking obstacle centers
        blocking = []

        for i, center in enumerate(self._button_centers):
            if i == lever_index:
                continue
            dist = segment_to_circle_dist(button_pos, pallet_pos, center, self._button_radius)
            if dist < self._lever_hw:
                blocking.append((center, self._button_radius))

        for i, center in enumerate(self._pallet_centers):
            if i == lever_index:
                continue
            dist = segment_to_circle_dist(button_pos, pallet_pos, center, self._pallet_radius)
            if dist < self._lever_hw:
                blocking.append((center, self._pallet_radius))

        if not blocking:
            return [button_pos, pallet_pos]

        # Generate waypoints around blocking obstacles
        dx = pallet_pos[0] - button_pos[0]
        dy = pallet_pos[1] - button_pos[1]
        seg_len = math.sqrt(dx*dx + dy*dy)
        if seg_len < 1e-6:
            return None
        nx = -dy / seg_len
        ny = dx / seg_len

        all_waypoints = []
        for center, radius in blocking:
            offset = radius + self._lever_hw * 2 + 1.5
            cx, cy = center
            # Perpendicular offsets
            all_waypoints.append((cx + nx * offset, cy + ny * offset))
            all_waypoints.append((cx - nx * offset, cy - ny * offset))
            # Radial offsets
            for angle_deg in range(0, 360, 30):
                angle = math.radians(angle_deg)
                all_waypoints.append((cx + offset * math.cos(angle), cy + offset * math.sin(angle)))

        best_path = None
        best_length = float("inf")

        if max_wp == 1:
            for wp in all_waypoints:
                path = [button_pos, wp, pallet_pos]
                if not _check_bend_angles(path, self._max_bend_angle):
                    continue
                if not all(self._is_clear(path[j], path[j+1], lever_index) for j in range(len(path)-1)):
                    continue
                length = _polyline_length(path)
                if length < best_length:
                    best_length = length
                    best_path = path

        elif max_wp == 2:
            for i, wp1 in enumerate(all_waypoints):
                if not self._is_clear(button_pos, wp1, lever_index):
                    continue
                for wp2 in all_waypoints[i+1:]:
                    for path in ([button_pos, wp1, wp2, pallet_pos],
                                 [button_pos, wp2, wp1, pallet_pos]):
                        if not _check_bend_angles(path, self._max_bend_angle):
                            continue
                        if not all(self._is_clear(path[j], path[j+1], lever_index) for j in range(len(path)-1)):
                            continue
                        length = _polyline_length(path)
                        if length < best_length:
                            best_length = length
                            best_path = path

        return best_path

    def _register_lever(self, points: list[tuple[float, float]]) -> None:
        """Add a placed lever's segments to the obstacle set."""
        for i in range(len(points) - 1):
            self._placed_lever_segments.append((points[i], points[i + 1]))

    def _build_path(
        self,
        points: list[tuple[float, float]],
        target_ratio: float,
        feasible: bool,
    ) -> LeverPath:
        total_length = _polyline_length(points)
        if total_length < self.config.instrument.min_lever_length:
            feasible = False

        pivot_distance = total_length / (target_ratio + 1.0)
        pivot_pos = _interpolate_along_segments(points, pivot_distance)

        btn_to_pivot = pivot_distance
        pivot_to_pallet = total_length - pivot_distance
        actual_ratio = pivot_to_pallet / btn_to_pivot if btn_to_pivot > 1e-6 else float("inf")

        return LeverPath(
            button_pos=points[0],
            pallet_pos=points[-1],
            pivot_pos=pivot_pos,
            path=LineString(points),
            segments=len(points) - 1,
            total_length=total_length,
            actual_ratio=actual_ratio,
            is_feasible=feasible,
        )


def _segment_segment_dist(
    a1: tuple[float, float], a2: tuple[float, float],
    b1: tuple[float, float], b2: tuple[float, float],
) -> float:
    """Minimum distance between two line segments."""
    from concertina.geometry import _segments_min_dist_sq
    return math.sqrt(_segments_min_dist_sq(
        a1[0], a1[1], a2[0], a2[1],
        b1[0], b1[1], b2[0], b2[1],
    ))


def route_all_levers(
    layout: HaydenLayout,
    reed_plates: list[ReedPlate],
    reed_specs: list = None,
    obstacle_field=None,
    config: ConcertinaConfig | None = None,
) -> list[LeverPath]:
    """Route all levers for a complete layout.

    The router builds obstacles from button holes and placed levers.
    Reed plates are NOT obstacles (they're on the reed pan side).
    Levers are routed in order and each placed lever becomes an
    obstacle for subsequent levers.
    """
    config = config or ConcertinaConfig.defaults()
    buttons = sorted(layout.get_all_enabled(), key=lambda b: b.midi)
    router = LeverRouter(buttons, reed_plates, config)

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
