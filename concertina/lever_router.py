"""Lever routing from buttons to pallets.

v1: Try straight line first, then single-bend dogleg.
v2: Multi-bend dogleg via visibility graph + Dijkstra.

Hot-path checks use numpy geometry (geometry.py). Shapely is only used
for the LineString in LeverPath (needed by visualize.py for plotting).
"""

from __future__ import annotations

import heapq
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


@dataclass
class LeverPath:
    """Result of routing one lever."""

    button_pos: tuple[float, float]
    pallet_pos: tuple[float, float]
    pivot_pos: tuple[float, float]
    path: LineString               # centerline (shapely, for visualization only)
    segments: int                  # 1 = straight, 2+ = dogleg
    total_length: float            # mm
    actual_ratio: float            # pallet-side / button-side length
    is_feasible: bool              # no hard-constraint violations


def _seg_length(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((b[0] - a[0])**2 + (b[1] - a[1])**2)


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


def _polyline_length(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for i in range(len(points) - 1):
        total += _seg_length(points[i], points[i + 1])
    return total


class LeverRouter:
    """Routes levers from buttons to pallets, avoiding obstacles.

    Uses numpy geometry for all collision checks. Shapely is only used
    to construct the final LineString for visualization.

    Routing obstacles are reed plates and pivot posts only -- NOT buttons.
    Levers pass through slots under the action board.
    """

    def __init__(
        self,
        reed_plates: list[ReedPlate],
        config: ConcertinaConfig | None = None,
    ):
        self.config = config or ConcertinaConfig.defaults()
        self._lever_hw = self.config.instrument.lever_width_min / 2
        self._clearance = self.config.clearance.static_floor

        # Pre-compute buffered corners for all reed plates (routing obstacles)
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

        Args:
            button_pos: (x, y) of the button center.
            pallet_pos: (x, y) of the pallet hole.
            target_ratio: Desired leverage ratio.
            lever_index: Index of this lever's reed (excluded from obstacles).

        Returns:
            LeverPath with the best route found.
        """
        # Obstacles = all reed plates except this lever's own
        obstacles = [c for i, c in enumerate(self._reed_corners) if i != lever_index]

        # Try straight line first
        if self._is_clear(button_pos, pallet_pos, obstacles):
            points = [button_pos, pallet_pos]
            return self._build_path(points, target_ratio, feasible=True)

        # Try single dogleg
        dogleg = self._try_single_dogleg(button_pos, pallet_pos, obstacles)
        if dogleg is not None:
            return self._build_path(dogleg, target_ratio, feasible=True)

        # v2: Try multi-bend dogleg via visibility graph
        multi = self._try_visibility_graph(button_pos, pallet_pos, obstacles)
        if multi is not None:
            return self._build_path(multi, target_ratio, feasible=True)

        # Infeasible: return straight line for visualization
        points = [button_pos, pallet_pos]
        return self._build_path(points, target_ratio, feasible=False)

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

    def _try_single_dogleg(
        self,
        button_pos: tuple[float, float],
        pallet_pos: tuple[float, float],
        obstacles: list[np.ndarray],
    ) -> list[tuple[float, float]] | None:
        """Try routing around blocking obstacles with a single bend.

        For each blocking obstacle, generate waypoints around it and
        test 2-segment paths.
        """
        # Find which obstacles block the straight path
        blocking = []
        for corners in obstacles:
            dist = segment_to_rect_dist(button_pos, pallet_pos, corners)
            if dist < self._lever_hw:
                blocking.append(corners)

        if not blocking:
            return [button_pos, pallet_pos]

        best_path = None
        best_length = float("inf")

        for corners in blocking:
            waypoints = self._waypoints_around(button_pos, pallet_pos, corners)
            for wp in waypoints:
                # Check both segments clear ALL obstacles
                if (self._is_clear(button_pos, wp, obstacles) and
                        self._is_clear(wp, pallet_pos, obstacles)):
                    path = [button_pos, wp, pallet_pos]
                    length = _polyline_length(path)
                    if length < best_length:
                        best_length = length
                        best_path = path

        return best_path

    def _waypoints_around(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        corners: np.ndarray,
    ) -> list[tuple[float, float]]:
        """Generate candidate waypoints around a blocking rectangle."""
        # Centroid of the rectangle
        cx = float(corners[:, 0].mean())
        cy = float(corners[:, 1].mean())

        # Direction from start to end
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            return []

        # Half-diagonal of the rectangle as obstacle radius
        half_diag = max(
            math.sqrt((corners[i, 0] - cx)**2 + (corners[i, 1] - cy)**2)
            for i in range(4)
        )
        offset = half_diag + self._lever_hw * 2 + 1.0

        # Perpendicular offsets
        nx = -dy / length
        ny = dx / length

        waypoints = [
            (cx + nx * offset, cy + ny * offset),
            (cx - nx * offset, cy - ny * offset),
        ]

        # Corner-based offsets (8 directions)
        for angle_deg in range(0, 360, 45):
            angle = math.radians(angle_deg)
            waypoints.append((
                cx + offset * math.cos(angle),
                cy + offset * math.sin(angle),
            ))

        return waypoints

    def _try_visibility_graph(
        self,
        button_pos: tuple[float, float],
        pallet_pos: tuple[float, float],
        obstacles: list[np.ndarray],
    ) -> list[tuple[float, float]] | None:
        """v2: Route around multiple obstacles using visibility graph + Dijkstra.

        Generates waypoints at offset corners of all obstacle rectangles,
        builds a graph of mutually visible waypoints, and finds the shortest
        clear path from button to pallet.
        """
        # Generate waypoints: offset corners of every obstacle
        waypoints: list[tuple[float, float]] = []
        offset = self._lever_hw + 1.5  # clearance from obstacle edges

        for corners in obstacles:
            cx = float(corners[:, 0].mean())
            cy = float(corners[:, 1].mean())

            for ci in range(4):
                # Push each corner outward from the rectangle center
                px, py = float(corners[ci, 0]), float(corners[ci, 1])
                dx = px - cx
                dy = py - cy
                d = math.sqrt(dx * dx + dy * dy)
                if d < 1e-6:
                    continue
                wp = (px + dx / d * offset, py + dy / d * offset)
                waypoints.append(wp)

        if not waypoints:
            return None

        # All nodes: start + waypoints + end
        nodes = [button_pos] + waypoints + [pallet_pos]
        n = len(nodes)

        # Build adjacency: edge exists if the segment between two nodes is clear
        # Use Dijkstra with edge weight = segment length
        adj: dict[int, list[tuple[int, float]]] = {i: [] for i in range(n)}

        for i in range(n):
            for j in range(i + 1, n):
                if self._is_clear(nodes[i], nodes[j], obstacles):
                    dist = _seg_length(nodes[i], nodes[j])
                    adj[i].append((j, dist))
                    adj[j].append((i, dist))

        # Dijkstra from node 0 (button) to node n-1 (pallet)
        start_idx = 0
        end_idx = n - 1

        dist_to = [float("inf")] * n
        dist_to[start_idx] = 0.0
        prev = [-1] * n
        heap = [(0.0, start_idx)]

        while heap:
            d, u = heapq.heappop(heap)
            if d > dist_to[u]:
                continue
            if u == end_idx:
                break
            for v, w in adj[u]:
                nd = d + w
                if nd < dist_to[v]:
                    dist_to[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))

        if dist_to[end_idx] == float("inf"):
            return None  # no path found

        # Reconstruct path
        path = []
        cur = end_idx
        while cur != -1:
            path.append(nodes[cur])
            cur = prev[cur]
        path.reverse()

        return path

    def _build_path(
        self,
        points: list[tuple[float, float]],
        target_ratio: float,
        feasible: bool,
    ) -> LeverPath:
        """Construct a LeverPath from a list of waypoints."""
        total_length = _polyline_length(points)

        # Check min lever length
        if total_length < self.config.instrument.min_lever_length:
            feasible = False

        # Compute pivot position
        pivot_distance = total_length / (target_ratio + 1.0)
        pivot_pos = _interpolate_along_segments(points, pivot_distance)

        # Compute actual ratio
        btn_to_pivot = pivot_distance
        pivot_to_pallet = total_length - pivot_distance
        if btn_to_pivot < 1e-6:
            actual_ratio = float("inf")
        else:
            actual_ratio = pivot_to_pallet / btn_to_pivot

        # Build shapely LineString for visualization only
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

    Uses numpy geometry for all routing checks. The obstacle_field
    parameter is accepted for backwards compatibility but ignored --
    the router builds its own obstacle set from reed_plates directly.

    Args:
        layout: HaydenLayout with button positions.
        reed_plates: Positioned ReedPlate objects.
        reed_specs: ReedSpec objects (for target ratios).
        obstacle_field: Ignored (backwards compat). Pass None.
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
