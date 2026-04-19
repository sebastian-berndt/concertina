"""No-go zone generation and collision detection.

Two distinct obstacle contexts:
- PLACEMENT: Where can a reed plate go? Must avoid buttons and other reeds.
- ROUTING: Where can a lever go? Must avoid other reeds and pivots.
  Levers pass through slots UNDER the action board, so button holes
  are NOT lever obstacles.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from shapely.geometry import Point, LineString, MultiPolygon
from shapely.ops import unary_union

from concertina.config import ConcertinaConfig, ClearanceSpec, InstrumentSpec
from concertina.hayden_layout import HaydenLayout, HaydenButton
from concertina.reed_specs import ReedPlate


class ObstacleField:
    """Manages no-go zones for both reed placement and lever routing.

    Obstacle types:
    - Button holes: circles around each button — placement obstacles ONLY
    - Reed plates: rectangles occupied by reed plates — both placement and routing
    - Pivot posts: circles around pivot points — routing obstacles ONLY
    """

    def __init__(
        self,
        layout: HaydenLayout,
        reed_plates: list[ReedPlate],
        config: ConcertinaConfig | None = None,
    ):
        self.layout = layout
        self.reed_plates = reed_plates
        self.config = config or ConcertinaConfig.defaults()

        self._button_obstacles: list = []
        self._reed_obstacles: list = []
        self._pivot_obstacles: list = []
        self._merged_routing: object | None = None

        self._build()

    def _build(self) -> None:
        """Construct button and reed obstacles."""
        inst = self.config.instrument
        clearance = self.config.clearance

        # Button holes: placement obstacles only
        button_radius = inst.button_radius + clearance.static_floor
        for btn in self.layout.get_all_enabled():
            circle = Point(btn.x, btn.y).buffer(button_radius)
            self._button_obstacles.append(circle)

        # Reed plates: both placement and routing obstacles
        for plate in self.reed_plates:
            poly = plate.get_polygon(clearance=clearance.static_floor)
            self._reed_obstacles.append(poly)

        self._merged_routing = None

    def set_pivot_obstacles(self, pivot_points: list[tuple[float, float]]) -> None:
        """Set pivot post obstacles. Call after pivots are computed.

        Args:
            pivot_points: List of (x, y) pivot positions.
        """
        self._pivot_obstacles = []
        for px, py in pivot_points:
            circle = Point(px, py).buffer(self.config.clearance.pivot_buffer)
            self._pivot_obstacles.append(circle)
        self._merged = None  # invalidate cache

    def get_button_obstacles(self) -> list:
        """Return all button hole no-go zones."""
        return list(self._button_obstacles)

    def get_reed_obstacles(self) -> list:
        """Return all reed plate no-go zones."""
        return list(self._reed_obstacles)

    def get_pivot_obstacles(self) -> list:
        """Return all pivot post no-go zones."""
        return list(self._pivot_obstacles)

    # --- Placement obstacles (buttons + reeds) ---

    def get_placement_obstacles(
        self,
        exclude_reed_index: int | None = None,
    ) -> list:
        """Obstacles for reed plate placement: buttons + other reeds."""
        obs = list(self._button_obstacles)
        for i, o in enumerate(self._reed_obstacles):
            if i != exclude_reed_index:
                obs.append(o)
        return obs

    # --- Routing obstacles (reeds + pivots, NOT buttons) ---

    def get_routing_obstacles(
        self,
        exclude_reed_index: int | None = None,
    ) -> list:
        """Obstacles for lever routing: other reeds + pivots. NOT buttons.

        Levers pass through slots under the action board, so button
        holes are not routing obstacles.
        """
        obs = []
        for i, o in enumerate(self._reed_obstacles):
            if i != exclude_reed_index:
                obs.append(o)
        obs.extend(self._pivot_obstacles)
        return obs

    def get_merged_routing(
        self,
        exclude_reed_index: int | None = None,
    ):
        """Union of routing obstacles for fast intersection testing."""
        if exclude_reed_index is not None:
            obs = self.get_routing_obstacles(exclude_reed_index)
            return unary_union(obs) if obs else Point(0, 0).buffer(0)

        if self._merged_routing is None:
            obs = self.get_routing_obstacles()
            self._merged_routing = unary_union(obs) if obs else Point(0, 0).buffer(0)
        return self._merged_routing

    # --- Legacy compatibility ---

    def get_all_obstacles(
        self,
        exclude_button_index: int | None = None,
        exclude_reed_index: int | None = None,
    ) -> list:
        """Return all obstacles (placement context). For backwards compat."""
        obs = []
        for i, o in enumerate(self._button_obstacles):
            if i != exclude_button_index:
                obs.append(o)
        for i, o in enumerate(self._reed_obstacles):
            if i != exclude_reed_index:
                obs.append(o)
        obs.extend(self._pivot_obstacles)
        return obs

    def get_merged_obstacle(
        self,
        exclude_button_index: int | None = None,
        exclude_reed_index: int | None = None,
    ):
        """Union of all obstacles. For backwards compat."""
        obs = self.get_all_obstacles(exclude_button_index, exclude_reed_index)
        return unary_union(obs) if obs else Point(0, 0).buffer(0)

    def check_reed_reed_collisions(self) -> list[tuple[int, int, float]]:
        """Check all reed plate pairs for overlap.

        Returns:
            List of (i, j, overlap_area) for intersecting pairs.
        """
        collisions = []
        polys = [p.get_polygon() for p in self.reed_plates]
        for i in range(len(polys)):
            for j in range(i + 1, len(polys)):
                if polys[i].intersects(polys[j]):
                    overlap = polys[i].intersection(polys[j]).area
                    if overlap > 0.01:  # ignore numerical noise
                        collisions.append((i, j, overlap))
        return collisions

    def check_lever_collision(
        self,
        lever_line: LineString,
        lever_width: float,
        lever_index: int,
    ) -> float:
        """Check if a lever path collides with routing obstacles.

        Uses routing obstacles only (reeds + pivots, NOT buttons).
        Excludes this lever's own reed plate.

        Args:
            lever_line: Centerline of the lever.
            lever_width: Physical width of the lever.
            lever_index: Index of this lever's reed to exclude.

        Returns:
            Total intersection area. 0 means no collision.
        """
        lever_shape = lever_line.buffer(lever_width / 2)
        merged = self.get_merged_routing(exclude_reed_index=lever_index)

        if lever_shape.intersects(merged):
            return lever_shape.intersection(merged).area
        return 0.0

    def check_lever_lever_distance(
        self,
        lever_lines: list[LineString],
        min_distance: float | None = None,
    ) -> list[tuple[int, int, float]]:
        """Check all lever pairs for proximity violations.

        Args:
            lever_lines: Centerlines of all levers.
            min_distance: Minimum allowed distance. Uses config default if None.

        Returns:
            List of (i, j, distance) for pairs closer than min_distance.
        """
        if min_distance is None:
            min_distance = self.config.clearance.min_lever_lever_distance

        violations = []
        for i in range(len(lever_lines)):
            for j in range(i + 1, len(lever_lines)):
                dist = lever_lines[i].distance(lever_lines[j])
                if dist < min_distance:
                    violations.append((i, j, dist))
        return violations
