"""Tests for obstacle field and collision detection."""

import math

from shapely.geometry import LineString, Point

from concertina.config import ConcertinaConfig
from concertina.obstacles import ObstacleField
from concertina.reed_specs import ReedPlate, ReedSpec
from tests.mini_fixture import make_mini_layout, make_mini_reeds, make_mini_good_placement, make_mini_bad_placement


def _make_field(placement_fn=make_mini_good_placement):
    layout = make_mini_layout()
    reeds = make_mini_reeds(layout)
    plates = placement_fn(reeds)
    return ObstacleField(layout, plates), layout, reeds, plates


def test_button_obstacle_count():
    field, layout, _, _ = _make_field()
    assert len(field.get_button_obstacles()) == 6


def test_reed_obstacle_count():
    field, _, _, _ = _make_field()
    assert len(field.get_reed_obstacles()) == 6


def test_button_obstacle_radius():
    """Button obstacle should be button_radius + static_clearance."""
    field, layout, _, _ = _make_field()
    config = ConcertinaConfig.defaults()
    expected_radius = config.instrument.button_radius + config.clearance.static_floor
    btn = layout.get_all_enabled()[0]
    obs = field.get_button_obstacles()[0]
    # Check that a point at expected_radius is on the boundary
    edge_pt = Point(btn.x + expected_radius, btn.y)
    assert obs.boundary.distance(edge_pt) < 0.1


def test_good_placement_no_reed_collisions():
    field, _, _, _ = _make_field(make_mini_good_placement)
    collisions = field.check_reed_reed_collisions()
    assert len(collisions) == 0


def test_bad_placement_has_reed_collisions():
    field, _, _, _ = _make_field(make_mini_bad_placement)
    collisions = field.check_reed_reed_collisions()
    assert len(collisions) > 0
    # All 6 reeds stacked: should have C(6,2) = 15 pairs
    assert len(collisions) == 15


def test_exclude_button_index():
    field, _, _, _ = _make_field()
    all_obs = field.get_all_obstacles()
    excluded = field.get_all_obstacles(exclude_button_index=0)
    assert len(excluded) == len(all_obs) - 1


def test_exclude_reed_index():
    field, _, _, _ = _make_field()
    all_obs = field.get_all_obstacles()
    excluded = field.get_all_obstacles(exclude_reed_index=0)
    assert len(excluded) == len(all_obs) - 1


def test_exclude_both():
    field, _, _, _ = _make_field()
    all_obs = field.get_all_obstacles()
    excluded = field.get_all_obstacles(exclude_button_index=0, exclude_reed_index=0)
    assert len(excluded) == len(all_obs) - 2


def test_lever_collision_clear_path():
    """A lever far from any obstacle should have zero collision."""
    field, _, _, _ = _make_field()
    # Line far away from everything
    line = LineString([(-200, -200), (-200, -300)])
    area = field.check_lever_collision(line, 3.0, lever_index=0)
    assert area == 0.0


def test_lever_collision_through_button():
    """A lever through another button should report collision."""
    field, layout, _, _ = _make_field()
    btns = layout.get_all_enabled()
    # Draw lever from btn[0] through btn[1]
    line = LineString([
        (btns[0].x, btns[0].y),
        (btns[2].x, btns[2].y - 50),  # passes near btn[1]
    ])
    # This should hit btn[1]'s obstacle if the line crosses it
    # Use a line that definitely crosses btn[1]
    line = LineString([
        (btns[0].x, btns[0].y + 1),
        (btns[3].x, btns[3].y + 1),
    ])
    area = field.check_lever_collision(line, 3.0, lever_index=0)
    assert area > 0


def test_pivot_obstacles():
    field, _, _, _ = _make_field()
    assert len(field.get_pivot_obstacles()) == 0
    field.set_pivot_obstacles([(10.0, 10.0), (20.0, 20.0)])
    assert len(field.get_pivot_obstacles()) == 2


def test_lever_lever_distance():
    field, _, _, _ = _make_field()
    # Two parallel levers very close
    line_a = LineString([(0, 0), (50, 0)])
    line_b = LineString([(0, 1), (50, 1)])  # only 1mm apart
    violations = field.check_lever_lever_distance([line_a, line_b], min_distance=1.8)
    assert len(violations) == 1
    assert violations[0][2] < 1.8
