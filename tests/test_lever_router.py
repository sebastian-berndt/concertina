"""Tests for the lever router."""

import math

from shapely.geometry import LineString, Point

from concertina.config import ConcertinaConfig
from concertina.obstacles import ObstacleField
from concertina.lever_router import LeverRouter, LeverPath, route_all_levers
from concertina.reed_specs import ReedPlate, ReedSpec
from tests.mini_fixture import make_mini_layout, make_mini_reeds, make_mini_good_placement


def _make_router():
    layout = make_mini_layout()
    reeds = make_mini_reeds(layout)
    plates = make_mini_good_placement(reeds)
    field = ObstacleField(layout, plates)
    router = LeverRouter(field)
    return router, layout, reeds, plates, field


def test_straight_lever_no_obstacles():
    """A lever far from any obstacle should be straight."""
    router, _, _, _, _ = _make_router()
    path = router.route(
        button_pos=(-200, 0),
        pallet_pos=(-200, -80),
        target_ratio=2.0,
        lever_index=0,
    )
    assert path.segments == 1
    assert path.is_feasible
    assert abs(path.total_length - 80.0) < 0.1


def test_pivot_at_correct_ratio():
    """Pivot should divide lever at 1/(R+1) from button end."""
    router, _, _, _, _ = _make_router()
    path = router.route(
        button_pos=(0, 0),
        pallet_pos=(0, -90),
        target_ratio=2.0,
        lever_index=0,
    )
    # For ratio 2.0, pivot at 1/3 of 90 = 30mm from button
    expected_pivot_y = -30.0
    assert abs(path.pivot_pos[1] - expected_pivot_y) < 0.5
    assert abs(path.actual_ratio - 2.0) < 0.05


def test_pivot_ratio_1_to_1():
    """For ratio 1.0, pivot should be at midpoint."""
    router, _, _, _, _ = _make_router()
    path = router.route(
        button_pos=(0, 0),
        pallet_pos=(0, -80),
        target_ratio=1.0,
        lever_index=0,
    )
    assert abs(path.pivot_pos[1] - (-40.0)) < 0.5
    assert abs(path.actual_ratio - 1.0) < 0.05


def test_route_all_returns_correct_count():
    layout = make_mini_layout()
    reeds = make_mini_reeds(layout)
    plates = make_mini_good_placement(reeds)
    field = ObstacleField(layout, plates)
    paths = route_all_levers(layout, plates, reeds, field)
    assert len(paths) == 6


def test_all_levers_have_positive_length():
    layout = make_mini_layout()
    reeds = make_mini_reeds(layout)
    plates = make_mini_good_placement(reeds)
    field = ObstacleField(layout, plates)
    paths = route_all_levers(layout, plates, reeds, field)
    for p in paths:
        assert p.total_length > 0


def test_lever_path_dataclass():
    """LeverPath should have all expected fields."""
    path = LeverPath(
        button_pos=(0, 0),
        pallet_pos=(0, -60),
        pivot_pos=(0, -20),
        path=LineString([(0, 0), (0, -60)]),
        segments=1,
        total_length=60.0,
        actual_ratio=2.0,
        is_feasible=True,
    )
    assert path.segments == 1
    assert path.is_feasible
    assert path.actual_ratio == 2.0


def test_min_lever_length_constraint():
    """Levers shorter than min_lever_length should be marked infeasible."""
    config = ConcertinaConfig.defaults()
    config.instrument.min_lever_length = 100.0  # set very high

    layout = make_mini_layout()
    reeds = make_mini_reeds(layout)
    plates = make_mini_good_placement(reeds)
    field = ObstacleField(layout, plates, config)
    router = LeverRouter(field, config)

    # Route a short lever
    path = router.route(
        button_pos=(0, 0),
        pallet_pos=(0, -50),  # only 50mm, below 100mm threshold
        target_ratio=2.0,
        lever_index=0,
    )
    assert not path.is_feasible
