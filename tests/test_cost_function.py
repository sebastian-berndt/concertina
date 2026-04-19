"""Tests for the cost function."""

import numpy as np

from concertina.config import ConcertinaConfig
from concertina.cost_function import evaluate, evaluate_detailed, decode_state
from tests.mini_fixture import (
    make_mini_layout,
    make_mini_reeds,
    make_mini_good_placement,
    make_mini_bad_placement,
    make_mini_config,
)


def _eval_placement(placement_fn, config=None):
    config = config or make_mini_config()
    layout = make_mini_layout()
    reeds = make_mini_reeds(layout)
    plates = placement_fn(reeds)
    x = np.array([[p.r, p.theta, p.phi] for p in plates]).flatten()
    return evaluate_detailed(x, layout, reeds, config)


def test_good_placement_finite_cost():
    bd = _eval_placement(make_mini_good_placement)
    assert bd.total < float("inf")
    assert bd.total > 0


def test_good_placement_no_collisions():
    bd = _eval_placement(make_mini_good_placement)
    assert bd.reed_collision == 0.0
    assert bd.lever_collision == 0.0


def test_bad_placement_dominated_by_collisions():
    bd = _eval_placement(make_mini_bad_placement)
    assert bd.reed_collision > 0
    collision_fraction = bd.reed_collision / bd.total
    assert collision_fraction > 0.9  # collisions dominate


def test_good_cheaper_than_bad():
    good = _eval_placement(make_mini_good_placement)
    bad = _eval_placement(make_mini_bad_placement)
    assert good.total < bad.total


def test_cost_breakdown_sums_to_total():
    bd = _eval_placement(make_mini_good_placement)
    expected = (
        bd.reed_collision + bd.lever_collision + bd.lever_length
        + bd.hex_area + bd.ratio_deviation + bd.bend
        + bd.uniformity + bd.lever_proximity + bd.min_length
        + bd.pivot_accessibility + bd.center_of_gravity
        + bd.chamber_proportionality
    )
    assert abs(bd.total - expected) < 0.01


def test_evaluate_returns_scalar():
    config = make_mini_config()
    layout = make_mini_layout()
    reeds = make_mini_reeds(layout)
    plates = make_mini_good_placement(reeds)
    x = np.array([[p.r, p.theta, p.phi] for p in plates]).flatten()
    cost = evaluate(x, layout, reeds, config)
    assert isinstance(cost, float)
    assert cost > 0


def test_decode_state_3_params():
    reeds = make_mini_reeds(make_mini_layout())
    x = np.array([60.0, 0.5, 0.3, 70.0, 1.0, 0.5,
                   80.0, 1.5, 0.7, 65.0, 0.8, 0.1,
                   75.0, 1.2, 0.4, 85.0, 1.8, 0.9])
    plates = decode_state(x, reeds)
    assert len(plates) == 6
    assert plates[0].r == 60.0
    assert plates[0].theta == 0.5
    assert plates[0].phi == 0.3


def test_decode_state_2_params():
    """Stage 1: only r and theta, phi auto-computed."""
    reeds = make_mini_reeds(make_mini_layout())
    x = np.array([60.0, 0.5, 70.0, 1.0, 80.0, 1.5,
                   65.0, 0.8, 75.0, 1.2, 85.0, 1.8])
    plates = decode_state(x, reeds)
    assert len(plates) == 6
    assert plates[0].r == 60.0
    assert plates[0].theta == 0.5
    # phi should default to theta + pi
    import math
    assert abs(plates[0].phi - (0.5 + math.pi)) < 0.01


def test_tier2_weights_off_by_default():
    bd = _eval_placement(make_mini_good_placement)
    assert bd.bend == 0.0
    assert bd.uniformity == 0.0
    assert bd.lever_proximity == 0.0


def test_tier2_bend_penalty():
    config = make_mini_config()
    config.weights.w_bend = 50.0
    bd = _eval_placement(make_mini_good_placement, config)
    # Good placement may have some doglegs, so bend >= 0
    assert bd.bend >= 0.0


def test_ratio_grace_zone():
    """Small deviations within grace zone should not be penalized."""
    config = make_mini_config()
    config.ratio.grace_zone = 10.0  # huge grace zone
    bd = _eval_placement(make_mini_good_placement, config)
    assert bd.ratio_deviation == 0.0
