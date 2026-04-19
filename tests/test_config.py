"""Tests for the configuration system."""

import json
import math
import tempfile
from pathlib import Path

from concertina.config import (
    ConcertinaConfig,
    InstrumentSpec,
    RatioSpec,
    ClearanceSpec,
    CostWeights,
    SolverBounds,
    ReedDimensions,
)


def test_defaults_create():
    config = ConcertinaConfig.defaults()
    assert config.instrument.button_diameter == 6.0
    assert config.instrument.h_pitch == 16.5
    assert abs(config.instrument.v_pitch - 16.5 * math.sqrt(3) / 2) < 0.01
    assert config.ratio.target_ratio == 2.0
    assert config.clearance.static_floor == 1.2


def test_button_radius():
    spec = InstrumentSpec()
    assert spec.button_radius == 3.0


def test_ratio_bounds():
    spec = RatioSpec()
    assert spec.ratio_min == 1.8
    assert spec.ratio_max == 2.3
    assert spec.bass_ratio > spec.treble_ratio


def test_tiered_weights():
    w = CostWeights()
    # Tier 1 should be active
    assert w.w_reed_collision > 0
    assert w.w_lever_collision > 0
    assert w.w_lever_length > 0
    assert w.w_ratio_deviation > 0
    # Tier 2/3 should be off by default
    assert w.w_bend == 0.0
    assert w.w_uniformity == 0.0
    assert w.w_pivot_accessibility == 0.0


def test_save_load_roundtrip():
    config = ConcertinaConfig.defaults()
    config.ratio.target_ratio = 1.5
    config.clearance.static_floor = 1.5

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    config.save(path)

    loaded = ConcertinaConfig.load(path)
    assert loaded.ratio.target_ratio == 1.5
    assert loaded.clearance.static_floor == 1.5
    assert loaded.instrument.button_diameter == 6.0
    Path(path).unlink()


def test_custom_instrument():
    spec = InstrumentSpec(
        button_diameter=6.5,
        h_pitch=18.5,
        keys_per_side=17,
    )
    assert spec.button_diameter == 6.5
    assert spec.button_radius == 3.25
    assert spec.keys_per_side == 17
