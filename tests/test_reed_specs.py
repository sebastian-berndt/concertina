"""Tests for reed plate dimensions and geometry."""

import math

from concertina.config import RatioSpec, ReedDimensions
from concertina.hayden_layout import HaydenLayout
from concertina.reed_specs import generate_reed_table, ReedPlate, ReedSpec


def _make_reeds():
    layout = HaydenLayout.from_beaumont("LH")
    return generate_reed_table(layout.get_all_enabled())


def test_generates_26_specs():
    reeds = _make_reeds()
    assert len(reeds) == 26


def test_sorted_by_midi():
    reeds = _make_reeds()
    midis = [r.midi for r in reeds]
    assert midis == sorted(midis)


def test_bass_is_largest():
    reeds = _make_reeds()
    assert reeds[0].length > reeds[-1].length
    assert reeds[0].width > reeds[-1].width


def test_dimensions_match_extremes():
    reeds = _make_reeds()
    assert reeds[0].length == 55.0  # bass
    assert reeds[0].width == 18.0
    assert reeds[-1].length == 24.0  # treble
    assert reeds[-1].width == 15.0


def test_graduated_ratios():
    reeds = _make_reeds()
    assert reeds[0].target_ratio == 2.2  # bass
    assert reeds[-1].target_ratio == 1.8  # treble
    # Mid should be between
    mid = reeds[len(reeds) // 2]
    assert 1.8 < mid.target_ratio < 2.2


def test_flat_ratio():
    layout = HaydenLayout.from_beaumont("LH")
    ratio = RatioSpec(graduated=False, target_ratio=2.0)
    reeds = generate_reed_table(layout.get_all_enabled(), ratio)
    for r in reeds:
        assert r.target_ratio == 2.0


def test_ratio_clamped_to_bounds():
    layout = HaydenLayout.from_beaumont("LH")
    ratio = RatioSpec(graduated=False, target_ratio=1.5, ratio_min=1.8)
    reeds = generate_reed_table(layout.get_all_enabled(), ratio)
    for r in reeds:
        assert r.target_ratio == 1.8  # clamped to minimum


def test_polygon_area():
    spec = ReedSpec(note="C4", midi=60, length=40.0, width=16.0, target_ratio=2.0)
    plate = ReedPlate(spec=spec, r=60.0, theta=0.0, phi=0.0)
    poly = plate.get_polygon()
    assert abs(poly.area - 40.0 * 16.0) < 0.1


def test_polygon_rotation():
    spec = ReedSpec(note="C4", midi=60, length=40.0, width=16.0, target_ratio=2.0)
    plate_0 = ReedPlate(spec=spec, r=60.0, theta=0.0, phi=0.0)
    plate_90 = ReedPlate(spec=spec, r=60.0, theta=0.0, phi=math.pi / 2)
    poly_0 = plate_0.get_polygon()
    poly_90 = plate_90.get_polygon()
    # Same area regardless of rotation
    assert abs(poly_0.area - poly_90.area) < 0.1
    # But different bounding boxes
    assert abs(poly_0.bounds[2] - poly_0.bounds[0] - 40.0) < 0.1  # width ~40 (length)
    assert abs(poly_90.bounds[2] - poly_90.bounds[0] - 16.0) < 0.1  # width ~16


def test_polygon_valid():
    spec = ReedSpec(note="C4", midi=60, length=40.0, width=16.0, target_ratio=2.0)
    plate = ReedPlate(spec=spec, r=60.0, theta=1.0, phi=0.5)
    poly = plate.get_polygon()
    assert poly.is_valid


def test_buffered_polygon_larger():
    spec = ReedSpec(note="C4", midi=60, length=40.0, width=16.0, target_ratio=2.0)
    plate = ReedPlate(spec=spec, r=60.0, theta=0.0, phi=0.0)
    poly = plate.get_polygon()
    buffered = plate.get_polygon(clearance=1.2)
    assert buffered.area > poly.area


def test_pallet_on_plate():
    spec = ReedSpec(note="C4", midi=60, length=40.0, width=16.0, target_ratio=2.0)
    plate = ReedPlate(spec=spec, r=60.0, theta=0.0, phi=0.0)
    px, py = plate.pallet_position
    poly = plate.get_polygon()
    # Pallet should be inside or on the edge of the plate
    from shapely.geometry import Point
    assert poly.buffer(0.1).contains(Point(px, py))


def test_center_polar_to_cartesian():
    spec = ReedSpec(note="C4", midi=60, length=40.0, width=16.0, target_ratio=2.0)
    plate = ReedPlate(spec=spec, r=50.0, theta=math.pi / 4, phi=0.0)
    cx, cy = plate.center
    expected_x = 50.0 * math.cos(math.pi / 4)
    expected_y = 50.0 * math.sin(math.pi / 4)
    assert abs(cx - expected_x) < 0.01
    assert abs(cy - expected_y) < 0.01
