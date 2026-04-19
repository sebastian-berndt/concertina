"""Tests for the numpy geometry module."""

import math

import numpy as np

from concertina.geometry import (
    rect_corners,
    rect_corners_buffered,
    rects_overlap,
    segment_to_rect_dist,
    segment_to_circle_dist,
    pallet_position,
)


def test_rect_corners_axis_aligned():
    corners = rect_corners(0, 0, 10, 4, 0)
    assert corners.shape == (4, 2)
    # Should be a 10x4 rect centered at origin
    assert abs(corners[:, 0].min() - (-5)) < 0.01
    assert abs(corners[:, 0].max() - 5) < 0.01
    assert abs(corners[:, 1].min() - (-2)) < 0.01
    assert abs(corners[:, 1].max() - 2) < 0.01


def test_rect_corners_rotated_90():
    corners = rect_corners(0, 0, 10, 4, math.pi / 2)
    # 90° rotation: length along Y, width along X
    assert abs(corners[:, 0].min() - (-2)) < 0.01
    assert abs(corners[:, 0].max() - 2) < 0.01
    assert abs(corners[:, 1].min() - (-5)) < 0.01
    assert abs(corners[:, 1].max() - 5) < 0.01


def test_rects_overlap_touching():
    a = rect_corners(0, 0, 10, 4, 0)
    b = rect_corners(10, 0, 10, 4, 0)  # touching at x=5
    assert rects_overlap(a, b)  # touching counts as overlap


def test_rects_overlap_separated():
    a = rect_corners(0, 0, 10, 4, 0)
    b = rect_corners(20, 0, 10, 4, 0)  # 5mm gap
    assert not rects_overlap(a, b)


def test_rects_overlap_with_clearance():
    a = rect_corners(0, 0, 10, 4, 0)
    b = rect_corners(12, 0, 10, 4, 0)  # 2mm gap
    assert not rects_overlap(a, b, clearance=0)
    assert rects_overlap(a, b, clearance=3)  # 3mm clearance closes the 2mm gap


def test_rects_overlap_rotated():
    a = rect_corners(0, 0, 10, 4, 0)
    b = rect_corners(0, 0, 10, 4, math.pi / 4)  # same center, rotated
    assert rects_overlap(a, b)


def test_segment_to_rect_dist_crossing():
    corners = rect_corners(5, 0, 6, 4, 0)
    dist = segment_to_rect_dist((0, 0), (10, 0), corners)
    assert dist == 0.0  # line passes through rect


def test_segment_to_rect_dist_parallel():
    corners = rect_corners(0, 0, 10, 4, 0)
    # Line 5mm above the rect (rect extends to y=2)
    dist = segment_to_rect_dist((-20, 5), (20, 5), corners)
    assert abs(dist - 3.0) < 0.1


def test_segment_to_rect_dist_far():
    corners = rect_corners(100, 100, 10, 4, 0)
    dist = segment_to_rect_dist((0, 0), (1, 0), corners)
    assert dist > 90


def test_segment_to_circle_dist_intersecting():
    dist = segment_to_circle_dist((0, 0), (10, 0), (5, 0), 3)
    assert dist == 0.0


def test_segment_to_circle_dist_outside():
    dist = segment_to_circle_dist((0, 0), (10, 0), (5, 10), 3)
    assert abs(dist - 7.0) < 0.1


def test_pallet_position_along_axis():
    px, py = pallet_position(0, 0, 40, 0, 0.2)
    # offset = 40/2 - 40*0.2 = 12, along x-axis
    assert abs(px - 12.0) < 0.01
    assert abs(py) < 0.01


def test_buffered_corners_larger():
    corners = rect_corners(0, 0, 10, 4, 0)
    buffered = rect_corners_buffered(0, 0, 10, 4, 0, 2.0)
    # Buffered should be 14x8
    assert abs(buffered[:, 0].max() - 7) < 0.01
    assert abs(buffered[:, 1].max() - 4) < 0.01
