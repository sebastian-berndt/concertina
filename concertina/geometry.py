"""Fast numpy-based 2D geometry primitives.

Replaces shapely in all hot paths (greedy placer, lever router, fast cost
function). Shapely is only used for final validation and visualization.

Key operations:
- Rotated rectangle representation and overlap (Separating Axis Theorem)
- Line segment to rotated rectangle distance
- Line segment to circle distance
"""

from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Rotated rectangle representation
# ---------------------------------------------------------------------------

def rect_corners(
    cx: float, cy: float,
    length: float, width: float,
    phi: float,
) -> np.ndarray:
    """Compute the 4 corners of a rotated rectangle.

    Args:
        cx, cy: Center position.
        length, width: Rectangle dimensions.
        phi: Rotation angle in radians.

    Returns:
        (4, 2) array of corner coordinates.
    """
    cos_p = math.cos(phi)
    sin_p = math.sin(phi)
    hl = length / 2
    hw = width / 2

    # Local corners, then rotate + translate
    return np.array([
        [cx + cos_p * (-hl) - sin_p * (-hw), cy + sin_p * (-hl) + cos_p * (-hw)],
        [cx + cos_p * ( hl) - sin_p * (-hw), cy + sin_p * ( hl) + cos_p * (-hw)],
        [cx + cos_p * ( hl) - sin_p * ( hw), cy + sin_p * ( hl) + cos_p * ( hw)],
        [cx + cos_p * (-hl) - sin_p * ( hw), cy + sin_p * (-hl) + cos_p * ( hw)],
    ])


def rect_from_plate(plate) -> tuple[np.ndarray, float, float, float]:
    """Extract geometry from a ReedPlate for fast checks.

    Returns:
        (corners, cx, cy, phi) where corners is (4, 2).
    """
    cx, cy = plate.center
    corners = rect_corners(cx, cy, plate.spec.length, plate.spec.width, plate.phi)
    return corners, cx, cy, plate.phi


# ---------------------------------------------------------------------------
# Separating Axis Theorem (SAT) for rotated rectangle overlap
# ---------------------------------------------------------------------------

def _project_polygon(corners: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    """Project polygon corners onto an axis. Returns (min, max) projection."""
    dots = corners @ axis
    return float(dots.min()), float(dots.max())


def rects_overlap(
    corners_a: np.ndarray,
    corners_b: np.ndarray,
    clearance: float = 0.0,
) -> bool:
    """Test if two rotated rectangles overlap using SAT.

    Args:
        corners_a: (4, 2) corners of rectangle A.
        corners_b: (4, 2) corners of rectangle B.
        clearance: Extra buffer distance. If > 0, rects "overlap" when
                   they are within this distance of each other.

    Returns:
        True if the rectangles overlap (or are within clearance distance).
    """
    # SAT: test 4 axes (2 edge normals per rectangle)
    for corners in (corners_a, corners_b):
        for i in range(2):  # only need 2 unique edge normals for a rectangle
            edge = corners[(i + 1) % 4] - corners[i]
            # Normal (perpendicular)
            axis = np.array([-edge[1], edge[0]])
            norm = math.sqrt(axis[0]**2 + axis[1]**2)
            if norm < 1e-10:
                continue
            axis /= norm

            min_a, max_a = _project_polygon(corners_a, axis)
            min_b, max_b = _project_polygon(corners_b, axis)

            # Check separation (with clearance buffer)
            if max_a + clearance < min_b or max_b + clearance < min_a:
                return False  # separating axis found → no overlap

    return True  # no separating axis → overlap


def rect_overlap_area_approx(
    corners_a: np.ndarray,
    corners_b: np.ndarray,
) -> float:
    """Approximate overlap area between two rotated rectangles.

    Uses the minimum overlap depth across SAT axes times the
    perpendicular extent. Fast approximation, not exact.
    """
    min_depth = float("inf")
    min_extent = 0.0

    for corners in (corners_a, corners_b):
        for i in range(2):
            edge = corners[(i + 1) % 4] - corners[i]
            axis = np.array([-edge[1], edge[0]])
            norm = math.sqrt(axis[0]**2 + axis[1]**2)
            if norm < 1e-10:
                continue
            axis /= norm

            min_a, max_a = _project_polygon(corners_a, axis)
            min_b, max_b = _project_polygon(corners_b, axis)

            overlap = min(max_a, max_b) - max(min_a, min_b)
            if overlap <= 0:
                return 0.0

            if overlap < min_depth:
                min_depth = overlap
                # Extent on perpendicular axis
                perp = np.array([axis[1], -axis[0]])
                _, max_pa = _project_polygon(corners_a, perp)
                min_pa, _ = _project_polygon(corners_a, perp)
                _, max_pb = _project_polygon(corners_b, perp)
                min_pb, _ = _project_polygon(corners_b, perp)
                min_extent = min(max_pa - min_pa, max_pb - min_pb)

    return min_depth * min_extent if min_depth < float("inf") else 0.0


# ---------------------------------------------------------------------------
# Line segment - rotated rectangle distance
# ---------------------------------------------------------------------------

def _point_to_segment_dist_sq(
    px: float, py: float,
    ax: float, ay: float,
    bx: float, by: float,
) -> float:
    """Squared distance from point (px,py) to segment (ax,ay)-(bx,by)."""
    dx = bx - ax
    dy = by - ay
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-10:
        return (px - ax)**2 + (py - ay)**2
    t = ((px - ax) * dx + (py - ay) * dy) / len_sq
    t = max(0.0, min(1.0, t))
    cx = ax + t * dx
    cy = ay + t * dy
    return (px - cx)**2 + (py - cy)**2


def _segments_min_dist_sq(
    a1x: float, a1y: float, a2x: float, a2y: float,
    b1x: float, b1y: float, b2x: float, b2y: float,
) -> float:
    """Minimum squared distance between two line segments.

    Checks endpoints-to-segment distances. Also checks for intersection.
    """
    # Check if segments intersect
    if _segments_intersect(a1x, a1y, a2x, a2y, b1x, b1y, b2x, b2y):
        return 0.0

    # Min of all endpoint-to-segment distances
    d = _point_to_segment_dist_sq(a1x, a1y, b1x, b1y, b2x, b2y)
    d = min(d, _point_to_segment_dist_sq(a2x, a2y, b1x, b1y, b2x, b2y))
    d = min(d, _point_to_segment_dist_sq(b1x, b1y, a1x, a1y, a2x, a2y))
    d = min(d, _point_to_segment_dist_sq(b2x, b2y, a1x, a1y, a2x, a2y))
    return d


def _cross2d(ox: float, oy: float, ax: float, ay: float, bx: float, by: float) -> float:
    """2D cross product of (a-o) x (b-o)."""
    return (ax - ox) * (by - oy) - (ay - oy) * (bx - ox)


def _segments_intersect(
    a1x: float, a1y: float, a2x: float, a2y: float,
    b1x: float, b1y: float, b2x: float, b2y: float,
) -> bool:
    """Test if two line segments intersect."""
    d1 = _cross2d(b1x, b1y, b2x, b2y, a1x, a1y)
    d2 = _cross2d(b1x, b1y, b2x, b2y, a2x, a2y)
    d3 = _cross2d(a1x, a1y, a2x, a2y, b1x, b1y)
    d4 = _cross2d(a1x, a1y, a2x, a2y, b2x, b2y)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True

    # Collinear cases (rare, skip for performance)
    return False


def segment_to_rect_dist(
    seg_start: tuple[float, float],
    seg_end: tuple[float, float],
    corners: np.ndarray,
) -> float:
    """Minimum distance from a line segment to a rotated rectangle.

    Args:
        seg_start: (x, y) of segment start.
        seg_end: (x, y) of segment end.
        corners: (4, 2) rectangle corners.

    Returns:
        Minimum distance in mm. 0 if they intersect.
    """
    sx, sy = seg_start
    ex, ey = seg_end

    min_dist_sq = float("inf")

    # Check segment against each edge of the rectangle
    for i in range(4):
        j = (i + 1) % 4
        d = _segments_min_dist_sq(
            sx, sy, ex, ey,
            corners[i, 0], corners[i, 1],
            corners[j, 0], corners[j, 1],
        )
        if d < min_dist_sq:
            min_dist_sq = d
            if d == 0:
                return 0.0

    # Also check if segment endpoints are inside the rectangle
    # (handles case where segment is fully inside)
    if _point_in_convex_polygon(sx, sy, corners):
        return 0.0
    if _point_in_convex_polygon(ex, ey, corners):
        return 0.0

    return math.sqrt(min_dist_sq)


def _point_in_convex_polygon(px: float, py: float, corners: np.ndarray) -> bool:
    """Test if a point is inside a convex polygon (corners in order)."""
    n = len(corners)
    sign = None
    for i in range(n):
        j = (i + 1) % n
        cross = _cross2d(
            corners[i, 0], corners[i, 1],
            corners[j, 0], corners[j, 1],
            px, py,
        )
        if cross == 0:
            continue
        s = cross > 0
        if sign is None:
            sign = s
        elif s != sign:
            return False
    return True


# ---------------------------------------------------------------------------
# Line segment to circle distance
# ---------------------------------------------------------------------------

def segment_to_circle_dist(
    seg_start: tuple[float, float],
    seg_end: tuple[float, float],
    center: tuple[float, float],
    radius: float,
) -> float:
    """Minimum distance from a line segment to a circle boundary.

    Returns 0 if the segment intersects the circle.
    """
    dist_sq = _point_to_segment_dist_sq(
        center[0], center[1],
        seg_start[0], seg_start[1],
        seg_end[0], seg_end[1],
    )
    dist = math.sqrt(dist_sq)
    return max(0.0, dist - radius)


# ---------------------------------------------------------------------------
# Buffered rectangle corners (for clearance checks)
# ---------------------------------------------------------------------------

def rect_corners_buffered(
    cx: float, cy: float,
    length: float, width: float,
    phi: float,
    clearance: float,
) -> np.ndarray:
    """Corners of a rectangle expanded by clearance on all sides."""
    return rect_corners(cx, cy, length + 2 * clearance, width + 2 * clearance, phi)


# ---------------------------------------------------------------------------
# Pallet position computation
# ---------------------------------------------------------------------------

def lever_obstacle_corners(
    start: tuple[float, float],
    end: tuple[float, float],
    half_width: float,
) -> np.ndarray:
    """Create a (4, 2) rectangle representing a lever's physical footprint.

    The lever is a line segment from start to end, buffered by half_width
    on each side. Returns the 4 corners of this rectangle.
    """
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.sqrt(dx * dx + dy * dy)
    if length < 1e-10:
        return rect_corners(start[0], start[1], 0.1, half_width * 2, 0)

    # Unit normal perpendicular to the lever
    nx = -dy / length * half_width
    ny = dx / length * half_width

    return np.array([
        [start[0] + nx, start[1] + ny],
        [end[0] + nx, end[1] + ny],
        [end[0] - nx, end[1] - ny],
        [start[0] - nx, start[1] - ny],
    ])


def pallet_position(
    cx: float, cy: float,
    length: float,
    phi: float,
    offset_ratio: float = 0.2,
) -> tuple[float, float]:
    """Compute pallet position from plate geometry.

    The pallet is near the tip of the plate, offset from center
    along the plate axis.
    """
    offset = length / 2 - length * offset_ratio
    px = cx + offset * math.cos(phi)
    py = cy + offset * math.sin(phi)
    return (px, py)
