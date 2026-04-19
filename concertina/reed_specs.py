"""Reed plate dimensions and geometry.

Accordion reed plates are rectangular. Their sizes taper logarithmically
from bass (large) to treble (small). Each reed plate is placed in polar
coordinates (r, theta) with a rotation angle (phi).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, Point

from concertina.config import RatioSpec, ReedDimensions
from concertina.hayden_layout import HaydenButton, note_to_midi


@dataclass
class ReedSpec:
    """Physical dimensions of one reed plate."""

    note: str
    midi: int
    length: float              # mm
    width: float               # mm
    target_ratio: float        # graduated leverage ratio for this note


@dataclass
class ReedPlate:
    """A reed plate placed at a specific position and orientation.

    Position is in polar coordinates relative to the button field center.
    """

    spec: ReedSpec
    r: float                   # mm, distance from origin
    theta: float               # radians, polar angle
    phi: float                 # radians, plate rotation

    @property
    def center(self) -> tuple[float, float]:
        """Cartesian center from polar coords."""
        return (self.r * math.cos(self.theta), self.r * math.sin(self.theta))

    def get_polygon(self, clearance: float = 0.0) -> Polygon:
        """Return the reed plate as a rotated rectangle (shapely Polygon).

        Args:
            clearance: Buffer distance added around the rectangle.
        """
        l = self.spec.length
        w = self.spec.width
        # Rectangle centered at origin
        rect = Polygon([
            (-l / 2, -w / 2), (l / 2, -w / 2),
            (l / 2, w / 2), (-l / 2, w / 2),
        ])
        # Rotate by phi around origin
        rect = rotate(rect, math.degrees(self.phi), origin=(0, 0))
        # Translate to position
        cx, cy = self.center
        rect = translate(rect, cx, cy)
        if clearance > 0:
            rect = rect.buffer(clearance)
        return rect

    @property
    def pallet_position(self) -> tuple[float, float]:
        """The pallet hole center -- near the tip of the reed plate.

        The pallet is offset from the plate center toward the button field
        (the "near" end), at pallet_offset_ratio from the tip.
        """
        offset = self.spec.length / 2 - self.spec.length * 0.2
        cx, cy = self.center
        # Offset along the plate's long axis (rotated by phi)
        px = cx + offset * math.cos(self.phi)
        py = cy + offset * math.sin(self.phi)
        return (px, py)

    @property
    def pallet_point(self) -> Point:
        """Pallet position as a shapely Point."""
        return Point(self.pallet_position)


def generate_reed_table(
    buttons: list[HaydenButton],
    ratio_spec: RatioSpec | None = None,
    reed_dims: ReedDimensions | None = None,
) -> list[ReedSpec]:
    """Generate reed specs for a list of buttons.

    Dimensions are log-interpolated from bass to treble.
    Ratios are linearly interpolated if graduated is enabled.

    Args:
        buttons: List of enabled buttons, determines which notes need reeds.
        ratio_spec: Ratio configuration. Uses defaults if None.
        reed_dims: Reed dimension ranges. Uses defaults if None.

    Returns:
        List of ReedSpec sorted by MIDI number (low to high).
    """
    if ratio_spec is None:
        ratio_spec = RatioSpec()
    if reed_dims is None:
        reed_dims = ReedDimensions()

    # Sort buttons by pitch
    sorted_buttons = sorted(buttons, key=lambda b: b.midi)
    n = len(sorted_buttons)
    if n == 0:
        return []

    specs = []
    for i, btn in enumerate(sorted_buttons):
        # Interpolation factor: 0 = lowest note (bass), 1 = highest (treble)
        t = i / (n - 1) if n > 1 else 0.5

        # Log-interpolate dimensions (reed plates don't shrink linearly)
        length = reed_dims.bass_length * (reed_dims.treble_length / reed_dims.bass_length) ** t
        width = reed_dims.bass_width * (reed_dims.treble_width / reed_dims.bass_width) ** t

        # Linear-interpolate ratio
        if ratio_spec.graduated:
            ratio = ratio_spec.bass_ratio + t * (ratio_spec.treble_ratio - ratio_spec.bass_ratio)
        else:
            ratio = ratio_spec.target_ratio

        # Clamp ratio to bounds
        ratio = max(ratio_spec.ratio_min, min(ratio_spec.ratio_max, ratio))

        specs.append(ReedSpec(
            note=btn.note,
            midi=btn.midi,
            length=round(length, 1),
            width=round(width, 1),
            target_ratio=round(ratio, 3),
        ))

    return specs
