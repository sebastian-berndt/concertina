"""Reed plate dimensions and geometry.

Accordion reed plates are rectangular. Their sizes taper from bass (large)
to treble (small). Each reed plate is placed in polar coordinates
(r, theta) with a rotation angle (phi).

Dimensions can come from three sources:
1. Measured: exact per-note measurements from your reed plates
2. Preset: standard Italian accordion reed dimensions (BINCI_STANDARD)
3. Interpolated: log-interpolated from bass/treble extremes (fallback)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon, Point

from concertina.config import RatioSpec, ReedDimensions
from concertina.hayden_layout import HaydenButton, note_to_midi


# -----------------------------------------------------------------------
# Preset reed plate dimensions: (length_mm, width_mm) per note.
#
# Based on standard Italian accordion reed plates (tipo a mano style).
# These are the PLATE dimensions (mounting frame), not the tongue length.
# Measurements approximate — replace with your actual reed measurements.
#
# To use: pass as the `measured` argument to generate_reed_table().
# Any notes not in the dict fall back to interpolation.
# -----------------------------------------------------------------------

BINCI_STANDARD: dict[str, tuple[float, float]] = {
    # Octave 2 (deep bass)
    "Bb2": (54.0, 18.0),
    # Octave 3
    "C3":  (52.0, 17.5),
    "C#3": (50.5, 17.5),
    "D3":  (49.0, 17.0),
    "Eb3": (47.5, 17.0),
    "E3":  (46.0, 16.5),
    "F3":  (44.5, 16.5),
    "F#3": (43.0, 16.0),
    "G3":  (41.5, 16.0),
    "G#3": (40.0, 16.0),
    "A3":  (38.5, 15.5),
    "Bb3": (37.0, 15.5),
    "B3":  (36.0, 15.5),
    # Octave 4
    "C4":  (35.0, 15.0),
    "C#4": (34.0, 15.0),
    "D4":  (33.0, 15.0),
    "Eb4": (32.0, 14.5),
    "E4":  (31.0, 14.5),
    "F4":  (30.0, 14.5),
    "F#4": (29.0, 14.0),
    "G4":  (28.0, 14.0),
    "G#4": (27.5, 14.0),
    "A4":  (27.0, 13.5),
    "Bb4": (26.5, 13.5),
    "B4":  (26.0, 13.5),
    # Octave 5
    "C5":  (25.5, 13.0),
    "C#5": (25.0, 13.0),
    "D5":  (24.5, 13.0),
    "D#5": (24.0, 12.5),
    "Eb5": (24.0, 12.5),
    "E5":  (23.5, 12.5),
    "F5":  (23.0, 12.5),
    "F#5": (22.5, 12.0),
    "G5":  (22.0, 12.0),
    "G#5": (21.5, 12.0),
    "A5":  (21.0, 12.0),
    "Bb5": (20.5, 11.5),
    "B5":  (20.0, 11.5),
    # Octave 6
    "C6":  (19.5, 11.5),
    "C#6": (19.0, 11.0),
    "D6":  (18.5, 11.0),
}

# Enharmonic aliases (so both sharp and flat names work)
for _sharp, _flat in [("D#3", "Eb3"), ("G#3", "Ab3"), ("A#3", "Bb3"),
                       ("D#4", "Eb4"), ("G#4", "Ab4"), ("A#4", "Bb4"),
                       ("D#5", "Eb5"), ("G#5", "Ab5"), ("A#5", "Bb5"),
                       ("D#6", "Eb6"), ("A#2", "Bb2")]:
    if _flat in BINCI_STANDARD and _sharp not in BINCI_STANDARD:
        BINCI_STANDARD[_sharp] = BINCI_STANDARD[_flat]
    elif _sharp in BINCI_STANDARD and _flat not in BINCI_STANDARD:
        BINCI_STANDARD[_flat] = BINCI_STANDARD[_sharp]


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
    measured: dict[str, tuple[float, float]] | None = None,
) -> list[ReedSpec]:
    """Generate reed specs for a list of buttons.

    Dimensions are log-interpolated from bass to treble, unless
    exact measurements are provided via the `measured` dict.

    Args:
        buttons: List of enabled buttons, determines which notes need reeds.
        ratio_spec: Ratio configuration. Uses defaults if None.
        reed_dims: Reed dimension ranges. Uses defaults if None.
        measured: Optional dict mapping note name to (length, width) in mm.
                  Notes in this dict use exact dimensions instead of
                  interpolation. Example: {"C3": (52.0, 17.5), "D3": (49.0, 17.0)}

    Returns:
        List of ReedSpec sorted by MIDI number (low to high).
    """
    if ratio_spec is None:
        ratio_spec = RatioSpec()
    if reed_dims is None:
        reed_dims = ReedDimensions()
    if measured is None:
        measured = {}

    # Sort buttons by pitch
    sorted_buttons = sorted(buttons, key=lambda b: b.midi)
    n = len(sorted_buttons)
    if n == 0:
        return []

    specs = []
    for i, btn in enumerate(sorted_buttons):
        # Interpolation factor: 0 = lowest note (bass), 1 = highest (treble)
        t = i / (n - 1) if n > 1 else 0.5

        # Use measured dimensions if available, otherwise interpolate
        if btn.note in measured:
            length, width = measured[btn.note]
        else:
            # Log-interpolate dimensions (reed plates don't shrink linearly)
            length = reed_dims.bass_length * (reed_dims.treble_length / reed_dims.bass_length) ** t
            width = reed_dims.bass_width * (reed_dims.treble_width / reed_dims.bass_width) ** t
            length = round(length, 1)
            width = round(width, 1)

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
            length=length,
            width=width,
            target_ratio=round(ratio, 3),
        ))

    return specs
