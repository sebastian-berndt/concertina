"""Reed bank model for grouped reed mounting.

A reed bank is a block with multiple reeds mounted side by side,
standing vertically on the reed pan. The 2D footprint is:
- Width: sum of individual reed widths + wall between each
- Depth: block depth (configurable, ~20mm)

Reed length goes vertical (irrelevant for 2D packing).
Pallet holes line up along the width edge facing the buttons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from concertina.config import ConcertinaConfig, ReedDimensions
from concertina.reed_specs import ReedSpec
from concertina.geometry import (
    rect_corners,
    rect_corners_buffered,
    pallet_position as _geo_pallet_position,
)


@dataclass
class ReedBank:
    """A group of reeds mounted side by side on a single block.

    Reeds are sorted by pitch (low to high). The block stands
    vertically on the reed pan. Pallet holes run along the
    width edge (the long edge, sum of reed widths).
    """

    reeds: list[ReedSpec]
    wall_thickness: float = 2.0     # mm between reed slots
    depth: float = 20.0             # mm, block depth on the reed pan

    @property
    def width(self) -> float:
        """Total width = sum of reed widths + walls."""
        n = len(self.reeds)
        return sum(r.width for r in self.reeds) + self.wall_thickness * (n + 1)

    @property
    def height(self) -> float:
        """Height = longest reed length (vertical, for bellows clearance)."""
        return max(r.length for r in self.reeds)

    @property
    def footprint(self) -> tuple[float, float]:
        """2D footprint on reed pan: (width, depth)."""
        return (self.width, self.depth)

    def pallet_offsets(self) -> list[float]:
        """Offset of each pallet hole along the width edge, from the left end.

        Each pallet sits at the center of its reed's slot.
        Returns offsets in mm from the left edge of the bank.
        """
        offsets = []
        pos = self.wall_thickness  # start after first wall
        for reed in self.reeds:
            offsets.append(pos + reed.width / 2)
            pos += reed.width + self.wall_thickness
        return offsets


@dataclass
class PlacedBank:
    """A reed bank placed at a specific position and orientation on the reed pan."""

    bank: ReedBank
    cx: float                   # center x on reed pan
    cy: float                   # center y on reed pan
    phi: float                  # rotation angle (radians)

    def get_corners(self, clearance: float = 0.0) -> np.ndarray:
        """4 corners of the bank footprint (width x depth rectangle)."""
        w, d = self.bank.footprint
        return rect_corners_buffered(
            self.cx, self.cy, w, d, self.phi, clearance,
        )

    def pallet_positions(self) -> list[tuple[float, float]]:
        """World-space (x, y) of each pallet hole.

        Pallets are along the width edge, on the side facing
        the pallet_edge direction (perpendicular to phi, offset
        by depth/2 from center).
        """
        offsets = self.bank.pallet_offsets()
        w, d = self.bank.footprint
        positions = []

        cos_p = math.cos(self.phi)
        sin_p = math.sin(self.phi)

        for offset in offsets:
            # Position along the width axis (centered)
            along = offset - w / 2

            # Pallet edge is at +depth/2 from center in the local y direction
            # (the edge facing inward toward buttons)
            local_x = along
            local_y = -d / 2  # pallet edge (toward buttons)

            # Rotate and translate to world space
            wx = self.cx + cos_p * local_x - sin_p * local_y
            wy = self.cy + sin_p * local_x + cos_p * local_y

            positions.append((wx, wy))

        return positions

    def reed_note_at_pallet(self, pallet_index: int) -> str:
        """Note name for the reed at a given pallet index."""
        return self.bank.reeds[pallet_index].note


@dataclass
class PlacedIndividual:
    """An individual reed plate (not on a bank) placed on the reed pan."""

    spec: ReedSpec
    cx: float
    cy: float
    phi: float

    @property
    def footprint(self) -> tuple[float, float]:
        """2D footprint: (width, depth). Width = reed width, depth = config depth."""
        return (self.spec.width, 20.0)  # individual plate standing vertically

    def get_corners(self, clearance: float = 0.0) -> np.ndarray:
        w, d = self.footprint
        return rect_corners_buffered(self.cx, self.cy, w, d, self.phi, clearance)

    def pallet_position(self) -> tuple[float, float]:
        """Pallet hole position (center of the plate, pallet edge)."""
        w, d = self.footprint
        cos_p = math.cos(self.phi)
        sin_p = math.sin(self.phi)
        local_y = -d / 2
        wx = self.cx - sin_p * local_y
        wy = self.cy + cos_p * local_y
        return (wx, wy)


@dataclass
class ReedPanLayout:
    """Complete reed pan layout: mix of banks and individual plates."""

    banks: list[PlacedBank]
    individuals: list[PlacedIndividual]

    def all_pallet_positions(self) -> dict[str, tuple[float, float]]:
        """Map note name → pallet (x, y) for every reed."""
        pallets = {}
        for pb in self.banks:
            positions = pb.pallet_positions()
            for i, reed in enumerate(pb.bank.reeds):
                pallets[reed.note] = positions[i]
        for pi in self.individuals:
            pallets[pi.spec.note] = pi.pallet_position()
        return pallets

    def all_footprint_corners(self, clearance: float = 0.0) -> list[np.ndarray]:
        """All bank + individual footprint corners for collision checking."""
        corners = []
        for pb in self.banks:
            corners.append(pb.get_corners(clearance))
        for pi in self.individuals:
            corners.append(pi.get_corners(clearance))
        return corners


def assign_banks(
    layout,
    reed_specs: list[ReedSpec],
    config: ConcertinaConfig | None = None,
    individual_threshold: float = 40.0,
    max_bank_width: float | None = None,
) -> tuple[list[ReedBank], list[ReedSpec]]:
    """Assign reeds to banks grouped by Hayden row.

    Large bass reeds (length > individual_threshold) stay as individuals.
    Remaining reeds are grouped by their Hayden row into banks.
    Banks wider than max_bank_width are split.

    Args:
        layout: HaydenLayout with button positions.
        reed_specs: All reed specs sorted by MIDI.
        config: Configuration for wall thickness and depth.
        individual_threshold: Reed length above which it stays individual.
        max_bank_width: Maximum bank width in mm. Defaults to hex inner radius.

    Returns:
        (banks, individuals): list of ReedBanks and list of individual ReedSpecs.
    """
    config = config or ConcertinaConfig.defaults()
    wall = config.reeds.bank_wall_thickness
    depth = config.reeds.bank_depth

    if max_bank_width is None:
        max_bank_width = config.hex_boundary.inner_radius * 1.5

    buttons = sorted(layout.get_all_enabled(), key=lambda b: b.midi)
    btn_by_note = {b.note: b for b in buttons}

    # Split into individuals (large bass) and bank candidates
    individuals = []
    bank_candidates: dict[int, list[ReedSpec]] = {}  # row → reeds

    for spec in reed_specs:
        if spec.length >= individual_threshold:
            individuals.append(spec)
        else:
            btn = btn_by_note.get(spec.note)
            if btn is None:
                individuals.append(spec)
                continue
            row = btn.row
            bank_candidates.setdefault(row, []).append(spec)

    # Build banks from each row, splitting if too wide
    banks = []
    for row in sorted(bank_candidates.keys()):
        row_reeds = sorted(bank_candidates[row], key=lambda r: r.midi)

        # Check if the row fits in one bank
        current_group: list[ReedSpec] = []
        current_width = wall  # start with one wall

        for reed in row_reeds:
            new_width = current_width + reed.width + wall
            if new_width > max_bank_width and current_group:
                # Split: save current bank, start new one
                banks.append(ReedBank(
                    reeds=list(current_group),
                    wall_thickness=wall,
                    depth=depth,
                ))
                current_group = [reed]
                current_width = wall + reed.width + wall
            else:
                current_group.append(reed)
                current_width = new_width

        if current_group:
            banks.append(ReedBank(
                reeds=current_group,
                wall_thickness=wall,
                depth=depth,
            ))

    return banks, individuals
