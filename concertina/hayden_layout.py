"""Hayden Duet button grid generation.

The Hayden layout is isomorphic: each step right is a whole tone,
each step up-right is a fifth. Rows are offset by half a pitch
to form the hexagonal pattern.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from concertina.config import InstrumentSpec


# Note name to MIDI number mapping (middle C = C4 = 60)
_NOTE_MIDI = {}
_NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
# Enharmonic aliases used in Hayden charts (sharps for black keys)
_ENHARMONIC = {"D#": "Eb", "G#": "Ab", "A#": "Bb"}
for _oct in range(0, 9):
    for _i, _name in enumerate(_NOTE_NAMES):
        _NOTE_MIDI[f"{_name}{_oct}"] = 12 + _oct * 12 + _i
    # Add sharp aliases
    for _sharp, _flat in _ENHARMONIC.items():
        _NOTE_MIDI[f"{_sharp}{_oct}"] = _NOTE_MIDI[f"{_flat}{_oct}"]


def note_to_midi(note: str) -> int:
    """Convert note name like 'C4' or 'Eb3' to MIDI number."""
    return _NOTE_MIDI[note]


@dataclass
class HaydenButton:
    """A single button on the Hayden grid."""

    note: str
    row: int
    col: int
    enabled: bool = True

    # Computed after placement
    x: float = 0.0
    y: float = 0.0

    @property
    def midi(self) -> int:
        return note_to_midi(self.note)

    @property
    def pos(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass
class HaydenRow:
    """A row of buttons in the Hayden grid."""

    index: int
    buttons: list[HaydenButton] = field(default_factory=list)

    def __iter__(self):
        return iter(self.buttons)

    def __len__(self):
        return len(self.buttons)


class HaydenLayout:
    """Complete button layout for one side of the instrument.

    Coordinate system: (0, 0) is the center of the button field.
    X increases to the right, Y increases upward.
    """

    def __init__(self, side: str, instrument: InstrumentSpec | None = None):
        if instrument is None:
            instrument = InstrumentSpec()
        self.side = side
        self.instrument = instrument
        self.rows: list[HaydenRow] = []

    def add_row(self, row_index: int, buttons: list[tuple[str, int]]) -> None:
        """Add a row of buttons.

        Args:
            row_index: Vertical index of this row.
            buttons: List of (note_name, col_index) tuples.
        """
        row = HaydenRow(index=row_index)
        for note, col in buttons:
            btn = HaydenButton(note=note, row=row_index, col=col)
            row.buttons.append(btn)
        self.rows.append(row)
        self._update_coordinates(row)

    def _update_coordinates(self, row: HaydenRow) -> None:
        """Calculate X, Y for every button in a row based on Hayden geometry."""
        for btn in row.buttons:
            # The 0.5 * row.index provides the hexagonal stagger
            btn.x = (btn.col + 0.5 * row.index) * self.instrument.h_pitch
            btn.y = row.index * self.instrument.v_pitch

    def get_all_enabled(self) -> list[HaydenButton]:
        """Return a flat list of all active buttons for the routing solver."""
        return [btn for row in self.rows for btn in row if btn.enabled]

    def get_all_buttons(self) -> list[HaydenButton]:
        """Return all buttons regardless of enabled state."""
        return [btn for row in self.rows for btn in row]

    def center_on_origin(self) -> None:
        """Shift all buttons so the centroid is at (0, 0)."""
        all_btns = self.get_all_buttons()
        if not all_btns:
            return
        cx = sum(b.x for b in all_btns) / len(all_btns)
        cy = sum(b.y for b in all_btns) / len(all_btns)
        for btn in all_btns:
            btn.x -= cx
            btn.y -= cy

    def get_neighbors(self, button: HaydenButton) -> list[HaydenButton]:
        """Return Hayden-grid-adjacent buttons (up to 6 hex neighbors)."""
        neighbors = []
        for other in self.get_all_enabled():
            if other is button:
                continue
            dr = other.row - button.row
            dc = other.col - button.col
            # In the Hayden hex grid, neighbors are:
            # Same row: (0, +/-1)
            # Adjacent row: (+/-1, 0) and (+/-1, -1) due to stagger
            if (dr, dc) in [(0, 1), (0, -1), (1, 0), (1, -1), (-1, 0), (-1, 1)]:
                neighbors.append(other)
        return neighbors

    @classmethod
    def from_beaumont(cls, side: str, instrument: InstrumentSpec | None = None) -> HaydenLayout:
        """Generate the Beaumont (R. Morse & Co.) button layout.

        Hayden Duet pattern: each step right = +2 semitones (whole tone),
        each row up = +7 semitones (perfect fifth) at the up-right neighbor.

        The Beaumont has 52 keys: 23 LH + 29 RH.
        Note layout verified against the official R. Morse & Co. Beaumont
        note chart (c' = middle C = C4).

        LH: 23 keys, Bb2 to B4 (4 rows)
        RH: 29 keys, C#4 to D6 (6 rows)
        """
        layout = cls(side=side, instrument=instrument)

        if side == "LH":
            # Left Hand (23 keys) — Bb2 to B4
            # Row, list of (Note, Column)
            # Hayden whole-tone rows alternate: Bb-type and Eb-type
            layout.add_row(0, [
                ("Bb2", 0), ("C3", 1), ("D3", 2), ("E3", 3), ("F#3", 4), ("G#3", 5),
            ])
            layout.add_row(1, [
                ("Eb3", -1), ("F3", 0), ("G3", 1), ("A3", 2), ("B3", 3), ("C#4", 4),
            ])
            layout.add_row(2, [
                ("Bb3", -1), ("C4", 0), ("D4", 1), ("E4", 2), ("F#4", 3), ("G#4", 4),
            ])
            layout.add_row(3, [
                ("Eb4", -2), ("F4", -1), ("G4", 0), ("A4", 1), ("B4", 2),
            ])
        elif side == "RH":
            # Right Hand (29 keys) — C#4 to D6
            # C#4 sits alone at far right of the bottom row
            layout.add_row(0, [
                ("C#4", 5),
            ])
            layout.add_row(1, [
                ("Bb3", 0), ("C4", 1), ("D4", 2), ("E4", 3), ("F#4", 4), ("G#4", 5),
            ])
            layout.add_row(2, [
                ("Eb4", -1), ("F4", 0), ("G4", 1), ("A4", 2), ("B4", 3), ("C#5", 4), ("D#5", 5),
            ])
            layout.add_row(3, [
                ("Bb4", -1), ("C5", 0), ("D5", 1), ("E5", 2), ("F#5", 3), ("G#5", 4),
            ])
            layout.add_row(4, [
                ("Eb5", -2), ("F5", -1), ("G5", 0), ("A5", 1), ("B5", 2), ("C#6", 3),
            ])
            layout.add_row(5, [
                ("Bb5", -2), ("C6", -1), ("D6", 0),
            ])
        else:
            raise ValueError(f"side must be 'LH' or 'RH', got '{side}'")

        layout.center_on_origin()
        return layout
