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
for _oct in range(0, 9):
    for _i, _name in enumerate(_NOTE_NAMES):
        _NOTE_MIDI[f"{_name}{_oct}"] = 12 + _oct * 12 + _i


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
        """Generate the standard 26-key Beaumont layout for one side.

        LH = bass/tenor focused, RH = alto/treble focused.
        The Beaumont uses 6 rows of buttons per side.
        """
        layout = cls(side=side, instrument=instrument)

        if side == "LH":
            # Left Hand (26 Notes) - Bass/Tenor focused
            # Row index, list of (Note, Column)
            layout.add_row(-2, [("G2", 0), ("A2", 1), ("B2", 2)])
            layout.add_row(-1, [("C3", -1), ("D3", 0), ("E3", 1), ("F#3", 2)])
            layout.add_row(0, [("F3", -2), ("G3", -1), ("A3", 0), ("B3", 1), ("C#4", 2)])
            layout.add_row(1, [("Bb3", -2), ("C4", -1), ("D4", 0), ("E4", 1), ("F#4", 2)])
            layout.add_row(2, [("Eb4", -2), ("F4", -1), ("G4", 0), ("A4", 1), ("B4", 2)])
            layout.add_row(3, [("Ab4", -1), ("Bb4", 0), ("C5", 1), ("D5", 2)])
        elif side == "RH":
            # Right Hand (26 Notes) - Alto/Treble focused
            layout.add_row(-2, [("C4", 0), ("D4", 1), ("E4", 2)])
            layout.add_row(-1, [("F4", -1), ("G4", 0), ("A4", 1), ("B4", 2)])
            layout.add_row(0, [("Bb4", -2), ("C5", -1), ("D5", 0), ("E5", 1), ("F#5", 2)])
            layout.add_row(1, [("Eb5", -2), ("F5", -1), ("G5", 0), ("A5", 1), ("B5", 2)])
            layout.add_row(2, [("Ab5", -2), ("Bb5", -1), ("C6", 0), ("D6", 1), ("E6", 2)])
            layout.add_row(3, [("F6", -1), ("G6", 0), ("A6", 1), ("B6", 2)])
        else:
            raise ValueError(f"side must be 'LH' or 'RH', got '{side}'")

        layout.center_on_origin()
        return layout
