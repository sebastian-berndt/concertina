"""Tests for the Hayden button grid."""

import math

from concertina.config import InstrumentSpec
from concertina.hayden_layout import HaydenLayout, HaydenButton, note_to_midi


def test_note_to_midi():
    assert note_to_midi("C4") == 60
    assert note_to_midi("A4") == 69
    assert note_to_midi("G2") == 43


def test_beaumont_lh_has_26_keys():
    layout = HaydenLayout.from_beaumont("LH")
    assert len(layout.get_all_enabled()) == 26


def test_beaumont_rh_has_26_keys():
    layout = HaydenLayout.from_beaumont("RH")
    assert len(layout.get_all_enabled()) == 26


def test_beaumont_centered():
    for side in ["LH", "RH"]:
        layout = HaydenLayout.from_beaumont(side)
        btns = layout.get_all_enabled()
        cx = sum(b.x for b in btns) / len(btns)
        cy = sum(b.y for b in btns) / len(btns)
        assert abs(cx) < 0.01, f"{side} not centered on X: {cx}"
        assert abs(cy) < 0.01, f"{side} not centered on Y: {cy}"


def test_neighbor_distance_equals_pitch():
    layout = HaydenLayout.from_beaumont("RH")
    instrument = InstrumentSpec()

    for btn in layout.get_all_enabled():
        neighbors = layout.get_neighbors(btn)
        for nbr in neighbors:
            dist = math.sqrt((btn.x - nbr.x) ** 2 + (btn.y - nbr.y) ** 2)
            assert abs(dist - instrument.h_pitch) < 0.1, (
                f"{btn.note}->{nbr.note}: dist={dist:.1f}, expected={instrument.h_pitch}"
            )


def test_no_buttons_overlap():
    layout = HaydenLayout.from_beaumont("LH")
    btns = layout.get_all_enabled()
    min_gap = InstrumentSpec().button_diameter
    for i, a in enumerate(btns):
        for b in btns[i + 1 :]:
            dist = math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)
            assert dist >= min_gap, f"{a.note} and {b.note} overlap: dist={dist:.1f}"


def test_neighbors_count():
    layout = HaydenLayout.from_beaumont("RH")
    for btn in layout.get_all_enabled():
        n = len(layout.get_neighbors(btn))
        # Corner buttons have 2-3, edge have 3-4, interior have 6
        assert 1 <= n <= 6, f"{btn.note} has {n} neighbors"


def test_enable_disable():
    layout = HaydenLayout.from_beaumont("RH")
    all_count = len(layout.get_all_enabled())
    # Disable one button
    for btn in layout.get_all_enabled():
        if btn.note == "F#5":
            btn.enabled = False
            break
    assert len(layout.get_all_enabled()) == all_count - 1


def test_custom_pitch():
    inst = InstrumentSpec(h_pitch=18.0)
    layout = HaydenLayout.from_beaumont("LH", instrument=inst)
    btns = layout.get_all_enabled()
    # Find two buttons in the same row
    row0 = [b for b in btns if b.row == 0]
    if len(row0) >= 2:
        row0.sort(key=lambda b: b.col)
        dx = row0[1].x - row0[0].x
        assert abs(dx - 18.0) < 0.01
