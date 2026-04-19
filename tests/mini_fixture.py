"""Mini 6-button, 6-reed test fixture for fast development iteration.

Provides a small-scale version of the full problem that runs in seconds.
Used by all test modules from Phase 3.5 onward.
"""

from concertina.config import ConcertinaConfig, InstrumentSpec, RatioSpec
from concertina.hayden_layout import HaydenLayout, HaydenButton
from concertina.reed_specs import generate_reed_table, ReedSpec, ReedPlate


def make_mini_layout(instrument: InstrumentSpec | None = None) -> HaydenLayout:
    """Create a 6-button layout: single row of C4-A4."""
    if instrument is None:
        instrument = InstrumentSpec()
    layout = HaydenLayout(side="MINI", instrument=instrument)
    layout.add_row(0, [
        ("C4", -2), ("D4", -1), ("E4", 0),
        ("F4", 1), ("G4", 2), ("A4", 3),
    ])
    layout.center_on_origin()
    return layout


def make_mini_reeds(
    layout: HaydenLayout,
    ratio_spec: RatioSpec | None = None,
) -> list[ReedSpec]:
    """Generate 6 reed specs for the mini layout."""
    return generate_reed_table(layout.get_all_enabled(), ratio_spec)


def make_mini_good_placement(reeds: list[ReedSpec]) -> list[ReedPlate]:
    """A known-good placement where all levers should be straight.

    Reeds are placed in a wide fan below the button row,
    far enough apart that nothing collides.
    """
    import math
    plates = []
    n = len(reeds)
    for i, spec in enumerate(reeds):
        # Wide fan from -80deg to +80deg, at large radius to avoid overlap
        t = i / (n - 1) if n > 1 else 0.5
        theta = -1.4 + t * 2.8  # ~-80 to +80 degrees
        r = 80.0 + i * 5  # generous spacing
        phi = theta + math.pi  # point back toward center
        plates.append(ReedPlate(spec=spec, r=r, theta=theta, phi=phi))
    return plates


def make_mini_bad_placement(reeds: list[ReedSpec]) -> list[ReedPlate]:
    """A known-bad placement with forced collisions.

    All reeds stacked on top of each other at the same position.
    """
    plates = []
    for spec in reeds:
        plates.append(ReedPlate(spec=spec, r=50.0, theta=0.0, phi=0.0))
    return plates


def make_mini_config() -> ConcertinaConfig:
    """Config tuned for the mini problem."""
    config = ConcertinaConfig.defaults()
    return config
