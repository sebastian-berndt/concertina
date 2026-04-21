"""Bank-aware reed placement.

Places reed banks and individual plates on the reed pan inside
the hexagonal boundary. Banks are oriented with their pallet edge
facing the button grid center. Individual bass plates go at the edges.

Uses numpy geometry for all collision checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from concertina.config import ConcertinaConfig
from concertina.geometry import (
    rect_corners_buffered,
    rects_overlap,
    rect_in_hexagon,
    segment_to_circle_dist,
)
from concertina.hayden_layout import HaydenLayout, HaydenButton
from concertina.reed_specs import ReedSpec
from concertina.reed_bank import (
    ReedBank, PlacedBank, PlacedIndividual, ReedPanLayout, assign_banks,
)


@dataclass
class BankPlacementResult:
    """Result of bank-aware placement."""

    pan_layout: ReedPanLayout
    feasible_banks: int
    feasible_individuals: int


def bank_place(
    layout: HaydenLayout,
    reed_specs: list[ReedSpec],
    config: ConcertinaConfig | None = None,
    target_lever_length: float = 60.0,
    individual_threshold: float = 40.0,
    verbose: bool = False,
) -> BankPlacementResult:
    """Place reed banks and individual plates on the reed pan.

    Strategy:
    1. Assign reeds to banks (by Hayden row) and individuals (large bass)
    2. Place banks first (they're rigid, fewer objects)
    3. Place individuals in remaining space

    Banks are oriented with their pallet edge facing the button centroid.
    """
    config = config or ConcertinaConfig.defaults()
    clearance = config.clearance.static_floor
    hex_af = config.hex_boundary.across_flats - 2 * config.hex_boundary.wall_thickness

    buttons = sorted(layout.get_all_enabled(), key=lambda b: b.midi)
    btn_xy = {b.note: (b.x, b.y) for b in buttons}

    # Centroid of buttons this bank serves
    def _btn_centroid(reeds: list[ReedSpec]) -> tuple[float, float]:
        xs = [btn_xy[r.note][0] for r in reeds if r.note in btn_xy]
        ys = [btn_xy[r.note][1] for r in reeds if r.note in btn_xy]
        if not xs:
            return (0.0, 0.0)
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    # Assign to banks and individuals
    banks, individuals = assign_banks(layout, reed_specs, config, individual_threshold)

    if verbose:
        print(f"  {len(banks)} banks + {len(individuals)} individuals")

    placed_corners: list[np.ndarray] = []
    placed_banks: list[PlacedBank] = []
    placed_individuals: list[PlacedIndividual] = []

    # --- Place banks first (largest/widest first) ---
    bank_order = sorted(range(len(banks)), key=lambda i: -banks[i].width)

    r_step = 2.0
    theta_step = math.radians(3)
    radii = np.arange(30, hex_af / 2 + r_step, r_step)
    thetas = np.arange(0, 2 * math.pi, theta_step)
    # Try multiple orientations for each bank
    phi_offsets = np.linspace(-math.pi / 2, math.pi / 2, 7)

    for bi in bank_order:
        bank = banks[bi]
        bcx, bcy = _btn_centroid(bank.reeds)
        w, d = bank.footprint

        best = None
        best_score = float("inf")

        for r in radii:
            for theta in thetas:
                cx = r * math.cos(theta)
                cy = r * math.sin(theta)

                # Base phi: pallet edge faces toward button centroid
                phi_base = math.atan2(bcy - cy, bcx - cx)

                for dphi in phi_offsets:
                    phi = phi_base + dphi

                    pb = PlacedBank(bank=bank, cx=cx, cy=cy, phi=phi)
                    corners = pb.get_corners(clearance)

                    if not rect_in_hexagon(corners, hex_af):
                        continue

                    if any(rects_overlap(corners, ec) for ec in placed_corners):
                        continue

                    # Score: mean lever length deviation from target
                    pallets = pb.pallet_positions()
                    lever_devs = []
                    for reed, pallet in zip(bank.reeds, pallets):
                        if reed.note in btn_xy:
                            bx, by = btn_xy[reed.note]
                            lever_len = math.sqrt((bx - pallet[0])**2 + (by - pallet[1])**2)
                            lever_devs.append(abs(lever_len - target_lever_length))

                    if not lever_devs:
                        continue

                    score = sum(lever_devs) / len(lever_devs) + r * 0.05

                    if score < best_score:
                        best_score = score
                        best = PlacedBank(bank=bank, cx=cx, cy=cy, phi=phi)

        if best is not None:
            placed_banks.append(best)
            placed_corners.append(best.get_corners(clearance))
            if verbose:
                notes = [r.note for r in bank.reeds]
                print(f"  Bank [{','.join(notes)}]: r={math.sqrt(best.cx**2+best.cy**2):.0f}mm [OK]")
        else:
            if verbose:
                notes = [r.note for r in bank.reeds]
                print(f"  Bank [{','.join(notes)}]: FAILED")

    # --- Place individual plates ---
    ind_order = sorted(range(len(individuals)), key=lambda i: -individuals[i].length * individuals[i].width)
    depth = config.reeds.bank_depth

    for ii in ind_order:
        spec = individuals[ii]
        bx, by = btn_xy.get(spec.note, (0, 0))
        w = spec.width
        d = depth

        best = None
        best_score = float("inf")

        for r in radii:
            for theta in thetas:
                cx = r * math.cos(theta)
                cy = r * math.sin(theta)
                phi_base = math.atan2(by - cy, bx - cx)

                for dphi in phi_offsets:
                    phi = phi_base + dphi

                    pi_candidate = PlacedIndividual(spec=spec, cx=cx, cy=cy, phi=phi)
                    corners = pi_candidate.get_corners(clearance)

                    if not rect_in_hexagon(corners, hex_af):
                        continue

                    if any(rects_overlap(corners, ec) for ec in placed_corners):
                        continue

                    px, py = pi_candidate.pallet_position()
                    lever_len = math.sqrt((bx - px)**2 + (by - py)**2)
                    score = abs(lever_len - target_lever_length) + r * 0.05

                    if score < best_score:
                        best_score = score
                        best = pi_candidate

        if best is not None:
            placed_individuals.append(best)
            placed_corners.append(best.get_corners(clearance))
            if verbose:
                print(f"  Individual {spec.note}: [OK]")
        else:
            if verbose:
                print(f"  Individual {spec.note}: FAILED")

    pan_layout = ReedPanLayout(banks=placed_banks, individuals=placed_individuals)

    return BankPlacementResult(
        pan_layout=pan_layout,
        feasible_banks=len(placed_banks),
        feasible_individuals=len(placed_individuals),
    )
