"""CLI entry point for the concertina layout solver."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Parametric Concertina Compiler - optimize action board layout",
    )
    parser.add_argument(
        "--side",
        choices=["LH", "RH", "both"],
        default="both",
        help="Which side to solve (default: both)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON file (uses defaults if not specified)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="result.json",
        help="Output file for results (default: result.json)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save layout plot to layout.png",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress",
    )

    args = parser.parse_args()

    from concertina.config import ConcertinaConfig
    if args.config:
        config = ConcertinaConfig.load(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = ConcertinaConfig.defaults()

    sides = ["LH", "RH"] if args.side == "both" else [args.side]

    from concertina.hayden_layout import HaydenLayout
    from concertina.reed_specs import generate_reed_table
    from concertina.sector_placer import sector_place
    from concertina.lever_router import route_all_levers

    all_results = {}

    for side in sides:
        print(f"\n{'='*50}")
        print(f"  {side} side")
        print(f"{'='*50}")

        t0 = time.time()

        layout = HaydenLayout.from_beaumont(side, config.instrument)
        buttons = sorted(layout.get_all_enabled(), key=lambda b: b.midi)
        reed_specs = generate_reed_table(buttons, config.ratio, config.reeds)

        print(f"Buttons: {len(buttons)} ({reed_specs[0].note} to {reed_specs[-1].note})")

        # Sector-based placement with interleaved routing
        result = sector_place(layout, reed_specs, config, verbose=args.verbose)

        # Full lever routing with v2 visibility graph
        paths = route_all_levers(layout, result.plates, reed_specs)

        elapsed = time.time() - t0

        feasible = sum(1 for lp in paths if lp.is_feasible)
        straight = sum(1 for lp in paths if lp.segments == 1 and lp.is_feasible)
        dogleg = feasible - straight
        lengths = [lp.total_length for lp in paths if lp.is_feasible]
        ratios = [lp.actual_ratio for lp in paths if lp.is_feasible]

        print(f"Feasible: {feasible}/{len(paths)} ({straight} straight, {dogleg} dogleg)")
        if lengths:
            print(f"Levers:  {min(lengths):.0f}–{max(lengths):.0f}mm")
            print(f"Ratios:  {min(ratios):.2f}–{max(ratios):.2f}")
        print(f"Time:    {elapsed:.1f}s")

        for lp, btn in zip(paths, buttons):
            if not lp.is_feasible:
                print(f"  WARNING: {btn.note} infeasible (len={lp.total_length:.0f}mm)")

        all_results[side] = {
            "layout": layout,
            "reed_specs": reed_specs,
            "plates": result.plates,
            "paths": paths,
            "elapsed": elapsed,
        }

    # --- Save plot ---
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from concertina.visualize import plot_layout

        n = len(sides)
        fig, axes = plt.subplots(1, n, figsize=(11 * n, 11))
        if n == 1:
            axes = [axes]

        for i, side in enumerate(sides):
            r = all_results[side]
            feasible = sum(1 for lp in r["paths"] if lp.is_feasible)
            straight = sum(1 for lp in r["paths"] if lp.segments == 1 and lp.is_feasible)
            dogleg = feasible - straight
            plot_layout(
                r["layout"], r["plates"], r["paths"],
                title=f"{side} — {feasible}/{len(r['paths'])} ({straight} straight, {dogleg} dogleg)",
                ax=axes[i],
            )

        fig.tight_layout()
        fig.savefig("layout.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\nPlot saved to layout.png")

    # --- Save JSON ---
    from dataclasses import asdict

    output_data = {}
    for side in sides:
        r = all_results[side]
        side_data = {
            "elapsed_seconds": r["elapsed"],
            "reeds": [],
            "levers": [],
        }
        for plate in r["plates"]:
            side_data["reeds"].append({
                "note": plate.spec.note,
                "r": round(plate.r, 2),
                "theta": round(plate.theta, 4),
                "phi": round(plate.phi, 4),
                "center": [round(c, 2) for c in plate.center],
                "pallet": [round(c, 2) for c in plate.pallet_position],
                "size": [plate.spec.length, plate.spec.width],
            })
        for lp in r["paths"]:
            side_data["levers"].append({
                "button": [round(c, 2) for c in lp.button_pos],
                "pallet": [round(c, 2) for c in lp.pallet_pos],
                "pivot": [round(c, 2) for c in lp.pivot_pos],
                "length": round(lp.total_length, 2),
                "segments": lp.segments,
                "ratio": round(lp.actual_ratio, 3),
                "feasible": lp.is_feasible,
            })
        output_data[side] = side_data

    output_data["config"] = asdict(config)

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
