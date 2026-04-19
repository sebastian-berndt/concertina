"""CLI entry point for the concertina layout solver."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Parametric Concertina Compiler - optimize action board layout",
    )
    parser.add_argument(
        "--side",
        choices=["LH", "RH", "both"],
        default="RH",
        help="Which side to solve (default: RH)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config JSON file (uses defaults if not specified)",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        default=None,
        help="Override max iterations for Stage 1",
    )
    parser.add_argument(
        "--popsize",
        type=int,
        default=None,
        help="Override population size for Stage 1",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (-1 for all cores)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
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
        help="Save layout and convergence plots",
    )
    parser.add_argument(
        "--stage1-only",
        action="store_true",
        help="Run only Stage 1 (coarse optimization)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress during optimization",
    )

    args = parser.parse_args()

    # --- Load config ---
    from concertina.config import ConcertinaConfig
    if args.config:
        config = ConcertinaConfig.load(args.config)
        print(f"Loaded config from {args.config}")
    else:
        config = ConcertinaConfig.defaults()

    # --- Build solver config ---
    from concertina.solver import SolverConfig
    solver_config = SolverConfig(
        workers=args.workers,
        seed=args.seed,
        skip_stage2=args.stage1_only,
    )
    if args.maxiter is not None:
        solver_config.stage1_maxiter = args.maxiter
    if args.popsize is not None:
        solver_config.stage1_popsize = args.popsize

    # --- Determine which sides to solve ---
    sides = ["LH", "RH"] if args.side == "both" else [args.side]

    results = {}
    for side in sides:
        print(f"\n{'='*50}")
        print(f"Solving {side} side ({config.instrument.keys_per_side} keys)")
        print(f"{'='*50}")

        # Generate layout and reeds
        from concertina.hayden_layout import HaydenLayout
        from concertina.reed_specs import generate_reed_table

        layout = HaydenLayout.from_beaumont(side, config.instrument)
        buttons = layout.get_all_enabled()
        reed_specs = generate_reed_table(buttons, config.ratio, config.reeds)

        print(f"Buttons: {len(buttons)}")
        print(f"Reeds: {len(reed_specs)} ({reed_specs[0].note} to {reed_specs[-1].note})")
        print(f"Dimensions: Stage 1 = {len(reed_specs)*2}D, Stage 2 = {len(reed_specs)*3}D")

        # Run solver
        from concertina.solver import solve
        result = solve(
            layout, reed_specs, config, solver_config,
            verbose=args.verbose or True,
        )

        results[side] = result

        # Save plot
        if args.plot:
            import matplotlib
            matplotlib.use("Agg")
            from concertina.visualize import plot_layout, plot_convergence

            plot_path = f"layout_{side.lower()}.png"
            plot_layout(
                layout, result.reed_plates, result.lever_paths,
                title=f"{side} Layout (cost={result.cost:.0f})",
                save_path=plot_path,
            )
            print(f"Layout plot saved to {plot_path}")

            conv_path = f"convergence_{side.lower()}.png"
            plot_convergence(result.history, title=f"{side} Convergence", save_path=conv_path)
            print(f"Convergence plot saved to {conv_path}")

    # --- Save results ---
    output_data = {}
    for side, result in results.items():
        side_data = {
            "cost": result.cost,
            "cost_breakdown": {
                "reed_collision": result.cost_breakdown.reed_collision,
                "lever_collision": result.cost_breakdown.lever_collision,
                "lever_length": result.cost_breakdown.lever_length,
                "hex_area": result.cost_breakdown.hex_area,
                "ratio_deviation": result.cost_breakdown.ratio_deviation,
            },
            "elapsed_seconds": result.elapsed_seconds,
            "reeds": [],
            "levers": [],
        }
        for plate in result.reed_plates:
            side_data["reeds"].append({
                "note": plate.spec.note,
                "r": round(plate.r, 2),
                "theta": round(plate.theta, 4),
                "phi": round(plate.phi, 4),
                "center": [round(c, 2) for c in plate.center],
                "pallet": [round(c, 2) for c in plate.pallet_position],
                "size": [plate.spec.length, plate.spec.width],
            })
        for lp in result.lever_paths:
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

    # Include config used for reproducibility
    from dataclasses import asdict
    output_data["config"] = asdict(config)

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
