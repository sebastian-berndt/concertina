"""2D visualization of concertina layouts.

Renders buttons, reed plates, lever paths, and obstacles
using matplotlib. Built early so every module can be visually debugged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon as ShapelyPolygon

if TYPE_CHECKING:
    from concertina.hayden_layout import HaydenLayout
    from concertina.reed_specs import ReedPlate


def plot_layout(
    layout: HaydenLayout,
    reed_plates: list[ReedPlate] | None = None,
    lever_paths: list | None = None,
    title: str = "",
    show_obstacles: bool = False,
    obstacle_field: object | None = None,
    config: object | None = None,
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Render the complete 2D layout.

    Args:
        layout: The button grid.
        reed_plates: Positioned reed plates (optional).
        lever_paths: Lever routing results (optional, for later phases).
        title: Plot title.
        show_obstacles: Draw no-go zones.
        obstacle_field: ObstacleField instance (for obstacle rendering).
        config: ConcertinaConfig for dimensions.
        save_path: Save figure to this path instead of showing.
        ax: Existing axes to draw on. Creates new figure if None.

    Returns:
        The matplotlib Figure.
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    else:
        fig = ax.get_figure()

    from concertina.config import InstrumentSpec
    instrument = config.instrument if config else InstrumentSpec()

    # --- Draw buttons ---
    buttons = layout.get_all_buttons()
    for btn in buttons:
        color = "#4488CC" if btn.enabled else "#AAAAAA"
        circle = plt.Circle(
            (btn.x, btn.y),
            instrument.button_radius,
            facecolor=color,
            edgecolor="black",
            linewidth=0.5,
            alpha=0.8,
        )
        ax.add_patch(circle)
        ax.annotate(
            btn.note,
            (btn.x, btn.y),
            ha="center", va="center",
            fontsize=5,
            color="white" if btn.enabled else "gray",
            fontweight="bold",
        )

    # --- Draw reed plates ---
    if reed_plates:
        for plate in reed_plates:
            poly = plate.get_polygon()
            xs, ys = poly.exterior.xy
            ax.fill(xs, ys, alpha=0.3, color="#44AA44", edgecolor="#226622", linewidth=1)

            # Pallet position
            px, py = plate.pallet_position
            ax.plot(px, py, "o", color="red", markersize=4, zorder=5)

            # Reed center label
            cx, cy = plate.center
            ax.annotate(
                plate.spec.note,
                (cx, cy),
                ha="center", va="center",
                fontsize=5,
                color="#224422",
            )

        # Draw convex hull of all reed plates
        from shapely.ops import unary_union
        all_polys = unary_union([p.get_polygon() for p in reed_plates])
        hull = all_polys.convex_hull
        hx, hy = hull.exterior.xy
        ax.plot(hx, hy, "--", color="gray", linewidth=1, alpha=0.5, label="Convex hull")

    # --- Draw hex boundary ---
    if config and hasattr(config, "hex_boundary"):
        from concertina.geometry import hexagon_corners
        hex_af = config.hex_boundary.across_flats - 2 * config.hex_boundary.wall_thickness
        hc = hexagon_corners(hex_af)
        # Close the hexagon
        hx = list(hc[:, 0]) + [hc[0, 0]]
        hy = list(hc[:, 1]) + [hc[0, 1]]
        ax.plot(hx, hy, "-", color="#AA4444", linewidth=1.5, alpha=0.6, label="Reed pan boundary")

    # --- Draw lever paths (Phase 5+) ---
    if lever_paths:
        for lp in lever_paths:
            path = lp.path
            xs, ys = path.xy
            color = "#44AA44" if lp.segments == 1 else "#DD8800"
            if not lp.is_feasible:
                color = "#DD2222"
            ax.plot(xs, ys, "-", color=color, linewidth=1, alpha=0.7)

            # Pivot point
            px, py = lp.pivot_pos
            ax.plot(px, py, "x", color="black", markersize=4, markeredgewidth=1)

    # --- Draw obstacles (optional) ---
    if show_obstacles and obstacle_field is not None:
        obstacles = obstacle_field.get_all_obstacles()
        for obs in obstacles:
            if hasattr(obs, "exterior"):
                xs, ys = obs.exterior.xy
                ax.fill(xs, ys, alpha=0.1, color="red", edgecolor="red", linewidth=0.5)

    # --- Formatting ---
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_title(title or f"Concertina Layout ({layout.side})")

    # Auto-scale with padding
    all_x = [b.x for b in buttons]
    all_y = [b.y for b in buttons]
    if reed_plates:
        for p in reed_plates:
            cx, cy = p.center
            all_x.append(cx)
            all_y.append(cy)
    if all_x and all_y:
        margin = 20
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig


def plot_convergence(
    history: list[float],
    title: str = "Solver Convergence",
    save_path: str | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot cost vs. generation number."""
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    else:
        fig = ax.get_figure()

    ax.semilogy(history)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cost (log scale)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig
