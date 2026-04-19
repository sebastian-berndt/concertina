"""Differential evolution solver for concertina layout optimization.

Default pipeline is two-stage:
1. Stage 1 (Coarse): Optimize r, theta only (phi fixed to point-at-center).
2. Stage 2 (Refine): Unlock phi and refine from Stage 1 result.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution, OptimizeResult

from concertina.config import ConcertinaConfig
from concertina.cost_function import evaluate, evaluate_detailed, decode_state, CostBreakdown
from concertina.hayden_layout import HaydenLayout
from concertina.lever_router import route_all_levers, LeverPath
from concertina.obstacles import ObstacleField
from concertina.reed_specs import ReedSpec, ReedPlate


@dataclass
class SolverConfig:
    """Tunable solver parameters."""

    # Stage 1 (coarse: r, theta only)
    stage1_maxiter: int = 300
    stage1_popsize: int = 15

    # Stage 2 (refine: r, theta, phi)
    stage2_maxiter: int = 200
    stage2_popsize: int = 10

    # Shared settings
    strategy: str = "best1bin"
    tol: float = 1e-4
    mutation: tuple[float, float] = (0.5, 1.0)
    recombination: float = 0.7
    polish: bool = True
    workers: int = 1           # -1 for all cores (requires picklable func)
    seed: int | None = 42

    # Which stages to run
    skip_stage1: bool = False  # jump straight to 3-param optimization
    skip_stage2: bool = False  # stop after stage 1


@dataclass
class SolverResult:
    """Optimization result with diagnostics."""

    x: np.ndarray                      # optimal state vector
    cost: float                        # final cost value
    cost_breakdown: CostBreakdown      # detailed penalty breakdown
    reed_plates: list[ReedPlate]       # final reed positions
    lever_paths: list[LeverPath]       # final lever routes
    history: list[float]               # cost per generation
    elapsed_seconds: float
    stage1_result: OptimizeResult | None = None
    stage2_result: OptimizeResult | None = None


def _make_bounds_stage1(n_reeds: int, config: ConcertinaConfig) -> list[tuple[float, float]]:
    """Bounds for Stage 1: [r, theta] per reed."""
    bounds = []
    b = config.bounds
    for _ in range(n_reeds):
        bounds.append((b.r_min, b.r_max))
        bounds.append((b.theta_min, b.theta_max))
    return bounds


def _make_bounds_stage2(n_reeds: int, config: ConcertinaConfig) -> list[tuple[float, float]]:
    """Bounds for Stage 2: [r, theta, phi] per reed."""
    bounds = []
    b = config.bounds
    for _ in range(n_reeds):
        bounds.append((b.r_min, b.r_max))
        bounds.append((b.theta_min, b.theta_max))
        bounds.append((b.phi_min, b.phi_max))
    return bounds


def _make_initial_guess_stage1(
    layout: HaydenLayout,
    reed_specs: list[ReedSpec],
    config: ConcertinaConfig,
) -> np.ndarray:
    """Generate a physically motivated starting point for Stage 1.

    Places reeds in a fan pattern: bass at wider radius, treble closer.
    Angles spread evenly around a semicircle below the button field.
    """
    n = len(reed_specs)
    x0 = np.zeros(n * 2)

    for i in range(n):
        t = i / (n - 1) if n > 1 else 0.5
        # Bass (t=0) at wider radius, treble (t=1) closer
        r = config.bounds.r_max * 0.8 - t * (config.bounds.r_max * 0.4)
        # Spread angles across lower semicircle
        theta = -np.pi * 0.7 + t * np.pi * 1.4
        x0[i * 2] = r
        x0[i * 2 + 1] = theta

    return x0


def _stage1_to_stage2(
    x_stage1: np.ndarray,
    n_reeds: int,
    config: ConcertinaConfig,
) -> np.ndarray:
    """Convert Stage 1 result (r, theta) to Stage 2 initial guess (r, theta, phi).

    Sets phi = theta + pi (point toward center), clamped to phi bounds.
    """
    b = config.bounds
    x_stage2 = np.zeros(n_reeds * 3)
    for i in range(n_reeds):
        r = x_stage1[i * 2]
        theta = x_stage1[i * 2 + 1]
        phi = theta + np.pi  # point toward center
        # Normalize phi to [-pi, pi] then clamp to bounds
        phi = (phi + np.pi) % (2 * np.pi) - np.pi
        phi = np.clip(phi, b.phi_min, b.phi_max)
        # Also clamp r and theta to be safe
        r = np.clip(r, b.r_min, b.r_max)
        theta = np.clip(theta, b.theta_min, b.theta_max)
        x_stage2[i * 3] = r
        x_stage2[i * 3 + 1] = theta
        x_stage2[i * 3 + 2] = phi
    return x_stage2


def solve(
    layout: HaydenLayout,
    reed_specs: list[ReedSpec],
    config: ConcertinaConfig | None = None,
    solver_config: SolverConfig | None = None,
    callback: Callable[[np.ndarray, float], None] | None = None,
    verbose: bool = False,
) -> SolverResult:
    """Run the two-stage differential evolution solver.

    Args:
        layout: Fixed button positions.
        reed_specs: Reed plate specifications (sorted by MIDI).
        config: Instrument/physics configuration.
        solver_config: Solver tuning parameters.
        callback: Called after each generation with (xk, convergence).
        verbose: Print progress to stdout.

    Returns:
        SolverResult with optimal layout and diagnostics.
    """
    config = config or ConcertinaConfig.defaults()
    solver_config = solver_config or SolverConfig()
    n_reeds = len(reed_specs)

    history: list[float] = []
    t_start = time.time()

    def _progress_callback(xk, convergence=0):
        cost = evaluate(xk, layout, reed_specs, config)
        history.append(cost)
        if verbose and len(history) % 10 == 0:
            print(f"  Gen {len(history):4d}: cost={cost:.1f}, convergence={convergence:.6f}")

    stage1_result = None
    stage2_result = None

    # ===== STAGE 1: Coarse (r, theta only) =====
    if not solver_config.skip_stage1:
        if verbose:
            print(f"Stage 1: {n_reeds * 2}D optimization (r, theta)")

        bounds1 = _make_bounds_stage1(n_reeds, config)
        x0_stage1 = _make_initial_guess_stage1(layout, reed_specs, config)

        stage1_result = differential_evolution(
            evaluate,
            bounds=bounds1,
            args=(layout, reed_specs, config),
            x0=x0_stage1,
            strategy=solver_config.strategy,
            maxiter=solver_config.stage1_maxiter,
            popsize=solver_config.stage1_popsize,
            tol=solver_config.tol,
            mutation=solver_config.mutation,
            recombination=solver_config.recombination,
            polish=False,  # save polish for Stage 2
            workers=solver_config.workers,
            seed=solver_config.seed,
            callback=_progress_callback,
        )

        if verbose:
            print(f"  Stage 1 done: cost={stage1_result.fun:.1f}, "
                  f"{stage1_result.nit} generations")

        x_current = _stage1_to_stage2(stage1_result.x, n_reeds, config)
    else:
        # Skip Stage 1: start with initial guess in 3-param format
        x0_s1 = _make_initial_guess_stage1(layout, reed_specs, config)
        x_current = _stage1_to_stage2(x0_s1, n_reeds, config)

    # ===== STAGE 2: Refine (r, theta, phi) =====
    if not solver_config.skip_stage2:
        if verbose:
            print(f"Stage 2: {n_reeds * 3}D optimization (r, theta, phi)")

        bounds2 = _make_bounds_stage2(n_reeds, config)

        stage2_result = differential_evolution(
            evaluate,
            bounds=bounds2,
            args=(layout, reed_specs, config),
            x0=x_current,
            strategy=solver_config.strategy,
            maxiter=solver_config.stage2_maxiter,
            popsize=solver_config.stage2_popsize,
            tol=solver_config.tol,
            mutation=solver_config.mutation,
            recombination=solver_config.recombination,
            polish=solver_config.polish,
            workers=solver_config.workers,
            seed=solver_config.seed,
            callback=_progress_callback,
        )

        if verbose:
            print(f"  Stage 2 done: cost={stage2_result.fun:.1f}, "
                  f"{stage2_result.nit} generations")

        x_final = stage2_result.x
    else:
        x_final = x_current

    # ===== Build final result =====
    elapsed = time.time() - t_start
    plates = decode_state(x_final, reed_specs)
    breakdown = evaluate_detailed(x_final, layout, reed_specs, config)
    obstacle_field = ObstacleField(layout, plates, config)
    lever_paths = route_all_levers(layout, plates, reed_specs, obstacle_field, config)

    if verbose:
        print(f"\nOptimization complete in {elapsed:.1f}s")
        print(f"  Final cost: {breakdown.total:.1f}")
        print(f"  Reed collisions: {breakdown.reed_collision:.1f}")
        print(f"  Lever collisions: {breakdown.lever_collision:.1f}")
        print(f"  Lever length: {breakdown.lever_length:.1f}")
        print(f"  Hex area: {breakdown.hex_area:.1f}")
        print(f"  Ratio deviation: {breakdown.ratio_deviation:.1f}")
        feasible = sum(1 for lp in lever_paths if lp.is_feasible)
        print(f"  Feasible levers: {feasible}/{len(lever_paths)}")

    return SolverResult(
        x=x_final,
        cost=breakdown.total,
        cost_breakdown=breakdown,
        reed_plates=plates,
        lever_paths=lever_paths,
        history=history,
        elapsed_seconds=elapsed,
        stage1_result=stage1_result,
        stage2_result=stage2_result,
    )
