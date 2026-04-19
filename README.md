# Concertina Compiler

A parametric layout solver for Hayden Duet concertinas with accordion reeds. Given a button grid and reed plate dimensions, it computes the optimal reed pan layout where all mechanical levers route from buttons to pallets without collisions.

## The Problem

A concertina action board is a mechanical puzzle. Each button connects to a reed via a lever that pivots on a post. The lever lifts a pallet (valve) to let air reach the reed. With 26 keys per side, you need 26 levers, 26 pivots, and 26 reed plates all packed into a hexagonal frame — and nothing can touch anything else.

```
  Button ──── Pivot ──────────── Pallet ──── Reed
    │          (1/3)              (2/3)        │
    ▼                                          ▼
  [Action Board]                          [Reed Pan]
```

Traditional makers do this by hand in 2D CAD. This tool does it mathematically.

## Architecture

The solver treats the problem as a constrained optimization: button positions are fixed, reed plate positions are free variables, and the cost function penalizes collisions, long levers, and wasted space.

### Module Overview

```
concertina/
  config.py          ─── Configuration dataclasses (all physical params)
  hayden_layout.py   ─── Button grid generation
  reed_specs.py      ─── Reed plate dimensions and geometry
  obstacles.py       ─── Collision detection (no-go zones)
  lever_router.py    ─── Lever pathfinding (straight + dogleg)
  cost_function.py   ─── Shapely-based objective (accurate, for validation)
  cost_fast.py       ─── Numpy-based objective (fast, for optimization)
  solver.py          ─── Differential evolution wrapper
  visualize.py       ─── Matplotlib 2D rendering
  main.py            ─── CLI entry point
```

### Data Flow

```
  InstrumentSpec          ReedDimensions
       │                       │
       ▼                       ▼
  HaydenLayout ──────► generate_reed_table() ──► [ReedSpec × 26]
       │                                              │
       │              ┌───────────────────────────────┘
       │              │
       ▼              ▼
  Greedy Placer / DE Solver ──► [ReedPlate × 26]  (positioned)
       │                              │
       ▼                              ▼
  ObstacleField ◄────────────── get_polygon()
       │
       ▼
  LeverRouter ──► [LeverPath × 26]
       │
       ▼
  CostBreakdown ──► Visualization / JSON output
```

## Phases of the Algorithm

### Phase 1: Define the Button Grid

The Hayden layout is isomorphic — every key has the same geometric relationship to its neighbors. Buttons sit on a hexagonal grid:

- **Horizontal pitch:** 16.5mm center-to-center
- **Vertical offset:** 14.3mm (pitch × √3/2)
- **Rows are staggered** by half a pitch

```python
layout = HaydenLayout.from_beaumont("RH")  # 26 keys, right hand
```

Each button has a fixed `(x, y)` position. These never move during optimization.

### Phase 2: Define the Reed Plates

Accordion reed plates are rectangles that taper from bass to treble:

| Register | Length | Width | Lever Ratio |
|----------|--------|-------|-------------|
| Bass     | 55mm   | 18mm  | 2.2:1       |
| Mid      | ~37mm  | ~16mm | 2.0:1       |
| Treble   | 24mm   | 15mm  | 1.8:1       |

Dimensions are log-interpolated, ratios linearly interpolated. Each reed plate is positioned in polar coordinates `(r, θ, φ)` — distance from center, angle, and plate rotation.

### Phase 3: Place the Reeds (Greedy Strategy)

Pure differential evolution on 52 dimensions doesn't converge in reasonable time. Instead, we place reeds one at a time using a greedy algorithm:

1. **Sort reeds by size** (largest/bass first — they need the most space)
2. **For each reed**, sweep over candidate positions `(r, θ)`:
   - Check that the reed plate doesn't overlap any already-placed plate
   - Check that the lever path (button → pallet) doesn't cross any placed plate
   - Enforce minimum lever length (35mm)
   - Point the plate toward its button (`φ = atan2(by-cy, bx-cx)`)
3. **Pick the position with the shortest lever**

This produces a collision-free layout for ~14/26 reeds. The remaining reeds fall back to collision-free placement without lever clearance guarantees.

### Phase 4: Route the Levers

For each button-pallet pair, the router finds a physical path for the lever:

**v1 (current):**
1. Try a **straight line** from button to pallet
2. If it intersects an obstacle (another reed plate), try a **single-bend dogleg** via tangent waypoints around the blocking obstacle
3. If no clear path exists, mark as **infeasible**

The lever has physical width (3mm min), so collision checks buffer the centerline by half the width.

**Pivot placement:** For a 2:1 ratio, the pivot sits at 1/3 of the lever length from the button end. This means 2.5mm of button travel produces 5mm of pallet lift.

### Phase 5: Evaluate the Cost

The cost function judges a layout. Lower is better.

**Tier 1 (always active):**

| Penalty | Formula | Weight | Purpose |
|---------|---------|--------|---------|
| Reed overlap | `overlap_area × w` | 1e6 | No overlapping plates |
| Lever collision | `count × w` | 1e6 | No blocked levers |
| Lever length | `Σ(L²) × w` | 0.1 | Prefer short levers |
| Hex area | `hull_area × w` | 0.01 | Compact instrument |
| Ratio deviation | `Σ(R-target)² × w` | 200 | Consistent feel |

**Tier 2 (activate later):** bend count, neighbor ratio uniformity, lever-lever proximity, minimum length.

**Tier 3 (production polish):** pivot accessibility, center of gravity balance, chamber proportionality.

Two implementations exist:
- **`cost_function.py`** — Shapely-based, exact geometry, ~154ms/eval. Used for final validation.
- **`cost_fast.py`** — Numpy-based, approximate geometry, ~3.4ms/eval (46× faster). Used during optimization.

### Phase 6: Optimize (Optional Polish)

After greedy placement, differential evolution can fine-tune positions:

1. **Stage 1 (52D):** Optimize `[r, θ]` per reed, φ fixed to point-at-center
2. **Stage 2 (78D):** Unlock φ, seed with Stage 1 result

This is optional — the greedy result is often good enough for initial prototyping.

## Physical Constraints

### What's Fixed
- **Button positions** — determined by the Hayden layout and player ergonomics
- **Reed plate dimensions** — determined by the reeds you bought
- **Lever ratio** — 2:1 target (graduated 2.2 bass → 1.8 treble)

### What's Flexible
- **Reed plate positions** — anywhere around the button field
- **Reed plate rotation** — any angle
- **Lever path** — straight or dogleg (bent)
- **Reed pan shape** — emergent from the optimization (convex hull of all plates)

### Clearances

| Type | Distance | Description |
|------|----------|-------------|
| Static floor | 1.2mm | Lever to any fixed object (hard constraint) |
| Dynamic gap | 1.8mm | Between moving levers (soft target) |
| Pivot buffer | 2.0mm | Around pivot posts |

### Key Insight: Levers Pass Under Buttons

Levers travel through slots in the action board, underneath the button holes. They do **not** need to route around buttons — only around other reed plates and pivot posts. This is critical for the obstacle model.

## Configuration

All parameters are configurable via dataclasses with JSON save/load:

```python
from concertina.config import ConcertinaConfig

# Use Beaumont/Holden defaults
config = ConcertinaConfig.defaults()

# Customize
config.ratio.target_ratio = 1.5
config.clearance.static_floor = 1.5
config.weights.w_bend = 50.0  # activate bend penalty

# Save/load
config.save("my_instrument.json")
config = ConcertinaConfig.load("my_instrument.json")
```

## Usage

```bash
# Install
pip install -e ".[dev]"

# Run solver on right hand side
python -m concertina.main --side RH --maxiter 200 --plot

# Run with custom config
python -m concertina.main --side both --config my_config.json --plot --output result.json

# Run tests
pytest tests/ -v
```

## Project Status

- **Phases 1-8:** Implemented, 58 tests passing
- **Mini fixture (6 reeds):** Solver converges to collision-free layout in ~55s
- **Full scale (26 reeds):** Greedy placer produces 14/26 feasible levers, needs v2 router for remaining doglegs

### Next Steps

1. Separate obstacle sets for placement vs routing (button holes aren't lever obstacles)
2. Extract greedy placer into its own module
3. Tighter collision checks in the fast cost function (OBB instead of bounding circles)
4. v2 visibility graph router for multi-bend doglegs
5. Build123d integration for 3D CAD export (STEP files for laser cutting)
