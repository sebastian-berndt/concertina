# Concertina Compiler

A parametric layout solver for Hayden Duet concertinas with accordion reeds. Given a button grid and reed plate dimensions, it computes the optimal action board layout — reed positions, lever paths, and pivot points — where nothing collides.

## The Problem

A concertina has two boards stacked vertically:

```
        ┌─────────────────────────────┐
        │       ACTION BOARD          │  ← buttons + levers + pivots
        │  [buttons poke through]     │
        │  [levers in slots below]    │
        ├─────────────────────────────┤
        │        REED PAN             │  ← reed plates + pallets
        │  [reed plates laid out]     │
        │  [pallet holes align with   │
        │   lever tips above]         │
        └─────────────────────────────┘
```

The **action board** has button holes on top and lever slots underneath. Each lever pivots on a post, connecting a button to a pallet hole. The **reed pan** sits below, holding the reed plates with pallet valves that the levers open.

The layout puzzle: place 23-29 reed plates on the reed pan and route 23-29 levers through the action board so that:
- No reed plate overlaps another (reed pan layer)
- No lever slot crosses a button hole (action board layer)
- No lever slot crosses another lever slot (action board layer)
- Each lever can reach from its button to its pallet with at most gentle bends

## Physical Collision Model

The action board and reed pan are **separate layers**. Collisions only happen within the same layer:

| Collision | Layer | Constraint |
|-----------|-------|------------|
| Reed plate vs reed plate | Reed pan | Can't overlap |
| Lever vs button hole | Action board | Lever slot can't cross a button hole |
| Lever vs lever | Action board | Lever slots can't cross each other |
| Lever vs pivot post | Action board | Lever can't pass through a pivot pin |
| Lever vs pallet hole | Action board | Lever can't cross another pallet opening |

**Levers do NOT collide with reed plates** — they are on opposite sides of the board.

## The Beaumont Layout

The solver is configured for the R. Morse & Co. Beaumont, a 52-key Hayden Duet:

- **Left hand:** 23 keys (Bb2 to B4), 4 rows
- **Right hand:** 29 keys (C#4 to D6), 6 rows
- **Hayden pattern:** each step right = +2 semitones (whole tone), each row up = +7 semitones at up-right neighbor

Note layout verified against the official R. Morse & Co. Beaumont note chart.

## Architecture

### Module Overview

```
concertina/
  config.py          ─── Configuration dataclasses (all physical params)
  hayden_layout.py   ─── Beaumont 52-key button grid (23 LH + 29 RH)
  reed_specs.py      ─── Reed plate dimensions and geometry
  geometry.py        ─── Fast numpy 2D geometry (SAT, segment distances)
  sector_placer.py   ─── Angular sector assignment + greedy radius search
  lever_router.py    ─── Lever routing with angle-limited doglegs
  greedy_placer.py   ─── Legacy greedy placer (no sector assignment)
  obstacles.py       ─── Shapely collision detection (validation only)
  cost_function.py   ─── Shapely objective function (validation only)
  cost_fast.py       ─── Numpy objective function (for DE optimizer)
  solver.py          ─── Differential evolution wrapper
  visualize.py       ─── Matplotlib 2D rendering
  main.py            ─── CLI entry point
```

### Data Flow

```
  ConcertinaConfig
       │
       ├── InstrumentSpec ──► HaydenLayout.from_beaumont()
       │                           │
       ├── ReedDimensions  ──► generate_reed_table()
       │                           │
       │                    ┌──────┘
       ▼                    ▼
  SectorPlacer ──────► [ReedPlate × N]  (positioned on reed pan)
       │
       ▼
  LeverRouter ──────► [LeverPath × N]   (routed through action board)
       │
       ▼
  Visualization + JSON output
```

## Algorithm

### Step 1: Sector Assignment (Hungarian Algorithm)

Each button has a natural outward angle from the grid center. We create N angular slots spread around 360° and use `scipy.optimize.linear_sum_assignment` to optimally assign buttons to slots, minimizing total angular deviation.

This ensures levers fan out radially in their own lanes, avoiding crossings.

### Step 2: Greedy Radius Search

For each button-slot pair (largest reed first), sweep over radii to find the closest position where:
- The reed plate doesn't overlap any already-placed plate (SAT check)
- The lever path clears all button holes in the action board (segment-to-circle distance)
- The lever is at least 35mm long (prevents steep pallet angle)

The plate is oriented to point its pallet toward the button (`φ = atan2(by-cy, bx-cx)`).

### Step 3: Lever Routing

For each button-pallet pair, the router tries paths in order:

1. **Straight line** — check if the lever slot clears all button holes, pallet holes, and previously placed lever slots
2. **Single gentle bend** (max 30°) — try waypoints around blocking obstacles
3. **Double gentle bend** (max 30° each) — try pairs of waypoints

Each placed lever becomes an obstacle for subsequent levers (incremental routing).

Physical constraints for laser-cut 1.5mm stainless steel levers:
- Maximum 2 bends per lever
- Maximum 30° deviation from straight per bend
- Minimum 35mm total lever length

### Step 4: Pivot Placement

The pivot sits along the lever path at `L / (R + 1)` from the button end, where R is the leverage ratio. For a 2:1 ratio, the pivot is at 1/3 of the lever length.

## Geometry: Shapely vs Numpy

**Shapely is never used in hot paths.** All placement and routing geometry uses pure numpy math in `geometry.py`:

| Operation | Method | Module |
|-----------|--------|--------|
| Reed-reed overlap | Separating Axis Theorem | `geometry.py` |
| Lever-button clearance | Segment-to-circle distance | `geometry.py` |
| Lever-lever clearance | Segment-to-segment distance | `geometry.py` |
| Lever-rectangle clearance | Segment-to-rect distance | `geometry.py` |

Shapely is only used for final validation (`cost_function.py`) and visualization (`visualize.py`).

## Configuration

All parameters are dataclasses with JSON save/load:

```python
from concertina.config import ConcertinaConfig

config = ConcertinaConfig.defaults()
config.ratio.target_ratio = 1.5
config.clearance.static_floor = 1.5
config.save("my_instrument.json")
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| Button diameter | 6.0mm | Beaumont standard |
| Horizontal pitch | 16.5mm | Button center-to-center |
| Button travel | 2.5mm | Holden-style |
| Lever thickness | 1.5mm | 304 stainless steel |
| Target ratio | 2.0:1 | Graduated: bass 2.2, treble 1.8 |
| Min pallet lift | 4.5mm | Hard constraint |
| Static clearance | 1.2mm | Lever to fixed object |
| Dynamic gap | 1.8mm | Between moving levers |
| Min lever length | 35mm | Prevents steep pallet angle |

## Usage

```bash
# Install
pip install -e ".[dev]"

# Generate layout for both sides
python -m concertina.main --plot

# Single side
python -m concertina.main --side RH --plot

# Custom config
python -m concertina.main --config my_config.json --plot --output result.json

# Run tests
pytest tests/ -v
```

## Current Results

### Reed Banks (current approach)

Reeds are grouped onto banks by Hayden row. Large bass reeds stay as
individual plates. Banks stand vertically on the reed pan.

| Side | Banks | Individual | Total Reeds | Placed | Levers Feasible |
|------|-------|-----------|-------------|--------|-----------------|
| LH | 3 | 9 bass | 23 | 23/23 | 5/23 |
| RH | 6 | 0 | 29 | 29/29 | 9/29 |

All 52 reeds fit inside the 200mm hex. Lever feasibility is low because
banks are placed too close to the button grid — levers cross button holes.
Next step: row-aligned bank placement (banks outside the button field).

### Individual Plates (previous approach, no banks)

| Constraint | LH | RH |
|------------|-----|-----|
| No hex boundary | 21/23 feasible | 26/29 feasible |
| 200mm hex | 17/23 feasible | 16/29 feasible |

73 tests passing. All geometry hot paths use numpy (no shapely).

### Reed Dimensions

Three sources supported:
- **BINCI_STANDARD preset**: 52 notes with realistic Italian accordion reed plate dimensions
- **Custom measurements**: pass a dict of `{note: (length, width)}` per note
- **Interpolated fallback**: log-interpolated from bass (54×18mm) to treble (18.5×11mm)

## Next Steps

1. **Row-aligned bank placement** — place each bank outward from its Hayden row so levers go straight out without crossing other buttons
2. Build123d integration for 3D CAD export (STEP files for laser cutting)
3. Optional DE polish stage
