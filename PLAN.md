# Parametric Concertina Compiler -- Implementation Plan

## Context

Build a Python-based "Concertina Compiler" that takes a fixed Hayden Duet button grid and uses global optimization (differential evolution) to find the smallest possible reed pan layout where all mechanical levers route from buttons to pallets without collisions. Based on a detailed design conversation covering a 52-key Beaumont-style instrument with accordion reeds, Holden-style 2:1 leverage, and laser-cut stainless steel levers.

## Project Structure

```
concertina/
  __init__.py
  config.py            # Physical constants and design parameters
  hayden_layout.py     # Button grid generation (Hayden isometric layout)
  reed_specs.py        # Reed plate dimensions and geometry
  geometry.py          # Fast numpy 2D geometry (SAT, line-rect distance)
  greedy_placer.py     # Sequential reed placement (primary strategy)
  obstacles.py         # Shapely-based collision detection (validation only)
  lever_router.py      # Lever pathfinding, numpy-based (straight + dogleg)
  cost_function.py     # Shapely-based objective (validation only)
  cost_fast.py         # Numpy-based objective (for DE optimizer)
  solver.py            # Differential evolution setup and execution
  visualize.py         # Matplotlib 2D layout visualization
  main.py              # CLI entry point
tests/
  test_hayden_layout.py
  test_reed_specs.py
  test_obstacles.py
  test_lever_router.py
  test_cost_function.py
pyproject.toml         # Dependencies: shapely>=2.0, scipy>=1.10, numpy, matplotlib
```

## Key Design Parameters

| Parameter | Value | Source |
|-----------|-------|--------|
| Button diameter | 6.0mm | Beaumont standard |
| Horizontal pitch | 16.5mm | Beaumont standard |
| Vertical offset | 14.3mm (16.5 * sqrt(3)/2) | Hexagonal geometry |
| Button travel | 2.5mm | Holden-style |
| Lever thickness | 1.5mm 304 SS | Laser cut |
| Lever width | 3-5mm | Tapers near button |
| Target ratio | 2.0:1 (graduated: bass 2.2, treble 1.8) | Holden-style |
| Min pallet lift | 4.5mm | Hard constraint |
| Static clearance | 1.2mm | Hard floor |
| Dynamic clearance | 1.8mm | Soft target |
| Pivot buffer | 2.0mm | Around pivot posts |
| Keys per side | 26 | Beaumont |
| Reed sizes | 55x18mm (bass) to 24x15mm (treble) | Log taper |

## Development Strategy

### Mini Problem First
All development uses a **6-button, 6-reed "mini" test fixture** that runs in seconds. Every module gets a mini-problem example before scaling to 26 keys. The mini fixture uses a single row of 6 buttons with 6 small reeds -- enough to exercise straight levers, collisions, and doglegs without waiting for full optimization.

### Tiered Cost Function
Don't enable all 11 penalty terms at once. Build incrementally:
- **Tier 1 (v1):** Reed-reed collision, lever-button collision, lever length (L^2), hex area, ratio deviation
- **Tier 2 (add once Tier 1 converges):** Bends, uniformity, lever-lever proximity, min lever length
- **Tier 3 (refinements):** Pivot accessibility, center of gravity, chamber proportionality

Each tier's weights default to 0 in `CostWeights` until activated. This way the optimizer solves the core problem first.

### Two-Stage Solver (Default Flow)
78D is too large for a cold start. The standard pipeline is:
1. **Stage 1 -- Coarse placement (52D):** Optimize `[r, theta]` per reed. Fix `phi` to "point toward center" (phi = theta + pi). This finds the rough radial arrangement.
2. **Stage 2 -- Refine rotation (78D):** Unlock `phi` and re-optimize with the Stage 1 result as the starting point. This fine-tunes reed plate angles.

### Button-to-Reed Assignment
Each button maps 1:1 to a reed by note name (e.g., "C4" button -> "C4" reed plate). The mapping is determined once from `HaydenLayout.get_all_enabled()` sorted by MIDI number, paired with `reed_table` sorted by MIDI number. This ordering is fixed during optimization -- only reed positions change, not assignments.

## Implementation Phases

### Phase 1: Config + Button Grid
**Files:** `config.py`, `hayden_layout.py`

**config.py -- Configuration System:**

All specs, rules, and constraints should be easily modifiable without touching algorithm code. The config system uses a layered approach:

1. **`InstrumentSpec` dataclass** -- Physical dimensions that define the instrument:
   - `button_diameter`, `h_pitch`, `v_pitch`, `button_travel`
   - `lever_thickness`, `lever_width_min`, `lever_width_max`, `lever_material` (label)
   - `keys_per_side`, `min_lever_length`
   - Ships with a `BEAUMONT_SPEC` preset, but user can create custom specs

2. **`RatioSpec` dataclass** -- Leverage ratio rules:
   - `target_ratio`, `ratio_min`, `ratio_max`
   - `graduated`: bool -- whether to taper ratio across the range
   - `bass_ratio`, `treble_ratio` (used if graduated=True)
   - `grace_zone`: float -- deviation within this range has zero penalty (e.g. 0.05)

3. **`ClearanceSpec` dataclass** -- All clearance rules:
   - `static_floor`, `dynamic_gap`, `pivot_buffer`
   - `min_lever_lever_distance`
   - Easy to tighten/loosen if manufacturing tolerance changes

4. **`CostWeights` dataclass** -- All penalty weights:
   - `w_lever_length`, `w_bend`, `w_ratio_deviation`, `w_uniformity`
   - `w_hex_area`, `w_reed_collision`, `w_lever_collision`
   - `w_pivot_accessibility`, `w_center_of_gravity`, `w_chamber_proportionality`
   - Each weight can be set to 0 to disable a penalty, or cranked up to prioritize it

5. **`SolverBounds` dataclass** -- Optimization bounds:
   - `r_min`, `r_max`, `theta_min`, `theta_max`, `phi_min`, `phi_max`

6. **`ConcertinaConfig` dataclass** -- Top-level container combining all of the above:
   - `instrument: InstrumentSpec`
   - `ratio: RatioSpec`
   - `clearance: ClearanceSpec`
   - `weights: CostWeights`
   - `bounds: SolverBounds`
   - `save(path)` / `load(path)`: serialize to/from TOML or JSON for easy editing
   - `DEFAULTS` class variable with the Beaumont/Holden preset

**Key principle:** Every module receives its relevant config as a parameter rather than importing global constants. This means you can run the solver with different configs (e.g. compare 1:1 vs 2:1 ratio, or 34-key vs 52-key) without modifying any code.

Example usage:
```python
# Use defaults
config = ConcertinaConfig.defaults()

# Or customize
config = ConcertinaConfig.defaults()
config.ratio.target_ratio = 1.5
config.clearance.static_floor = 1.5
config.weights.w_bend = 100  # prioritize straight levers

# Or load from file
config = ConcertinaConfig.load("my_concertina.toml")
```

**hayden_layout.py:**
- `HaydenButton` dataclass: note, row, col, center (x,y), enabled flag
- `HaydenRow` class: groups buttons by row
- `HaydenLayout` class:
  - `__init__(self, instrument: InstrumentSpec)` -- takes pitch from config, not hardcoded
  - `from_beaumont(side, instrument)` classmethod to generate 26-key LH or RH layout
  - Hayden coordinate math: `x = (col + 0.5 * row) * instrument.h_pitch`, `y = row * instrument.v_pitch`
  - `get_all_enabled()` returns flat list of active buttons
  - The note-to-position mapping is stored as data (list of tuples), easily editable to add/remove buttons or create a different layout (e.g. 34-key Elise, 46-key custom)
- **Test:** Verify 26 buttons generated, correct spacing, no overlaps, plot to confirm

### Phase 2: Reed Plate Geometry
**File:** `reed_specs.py`

- `ReedSpec` dataclass: note, midi, length, width, target_ratio
- `generate_reed_table(notes, ratio_spec)` function:
  - Ratios interpolated linearly from `ratio_spec.bass_ratio` to `ratio_spec.treble_ratio`
  - Dimensions interpolated log-linearly from bass (55x18mm) to treble (24x15mm)
  - Bass/treble dimensions can be overridden (e.g. when user measures actual reed plates)
  - Returns list of `ReedSpec` -- not a module-level constant, so different reed sets can be used
- `STANDARD_ACCORDION_REEDS`: default dimensions dict, user can replace with measured values
- `ReedPlate` class: takes `ReedSpec` + polar coords (r, theta, phi)
  - `get_polygon(clearance)`: rotated rectangle as shapely Polygon
  - `pallet_position`: point at `0.2 * length` from the tip of the plate (near edge)
  - `center`: polar-to-cartesian conversion
- **Test:** Verify polygon area, rotation, pallet position on correct edge

### Phase 3: Visualization + Manual Placement Mode
**File:** `visualize.py`

- `plot_layout()`: Draw buttons (blue circles), reed plates (green rectangles), pallet positions (red dots), convex hull (dashed gray)
- Build early so every subsequent module can be visually debugged
- **Manual placement mode:** A script/function where the user hand-places reeds by specifying (r, theta, phi) per reed, then visualizes the result and sees the cost breakdown. This builds intuition about the problem space before trusting the optimizer, and acts as a smoke test for every module.
  ```python
  # Example: hand_place.py
  layout = HaydenLayout.from_beaumont("RH", config.instrument)
  reeds = generate_reed_table(notes, config.ratio)
  # Hand-place 6 reeds for testing
  placements = [(60, 0.5, 0.0), (65, 1.0, 0.2), ...]
  plot_layout(layout, placements, reeds, config)
  ```
- **Test:** Visual inspection of hand-placed configurations

### Phase 3.5: Mini Test Fixture
**File:** `tests/conftest.py` or `tests/mini_fixture.py`

- Define a reusable 6-button, 6-reed mini problem:
  - 6 buttons in a single row (C4, D4, E4, F4, G4, A4)
  - 6 corresponding reed plates with proportional sizes
  - Known-good hand placement where all levers are straight (for baseline testing)
  - Known-bad placement with forced collisions (for penalty testing)
- Every subsequent test module imports and uses this fixture
- **Test:** Run `plot_layout()` on the fixture to verify it looks reasonable

### Phase 4: Obstacle Field + Collision Detection
**File:** `obstacles.py`

- `ObstacleField` class: manages all no-go zones
  - Button holes: circle radius = `BUTTON_RADIUS + STATIC_FLOOR_CLEARANCE` (4.2mm)
  - Pivot posts: circle radius = `PIVOT_BUFFER` (2.0mm)
  - Reed plates: polygon buffered by `STATIC_FLOOR_CLEARANCE`
- `get_merged_obstacle()`: shapely union for fast intersection tests
- `check_reed_reed_collisions()`: pairwise reed plate overlap detection
- `check_lever_collisions()`: lever path vs all obstacles
- **Test:** Known intersecting/non-intersecting geometries

### Phase 5: Lever Router
**File:** `lever_router.py`

- `LeverPath` dataclass: button_pos, pallet_pos, pivot_pos, path (LineString), segments count, total_length, actual_ratio, is_feasible
- `LeverRouter` class:
  - `route(button, pallet, target_ratio, lever_index)` -> `LeverPath`
  - `_compute_pivot()`: walk along path from button end, pivot at distance `L / (R + 1)` where R = ratio. For 2:1, pivot is at 1/3 of lever length from button.
- Lever width: buffer centerline LineString by `lever_width / 2` before intersection tests
- Minimum lever length: reject paths shorter than 30mm (steep pallet angle causes poor seal)

**v1 -- Simple router (build first):**
  - `_try_straight()`: straight line from button to pallet, check if buffered line clears obstacles
  - `_try_single_dogleg()`: if straight fails, try a single-bend dogleg. For each blocking obstacle circle, compute the two tangent points from button and from pallet, test the 2-segment path through each. Pick the shortest valid path.
  - If no single dogleg works, mark lever as infeasible (the optimizer will avoid this configuration)

**v2 -- Full visibility graph router (upgrade later):**
  - `_build_visibility_graph()`: tangent points on all obstacle circles as graph nodes
  - `_dijkstra()`: shortest path through visibility graph
  - Handles multi-obstacle doglegs and complex routing
  - Only needed once the v1 router proves insufficient for dense layouts

- **Test:** Straight lever with no obstacles; single-dogleg around one obstacle; infeasible case

### Phase 6: Cost Function
**File:** `cost_function.py`

- `evaluate(x, layout, reed_specs, config)` -> float (for scipy)
- `evaluate_detailed(x, layout, reed_specs, config)` -> `CostBreakdown` dataclass
- All penalty weights come from `config.weights`, all clearances from `config.clearance`, all ratio rules from `config.ratio` -- nothing hardcoded
- State vector X: flat array of 78 values = 26 reeds * [r, theta, phi]
- `_decode_state(x, reed_specs)`: unpack flat array into list of `ReedPlate` objects
- Penalty terms (organized by tier -- see Development Strategy):

  **Tier 1 (v1 -- enable from the start):**
  - **Reed-reed collision:** `overlap_area * w_reed_collision` (hard constraint, w=1e6)
  - **Lever-button collision:** `intersection_length * w_lever_collision` (hard constraint, w=1e6)
  - **Lever length:** `sum(L^2) * w_len` -- squared penalizes long levers much more
  - **Hex area:** `convex_hull_area * w_hex_area` -- pulls reeds toward center
  - **Ratio deviation:** `(actual - target)^2 * w_ratio_dev` per lever, target graduated per reed spec

  **Tier 2 (enable once Tier 1 produces valid layouts):**
  - **Bends:** `num_bends * w_bend` per dogleg
  - **Uniformity:** compare ratios of Hayden-grid-adjacent buttons. Penalty = `sum((ratio_i - ratio_neighbor)^2) * w_uniformity`. Ensures consistent feel across neighboring keys.
  - **Lever-lever proximity:** penalty if distance < `config.clearance.min_lever_lever_distance`
  - **Minimum lever length:** hard penalty if total length < `config.instrument.min_lever_length`

  **Tier 3 (refinements for production-quality layouts):**
  - **Pivot accessibility:** penalty if a pivot post sits directly under another lever
  - **Center of gravity:** `distance(centroid_of_reed_mass, origin)^2 * w_cog`
  - **Chamber proportionality:** penalty if air space is not proportional to reed plate area

- All weights live in `CostWeights` dataclass. Tier 2/3 weights default to 0 and are turned on explicitly.
- Hard constraints encoded as massive penalties (proportional to violation magnitude)
- **Test:** Known-good config has finite cost; overlapping reeds dominated by collision penalty

### Phase 7: Solver
**File:** `solver.py`

- `SolverConfig` dataclass: strategy="best1bin", maxiter=200, popsize=20, tol=1e-4, mutation=(0.5, 1.0), recombination=0.7, polish=True, workers=-1, seed=42
- `SolverResult` dataclass: x, cost, cost_breakdown, reed_plates, lever_paths, elapsed_seconds
- `solve(layout, reed_specs, config)` -> `SolverResult`
- `_make_bounds(n_reeds, stage)`: generate bounds -- Stage 1 has 2 params/reed (r, theta), Stage 2 has 3 (r, theta, phi)
- `_make_initial_guess()`: hybrid arrangement -- treble reeds radially at closer radius, bass at wider radius in block-style positions. Phi defaults to "point toward center" (phi = theta + pi).

**Default two-stage pipeline:**
1. **Stage 1 -- Coarse (52D for 26 reeds):** Optimize `[r, theta]` only. Phi fixed to "point at center." Uses `differential_evolution` with popsize=15, maxiter=300. Finds rough placement.
2. **Stage 2 -- Refine (78D):** Unlock phi. Seed with Stage 1 result. Uses `differential_evolution` with popsize=10, maxiter=200, tighter bounds around Stage 1 solution. Fine-tunes rotation.
3. `polish=True` on Stage 2 runs a local Nelder-Mead cleanup at the end.

- `workers=-1` for parallel evaluation on all CPU cores
- Callback for progress logging (cost per generation) and checkpoint saving
- **Test:** Run both stages on the 6-reed mini fixture to verify convergence

### Phase 8: CLI + Full Visualization
**Files:** `main.py`, `visualize.py` (complete)

- CLI args: `--side LH/RH/both`, `--maxiter`, `--popsize`, `--workers`, `--seed`, `--output`, `--plot`, `--config path/to/config.toml`
- Default config used if `--config` not specified; any CLI arg overrides the config file value
- Output: JSON with optimal coordinates for all reeds, pivots, lever paths, plus the config used (for reproducibility)
- Full visualization: lever paths (green=straight, orange=dogleg, red=infeasible), pivot points (black crosses), obstacle zones (transparent red)
- `plot_convergence()`: cost vs generation number
- `plot_cost_breakdown()`: bar chart of cost components

## Build Order

Each phase is independently testable before moving to the next. The 6-reed mini fixture is used from Phase 3.5 onward.

| # | Module | Dependencies | Test Method |
|---|--------|-------------|-------------|
| 1 | `config.py` | none | import, verify defaults |
| 2 | `hayden_layout.py` | config | unit test grid geometry + visual plot |
| 3 | `reed_specs.py` | config, numpy | unit test dimensions + polygon generation |
| 3.5 | mini fixture | config, hayden_layout, reed_specs | 6-button/6-reed baseline for all later tests |
| 4 | `visualize.py` (partial) | hayden_layout, reed_specs | plot mini fixture + hand-placed reeds |
| 5 | `obstacles.py` | hayden_layout, reed_specs, shapely | unit test collisions on mini fixture |
| 6 | `lever_router.py` (v1) | obstacles, shapely | straight + single-dogleg on mini fixture |
| 7 | `cost_function.py` (Tier 1) | all above | cost of mini fixture known-good vs known-bad |
| 8 | `solver.py` (two-stage) | cost_function, scipy | solve mini fixture, verify convergence |
| 9 | Scale to 26 keys | all | run full Beaumont side, tune weights |
| 10 | `cost_function.py` (Tier 2) | all | add bends, uniformity, lever proximity |
| 11 | `main.py` + `visualize.py` (full) | all | end-to-end CLI |
| 12 | `lever_router.py` (v2) | obstacles, shapely | visibility graph + Dijkstra (if needed) |
| 13 | `cost_function.py` (Tier 3) | all | pivot access, CoG, chamber proportion |

## Completed (as of 2026-04-20)

Phases 1-8 are implemented with 58 passing tests. All modules exist and work end-to-end on the 6-reed mini fixture. Full 26-key runs produce plausible layouts but need further tuning.

**Files implemented:**
- `config.py`, `hayden_layout.py` (Beaumont 23+29 keys), `reed_specs.py`, `geometry.py`
- `sector_placer.py` (primary), `greedy_placer.py` (legacy), `lever_router.py` (v2 with angle limits)
- `cost_function.py`, `cost_fast.py`, `obstacles.py` (all validation only)
- `solver.py`, `visualize.py`, `main.py`
- Tests: 73 passing, all modules covered
- `tests/mini_fixture.py`

**Shapely boundary:** Shapely is only used in `cost_function.py` (validation), `obstacles.py` (validation), and `visualize.py` (plotting). All hot paths use `geometry.py` (numpy).

**Current results:** LH 21/23, RH 26/29 feasible with physically realistic levers (max 2 bends, max 30° each).

## Learnings from First Full-Scale Runs

### 1. Correct physical collision model
- **The action board and reed pan are separate layers.** Levers (action board) do NOT collide with reed plates (reed pan). They're on opposite sides.
- **Lever obstacles are:** other button holes, other lever slots, pivot posts, pallet holes.
- **Reed obstacles are:** other reed plates only.
- Getting this wrong (treating reed plates as lever obstacles) made the problem much harder than it actually is. RH went from 19/29 to 26/29 feasible just by fixing this.

### 2. Shapely is too slow for ANY hot path
- Shapely cost function: ~154ms/eval. Numpy (`geometry.py`): ~3ms/eval (46x faster).
- All placement and routing uses numpy. Shapely only for final validation and visualization.

### 3. Sector-based placement beats global optimization
- DE on 52D doesn't converge in reasonable time.
- Sector assignment (Hungarian algorithm) + greedy radius search produces good layouts in ~6 seconds.
- The key insight: assign each button an angular lane so levers fan out radially without crossing. Then only the radius is a free variable per reed.

### 4. Physical lever constraints
- Max 2 bends per laser-cut lever, max 30° per bend.
- Minimum lever length 35mm (prevents steep pallet angle).
- The router rejects any path that violates these — earlier versions allowed unrealistic zigzag paths.

### 5. Interleaved routing matters
- Each placed lever becomes an obstacle for subsequent levers (lever-lever collision).
- The router processes levers incrementally, building up the obstacle set.
- Placement order (largest reed first) still works well.

### 6. The Beaumont is 23+29, not 26+26
- R. Morse Beaumont: 23 LH (Bb2–B4), 29 RH (C#4–D6).
- Verified against the official note chart. The Hayden pattern uses whole-tone rows with +7 semitone row jumps.

## What to Do Next

### Priority 1: Bank Placement for Lever Routing ← CURRENT FOCUS

Reed banks are implemented and all 52 reeds pack inside the 200mm hex.
But lever routing is poor (LH 5/23, RH 9/29) because banks are placed
too close to the button grid — levers cross through other button holes.

**The fix: row-aligned bank placement.**
Each bank groups a Hayden row. Place each bank directly outward from
its row, outside the button field. Levers go straight outward in
parallel without crossing other rows' buttons.

```
  Current (bad):                Better (row-aligned):

       bank somewhere              ● ● ● ●        ← other rows
      ╔═══════╗                   ● ● ● ● ●
      ║       ║                    ● ● ● ● ● ●
      ╚═══════╝                     B1 B2 B3 B4    ← this row's buttons
     ╲  ╲│╱  ╱ cross                 │  │  │  │    ← parallel levers
      ● ● ● ●                      ╔════════════╗
      ● ● ● ●                      ║   bank     ║  ← outside button field
                                    ╚════════════╝
```

Implementation steps:
1. Compute each Hayden row's average position and outward direction
2. Place bank at r = max_button_radius + margin, aligned with its row
3. Score by lever feasibility (count how many levers clear button holes)
4. Enforce hex boundary and bank-bank non-overlap
5. Individual bass plates fill remaining edge space

### Priority 1b: Reed Banks (DONE)

Reed banks implemented and working. Details below for reference.

#### The Physical Reality

```
  Individual plates (current):     Reed bank:
  ┌──┐ ┌──┐ ┌──┐ ┌──┐             ┌──────────────────────┐
  │R1│ │R2│ │R3│ │R4│             │ R1 │ R2 │ R3 │ R4   │
  └──┘ └──┘ └──┘ └──┘             └──────────────────────┘
  29 objects to pack               4-6 banks to pack
```

An accordion reed bank is a long rectangular block with multiple reed slots cut into it. Reeds are arranged by pitch (low to high along the bank). Each slot has its own pallet hole. The bank mounts as a single unit onto the reed pan.

#### 3D Clearance Problem

When the bellows closes, the two reed pans face each other with limited vertical clearance:

```
  Side view (bellows closed):
  ┌──────── Action Board LH ────────┐
  │  buttons  │  levers  │          │
  ├───────────┴──────────┴──────────┤
  │         Reed Pan LH             │
  │  ┌─flat─┐  ┌─stand─┐  ┌─flat─┐ │  ← reed banks
  │  └──────┘  │  ing  │  └──────┘ │
  │            └───────┘            │
  ╞═══════ bellows fold ════════════╡  ← minimum clearance here
  │            ┌───────┐            │
  │  ┌─flat─┐  │  ing  │  ┌─flat─┐ │
  │  └──────┘  │  stand│  └──────┘ │
  │         Reed Pan RH             │
  ├───────────┬──────────┬──────────┤
  │  buttons  │  levers  │          │
  └──────── Action Board RH ────────┘
```

**Key constraint:** Reed banks can be oriented two ways:
- **Flat (lying down):** Low profile, fits near the edges where clearance is tight. But takes more surface area.
- **Standing (on edge):** Compact surface footprint, but tall. Only fits in the CENTER of the reed pan where there's enough vertical clearance between the two sides.

**The center zone** (roughly inner 60% of the hex) has enough depth for standing banks. The **edge zone** (outer 40%) is too shallow — only flat reeds fit there.

#### Reed Chamber Sizing

Each reed slot needs an air chamber around it. Chambers vary in size:
- Bass reeds: large chamber (more air volume for low frequencies)
- Treble reeds: small chamber (less air needed)

On a reed bank, the chambers are built into the bank or into the reed pan walls around it. The bank width is determined by the widest reed slot on it. This means:

**Risk: Mixed banks waste space.** A bank with one bass reed and five treble reeds is as wide as the bass reed for its entire length. Better to group similar-sized reeds together.

#### Design Decisions (Settled)

1. **Reed grouping: by Hayden row/column.**
   Reeds in the same Hayden row share a whole-tone scale and their buttons
   are in a line. Grouping by row means levers run roughly parallel, which
   minimizes crossing. Each row becomes one bank (or part of one).

2. **Not all reeds on banks.**
   Large bass reeds stay as individual plates — they're too varied in size
   and too few to bank efficiently. Mid and treble reeds go on banks.
   This hybrid approach packs the many small/medium reeds densely while
   giving the large bass reeds placement flexibility.

3. **Bank orientation is a clearance decision, not mechanical.**
   The pallet mechanism is the same for flat and standing. The choice is
   driven by where the bank sits on the reed pan:
   - **Center zone:** bellows has clearance → standing banks OK (compact footprint)
   - **Edge zone:** bellows closes tight → flat or individual only

4. **Bank count is configurable.**
   Not hardcoded. The solver tries different groupings. Typical: 2-4 banks
   per side for mid/treble, plus 4-8 individual bass plates.

5. **Pallet position within a bank:**
   Each reed on the bank has its own pallet hole along the bank's long edge.
   Pallet spacing is fixed by reed slot sizes. The lever must reach the
   specific pallet hole for its reed.

#### Implementation Plan

**Phase 1: Data model**

```python
@dataclass
class ReedBank:
    """A group of reeds mounted on a single block."""
    reeds: list[ReedSpec]           # reeds on this bank, sorted by pitch
    orientation: str                # "flat" or "standing"
    wall_thickness: float = 2.0     # mm between reed slots
    
    @property
    def length(self) -> float:
        """Sum of reed slot lengths + walls."""
        return sum(r.length for r in self.reeds) + self.wall_thickness * (len(self.reeds) + 1)
    
    @property 
    def width(self) -> float:
        """Widest reed + chamber walls."""
        return max(r.width for r in self.reeds) + 2 * self.wall_thickness
    
    @property
    def height(self) -> float:
        """Standing: width becomes height. Flat: just reed thickness."""
        if self.orientation == "standing":
            return self.width
        return 8.0  # mm, typical flat block thickness

    def pallet_positions(self, bank_cx, bank_cy, bank_phi) -> list[tuple[float, float]]:
        """Compute pallet hole (x,y) for each reed along the bank edge."""
        ...

@dataclass  
class ReedPanLayout:
    """Mixed layout: banks + individual plates."""
    banks: list[tuple[ReedBank, float, float, float]]  # (bank, cx, cy, phi)
    individuals: list[ReedPlate]                        # individual plates
```

**Phase 2: Reed assignment**
- Input: 23 or 29 ReedSpecs
- Split into bank candidates and individual candidates:
  - Individual: largest N reeds (bass), configurable threshold
  - Banks: remaining reeds, grouped by Hayden row
- Each Hayden row's reeds become one bank (or split into sub-banks if too long)
- Standing banks for center zone, flat for edge zone

**Phase 3: Placement (much simpler packing)**
- Place individual bass plates first (they need the most space, go at edges)
- Then place 2-4 banks (much simpler than 20+ individual plates)
- Banks in center zone can stand (small footprint)
- Total objects to pack: ~6-10 instead of 23-29

**Phase 4: Lever routing**
- Each lever goes from its button to the specific pallet hole on its bank
- Pallet positions computed from bank position + reed offset within bank
- Same collision model: levers vs buttons, levers vs levers
- The parallel levers from a single bank's row will naturally fan out together

#### Risks

1. **Bank rigidity reduces flexibility.** Banks lock multiple reeds into relative positions. Mitigated by keeping large bass reeds as individuals — they need the most placement flexibility.

2. **Pallet hole spacing is fixed within a bank.** Once reeds are grouped, lever endpoints are determined. Mitigated by grouping Hayden rows — levers from the same row naturally fan out in parallel, so fixed spacing works well.

3. **Center zone radius needs a new config parameter.** The bellows fold depth determines where standing banks can go. If this is wrong, banks won't fit when the bellows closes. Need `bellows_clearance_radius` in config.

4. **Bank width = widest reed on it.** Grouping by Hayden row means reeds in a whole-tone scale — their widths are similar but not identical. Waste is small (1-2mm) for row-grouped banks.

5. **Splitting rows into sub-banks.** A full Hayden row (6-7 reeds) might produce a bank that's too long for the hex. May need to split rows into 2 sub-banks. The grouping algorithm should handle this.

### Priority 2: Build123d CAD export
Generate 3D parts from the 2D layout.

### Priority 3: Optional DE polish
Use the sector placement as `x0` for a short DE run to fine-tune reed positions.

## Completed Milestones

| Date | Milestone | Result |
|------|-----------|--------|
| 2026-04-19 | Initial solver | 58 tests, mini fixture converges |
| 2026-04-19 | Fast numpy geometry | 46x speedup, no shapely in hot paths |
| 2026-04-19 | Greedy placer | 10/26 feasible |
| 2026-04-19 | Sector assignment | 19/26 feasible |
| 2026-04-19 | Interleaved routing | 22/26 feasible |
| 2026-04-19 | v2 visibility graph | 26/26 feasible (unrestricted bends) |
| 2026-04-20 | Correct Beaumont notes | 23 LH + 29 RH verified against chart |
| 2026-04-20 | Angle-limited router | Max 2 bends, 30° each |
| 2026-04-20 | Correct obstacle model | Levers vs buttons+levers, not reed plates |
| 2026-04-20 | Current best (no hex) | LH 21/23, RH 26/29 |
| 2026-04-20 | Plate rotation search | Tangential packing, 7 phi angles per position |
| 2026-04-20 | Hex boundary 200mm | Hard constraint, all plates must fit inside |
| 2026-04-20 | Force-directed / L-BFGS-B | Attempted, doesn't converge (non-convex) |
| 2026-04-20 | Current best (200mm hex) | LH 17/23, RH 16/29 — packing needs work |
| 2026-04-21 | Measured reed dimensions | BINCI_STANDARD preset, per-note override support |
| 2026-04-21 | Reed bank model | ReedBank, PlacedBank, assign_banks() by Hayden row |
| 2026-04-21 | Bank placement | 52/52 reeds placed in 200mm hex (banks + individuals) |
| 2026-04-21 | Lever routing with banks | LH 5/23, RH 9/29 — banks too close to button grid |

## Future: Build123d Integration

Once the 2D solver produces optimal coordinates, a separate `cad_export.py` module will:
- Generate 3D lever shapes (extrude the 2D lever paths to 1.5mm thickness)
- Cut button holes and pivot holes in the action board
- Generate the reed pan with pallet holes and chamber walls
- Export STEP files for laser cutting and CNC
