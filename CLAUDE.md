# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Synthetic (aka ModelBuilder) is a library for generating virtual cell data using ODE models based on biochemical laws common in cancer cell signalling networks. It creates datasets compatible with scikit-learn's `make_regression` format.

## Build and Test Commands

```bash
# Install dependencies (using uv)
uv sync

# Run tests
pytest

# Run a specific test
pytest tests/test_specific_file.py::test_function_name

# Build package (using hatchling)
python -m build
```

## Core Architecture

### Three-Layer Abstraction Pattern

The codebase follows a hierarchical three-layer pattern for model generation:

1. **Spec Layer** (`Specs/`): Network topology and regulation specifications
   - `BaseSpec`: Abstract base defining the contract for all specifications
   - `MichaelisNetworkSpec`: Generic Michaelis-Menten networks with drug mechanisms
   - `DegreeInteractionSpec`: Multi-degree hierarchical drug interaction networks
   - **Purpose**: Define species lists, regulations (including feedback), and drug interactions without concrete reaction details

2. **Model Builder Layer** (`ModelBuilder.py`, `Reaction.py`, `ReactionArchtype.py`):
   - `ReactionArchtype`: Factory defining reaction patterns/templates (mass action, Michaelis-Menten, synthesis, degradation)
   - `Reaction`: Instantiates archtypes with specific species names, parameter values, and initial conditions
   - `ModelBuilder`: Aggregates reactions, compiles state/parameter dictionaries, generates Antimony/SBML
   - **Purpose**: Abstract biochemical patterns away from specific instances

3. **Solver Layer** (`Solver/`): ODE simulation engines
   - `Solver`: Abstract base class with `compile()` and `simulate()` methods
   - `ScipySolver`: Uses `scipy.integrate.odeint` with sympy-based ODE generation
   - `RoadrunnerSolver`: Uses libRoadRunner for SBML/Antimony simulation
   - **Purpose**: Pluggable simulation backends with unified interface

### Key Design Pattern: Archtype-Instantiation Separation

- **Archtypes** (in `ArchtypeCollections.py`) use placeholder species names like `&S`, `&R`, `&I` with patterns like:
  - `&` prefix: Main reactants/products (will be replaced with actual species names)
  - `?` prefix: Extra states used in reverse reactions only
  - Parameters like `Km`, `Vmax`, `Ka`, `Kd` map to biochemical roles
- **Reactions** instantiate archtypes by mapping placeholders to actual species names
- **ModelBuilder** assigns reaction indices (`J0`, `J1`, etc.) to parameters, making them unique

### Regulator-Parameter Automatic Mapping

The system automatically maps regulators (extra_states) to parameters based on naming conventions defined in `Reaction._compute_regulator_parameter_mapping()`:

| Parameter Prefix | Regulator Prefix | Type |
|----------------|------------------|-------|
| `Ka`, `Ks` | `A` | Allosteric stimulator |
| `Kc`, `Kw` | `W` | Weak/additive stimulator |
| `Ki` | `L`, `I` | Allosteric inhibitor |
| `Kic` | `I` | Competitive inhibitor |
| `Kir`, `Kcr` | `I` | Reverse inhibitors |

Use `ModelBuilder.get_regulator_parameter_map()` or `ModelBuilder.get_parameter_regulator_map()` to access these mappings.

### Network Generation Workflow

Typical workflow for generating a synthetic network using `DegreeInteractionSpec`:

```python
# 1. Create specification with hierarchical degrees
# degree_cascades defines number of cascades per degree:
# [1, 2, 5] means: Degree 1 has 1 cascade, Degree 2 has 2, Degree 3 has 5
# Each cascade creates: R{deg}_{idx} -> I{deg}_{idx}
# Degree 1 also connects I{1}_{idx} -> O (outcome)
spec = DegreeInteractionSpec(degree_cascades=[1, 2, 5])
spec.generate_specifications(random_seed=42, feedback_density=0.5)

# 2. Add drugs (optional - must target degree 1 R species only)
drug = Drug(name="DrugX", start_time=500, default_value=0,
             regulation=["R1_1"], regulation_type=["up"])
spec.add_drug(drug, value=10.0)

# 3. Generate model
model = spec.generate_network("multi_degree_network", random_seed=42)

# 4. Compile and use
model.precompile()

# 5. Get Antimony/SBML
antimony_str = model.get_antimony_model()
sbml_str = model.get_sbml_model()

# 6. Simulate with solver (choose one)
# Option A: ScipySolver (fast for batch simulations, accepts Antimony)
from synthetic.Solver.ScipySolver import ScipySolver
solver = ScipySolver()
solver.compile(antimony_str, jit=True)  # Enable JIT for faster execution
results = solver.simulate(start=0, stop=1000, step=100)

# Option B: RoadrunnerSolver (mature, requires SBML format)
from synthetic.Solver.RoadrunnerSolver import RoadrunnerSolver
solver = RoadrunnerSolver()
solver.compile(sbml_str)  # Note: RoadrunnerSolver requires SBML, not Antimony
results = solver.simulate(start=0, stop=1000, step=100)

# Both solvers return pandas DataFrame with 'time' column and species columns
print(results.head())
```

### Drug Mechanism

Drugs are modeled as species that appear at `start_time` via piecewise assignment rules. They regulate target species through:
- **Up-regulation**: Stimulator-like effect (increases reaction rate)
- **Down-regulation**: Inhibitor-like effect (decreases reaction rate)

Drugs are added to the species list as regular species, but their values change only at `start_time`.

### Parameter Randomization Utilities

The `utils/` package provides controlled randomization:
- `parameter_randomizer`: Randomize kinetic parameters with type-based ranges
- `initial_condition_randomizer`: Randomize state initial conditions with pattern-based ranges
- `parameter_mapper`: Map parameters to reactions and states for analysis

These utilities are essential for generating diverse training datasets while keeping parameters biologically plausible.

### Important Naming Conventions

- **Species names ending in 'a'**: Activated forms (e.g., `S1` → `S1a`)
- **Reaction parameters**: Include reaction index (e.g., `Km_J0`, `Vmax_J1`)
- **Linked parameters**: Bypass reaction indexing, use `LinkedParameters` class for shared parameters
- **Extra states**: Regulators that don't participate in mass balance but affect rate laws

### Solver Differences

| Solver | Features | Performance | Use Case |
|---------|------------|--------------|-----------|
| `ScipySolver` | JIT compilation via numba, sympy ODE generation | Fast for many simulations | Batch simulation, parameter sweeps |
| `RoadrunnerSolver` | Full SBML support, libRoadRunner engine | Mature, robust | Single simulations, complex models |

`ScipySolver` handles piecewise assignment rules by detecting change-points and re-integrating segments. `RoadrunnerSolver` handles them natively.

## File Organization

- `src/synthetic/`: Main package
  - `ModelBuilder.py`, `Reaction.py`, `ReactionArchtype.py`: Core model building
  - `Specs/`: Network specifications (BaseSpec, MichaelisNetworkSpec, DegreeInteractionSpec, Drug, Regulation)
  - `Solver/`: ODE solvers (Solver base, ScipySolver, RoadrunnerSolver, AMICISolver)
  - `ArchtypeCollections.py`: Predefined reaction archtypes and factory functions
  - `utils/`: Parameter randomization, mapping, plotting, data generation helpers
  - `SyntheticGenUtils/`: Data processing, simulation, perturbation, validation utilities

## Model Precompilation

Always call `model.precompile()` before accessing parameters/states or generating Antimony. This caches the state and parameter dictionaries and enables parameter/state manipulation via `set_parameter()` and `set_state()`.

## Tuple vs Dict for Values

Throughout the codebase, parameters and values can be specified as:
- **Tuples**: Positional matching to archtype definitions (e.g., `(100.0, 50.0)` for `Vmax, Km`)
- **Dicts**: Name-based matching (e.g., `{'Vmax': 100.0, 'Km': 50.0}`)

Dicts are safer and more readable, especially for reactions with many parameters.
