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

## Python Dependencies

| Package | Purpose |
|---------|---------|
| `antimony>=2.15.0` | SBML/Antimony model format conversion |
| `joblib>=1.5.0` | Parallel processing for batch simulations |
| `matplotlib>=3.10.3` | Plotting and visualization |
| `numba>=0.61.2` | JIT compilation for performance optimization |
| `numpy>=2.2.5` | Numerical operations and array handling |
| `pandas>=2.2.3` | Data structure for simulation results and datasets |
| `python-dotenv>=1.1.0` | Environment variable management |
| `pyyaml>=6.0.2` | YAML configuration file parsing |
| `libroadrunner>=2.3.0` | libRoadRunner SBML simulation engine |
| `scipy>=1.15.3` | Scientific computing, ODE integration (`odeint`) |
| `sympy>=1.14.0` | Symbolic mathematics for ODE generation |
| `tqdm>=4.67.1` | Progress bars for long-running operations |

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

## Kinetic Parameter Tuning

The `utils.kinetic_tuner` module provides `KineticParameterTuner` for generating kinetic parameters that ensure robust signal propagation through hierarchical networks. This tuner solves full nonlinear kinetic equations to achieve target active percentages.

### Using KineticParameterTuner

```python
from synthetic.Specs.DegreeInteractionSpec import DegreeInteractionSpec
from synthetic.Specs.Drug import Drug
from synthetic.utils.kinetic_tuner import KineticParameterTuner

# 1. Create and generate specification
spec = DegreeInteractionSpec(degree_cascades=[3, 6, 15, 25])
spec.generate_specifications(random_seed=42, feedback_density=0.5)

# 2. Add drugs (optional - must target degree 1 R species)
drug_d = Drug(
    name="D",
    start_time=5000.0,
    default_value=100.0,
    regulation=["R1_1", "R1_2", "R1_3"],
    regulation_type=["down", "down", "down"],
)
spec.add_drug(drug_d)

# 3. Generate model with initial value ranges
model = spec.generate_network(
    network_name="MultiDegree_Network",
    mean_range_species=(50, 150),      # Initial concentrations
    rangeScale_params=(0.8, 1.2),       # ±20% variation
    rangeMultiplier_params=(0.9, 1.1),    # Small additional variation
    random_seed=42,
)

# 4. Precompile model (required before tuning)
model.precompile()

# 5. Tune kinetic parameters for target active percentages
tuner = KineticParameterTuner(model, random_seed=42)
updated_params = tuner.generate_parameters(
    active_percentage_range=(0.3, 0.7),  # Target 30-70% active states
    X_total_multiplier=5.0,                # Km_b = X_total × 5.0
    ki_val=100.0,                           # Constant Ki for all inhibitors
    v_max_f_random_range=(5.0, 10.0),     # Total forward Vmax range
)

# 6. Apply tuned parameters to model
for param_name, value in updated_params.items():
    model.set_parameter(param_name, value)

# 7. Get target concentrations (for validation)
target_concentrations = tuner.get_target_concentrations()
for species_name, concentration in target_concentrations.items():
    print(f"Target concentration for {species_name}: {concentration:.3f}")

# 8. Access regulator-parameter mappings
regulator_parameter_map = model.get_regulator_parameter_map()
drug_map = regulator_parameter_map.get("D", {})
for drug_param in drug_map:
    print(f"Drug D regulates parameter: {drug_param}")
    model.set_parameter(drug_param, 10.0)  # Override for drug effect
```

### Tuner Algorithm

The `KineticParameterTuner` solves full nonlinear kinetic equations:

1. **Target Assignment**: For each species X, randomly assign active percentage p ∈ `active_percentage_range`
2. **Compute Target**: Target active concentration `[X_a] = p × X_total`
3. **Parameter Solving**: For each activated species Xa:
   - Identify forward parameters (kc_i, Ki values) and backward parameters (vmax_b)
   - Get regulator active concentrations from target concentrations
   - Solve: `p = (Σ kc_i × [Y_i_a]) / (Σ kc_i × [Y_i_a] + vmax_b)`
   - Set individual kc_i values for equal contribution
   - Set km_b = X_total × X_total_multiplier
   - Set km_f = km_b / (1 + Σ [Y_inhibitor_a]/ki_val) for competitive inhibition
   - Set ki = ki_val (constant)

### Complexity

- **Time**: O(n × m) where n = activated species, m = average regulators per species
- **Space**: O(n + p) where n = species, p = total parameters

The algorithm processes each species independently once all target concentrations are known. Drug concentrations are not considered in the tuning process.
