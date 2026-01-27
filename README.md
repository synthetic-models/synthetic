# Synthetic

Synthetic (aka ModelBuilder) is a virtual cell generation library for generating biological synthetic data that can be used for benchmarking predictive modelling workflows. The synthetic data are generated using ODE models formulated in biochemical laws common in cancer cell signalling networks. Synthetic can create datasets in the format of scikit-learn's `make_regression` method.

## Installation

```bash
pip install synthetic
```

For development:

```bash
git clone https://github.com/AnEvilBurrito/Synthetic.git
cd Synthetic
uv sync
```

### Alpha Testing

For lab members participating in alpha testing, install directly from the lab's GitHub repository:

```bash
pip install git+https://github.com/IntegratedNetworkModellingLab/Synthetic.git
```

To install a specific version or commit:

```bash
pip install git+https://github.com/IntegratedNetworkModellingLab/Synthetic.git@v0.1.0
# or
pip install git+https://github.com/IntegratedNetworkModellingLab/Synthetic.git@<commit-hash>
```

For lab members with write access, you can use SSH:

```bash
pip install git+ssh://git@github.com/IntegratedNetworkModellingLab/Synthetic.git
```

## Quick Start

The high-level API provides a simple interface for creating virtual cell models and generating sklearn-compatible datasets:

```python
from synthetic import Builder, make_dataset_drug_response

# Create a virtual cell model (auto-compiles with default settings)
vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42)

# Generate a sklearn-compatible dataset
X, y = make_dataset_drug_response(n=1000, cell_model=vc, target_specie='Oa')

print(f"Feature matrix shape: {X.shape}")  # (1000, n_features)
print(f"Target vector shape: {y.shape}")    # (1000,)
```

## Examples

The `examples/` directory contains practical examples demonstrating various features of Synthetic:

### create_dataset.py

Basic example showing how to generate a synthetic drug response dataset and analyze feature correlations.

```bash
python examples/create_dataset.py
```

**Demonstrates:**
- Creating a virtual cell model with custom network topology
- Generating sklearn-compatible datasets
- Calculating Pearson correlations between features and targets
- Identifying predictive features in the network

### export_model_sbml.py

Shows how to export synthetic models to standard SBML and Antimony formats for interoperability with other tools.

```bash
python examples/export_model_sbml.py
```

**Demonstrates:**
- Creating a virtual cell model
- Accessing the underlying ModelBuilder
- Exporting models to SBML format (for COPASI, libRoadRunner, etc.)
- Exporting models to Antimony format (human-readable)
- Retrieving model strings for programmatic use

### detailed_examples.md

Comprehensive guide with detailed workflows and advanced use cases, including:

- **Feature correlation analysis** - Complete workflow with CSV export
- **Combination therapy** - Testing multi-drug interactions
- Step-by-step explanations of each workflow

View the guide:
```bash
cat examples/detailed_examples.md
```

or open it in your preferred markdown viewer.

## API Overview

### VirtualCell

The `VirtualCell` class represents a virtual cell model with a hierarchical network topology.

```python
from synthetic import VirtualCell

# Create a virtual cell without auto-compiling
vc = VirtualCell(
    degree_cascades=[1, 2, 5],      # Network topology: 1 cascade degree 1, 2 cascades degree 2, 5 cascades degree 3
    name="MyCell",
    random_seed=42,
    feedback_density=0.5,           # 50% of possible feedback connections
    auto_compile=False,             # Disable auto-compile for custom configuration
    auto_drug=True,                 # Auto-generate a drug targeting degree 1 species
    drug_name="D",
    drug_start_time=5000.0,          # Drug becomes active at t=5000
    drug_value=100.0,                # Drug active concentration
    drug_regulation_type="down",    # Drug inhibits targets
    simulation_end=10000.0,          # Default simulation end time
)

# Compile the model (required before simulation)
vc.compile(
    mean_range_species=(50, 150),          # Initial concentration range
    rangeScale_params=(0.8, 1.2),          # Parameter scaling range
    use_kinetic_tuner=True,                # Use KineticParameterTuner for biologically plausible parameters
    active_percentage_range=(0.3, 0.7),     # Target 30-70% active species
)
```

#### Methods

| Method | Description |
|--------|-------------|
| `add_drug(...)` | Add a drug to the model |
| `list_drugs()` | List all drugs in the system |
| `compile(...)` | Compile the model (lazy initialization) |
| `get_species_names(exclude_drugs=True)` | Get list of species names |
| `get_initial_values(exclude_drugs=True)` | Get initial species values |
| `get_target_concentrations()` | Get target active concentrations from kinetic tuner |

#### Properties

| Property | Description |
|----------|-------------|
| `spec` | Underlying `DegreeInteractionSpec` object |
| `model` | Underlying `ModelBuilder` object |
| `tuner` | Underlying `KineticParameterTuner` object (if used) |

### Builder

The `Builder` class provides a factory method for creating `VirtualCell` instances with a clean, fluent API.

```python
from synthetic import Builder

# Basic usage (auto-compiles)
vc = Builder.specify(degree_cascades=[1, 2, 5])

# With custom parameters
vc = Builder.specify(
    degree_cascades=[3, 6, 15, 25],
    name="ComplexCell",
    random_seed=42,
    feedback_density=0.5,
    auto_compile=True,
    auto_drug=True,
    drug_name="Inhibitor",
    drug_start_time=5000.0,
    drug_value=100.0,
    drug_regulation_type="down",
    simulation_end=10000.0,
)
```

### Adding Drugs

Drugs can be added manually to target degree 1 R species:

```python
# Create a virtual cell without auto-drug
vc = Builder.specify(degree_cascades=[1, 2, 5], auto_drug=False)

# Add a drug manually (returns self for chaining)
vc.add_drug(
    name="DrugX",
    start_time=500.0,               # Time at which drug becomes active
    default_value=0.0,              # Default value when not active
    regulation=["R1_1", "R1_2"],     # Target species (must be degree 1 R species)
    regulation_type=["down", "down"], # Regulation types: "up" or "down"
    value=100.0,                    # Optional override for active concentration
)

# Compile to apply drugs
vc.compile()
```

List all drugs in the system:

```python
drugs = vc.list_drugs()
for drug in drugs:
    print(f"Drug: {drug['name']}, Targets: {drug['targets']}, Types: {drug['types']}")
```

### make_dataset_drug_response

Generate synthetic drug response datasets compatible with scikit-learn's `make_regression` format.

```python
from synthetic import Builder, make_dataset_drug_response

vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42)

# Basic usage
X, y = make_dataset_drug_response(
    n=1000,                          # Number of samples
    cell_model=vc,                   # Compiled VirtualCell instance
    target_specie='Oa',              # Outcome species to use as target
)

# With custom parameters
X, y = make_dataset_drug_response(
    n=500,
    cell_model=vc,
    target_specie='Oa',
    perturbation_type='gaussian',    # 'uniform', 'gaussian', 'lognormal', 'lhs'
    perturbation_params={'rsd': 0.2}, # Relative standard deviation
    simulation_params={'start': 0, 'end': 10000, 'points': 101},
    solver_type='scipy',            # 'scipy' or 'roadrunner'
    jit=True,                        # Enable JIT compilation (scipy only)
    verbose=True,                    # Show progress bar
    seed=42,                         # Random seed for reproducibility
)
```

#### Extended Return Format

For more detailed output, use `return_details=True`:

```python
result = make_dataset_drug_response(
    n=100,
    cell_model=vc,
    return_details=True,
)

# Access extended data structure
X = result['X']                      # Feature matrix
y = result['y']                      # Target vector
features = result['features']       # Feature dataframe (initial values)
targets = result['targets']         # Target dataframe (outcome values)
timecourse = result['timecourse']   # Timecourse simulation data
metadata = result['metadata']        # Metadata about generation
```

#### Perturbation Types

| Type | Description | Parameters |
|------|-------------|-------------|
| `uniform` | Uniform distribution | `min`, `max` |
| `gaussian` | Gaussian (normal) distribution | `rsd` (relative standard deviation) |
| `lognormal` | Log-normal distribution | `rsd` |
| `lhs` | Latin Hypercube Sampling | `rsd` |

#### Kinetic Parameter Perturbation

You can also perturb kinetic parameters:

```python
X, y = make_dataset_drug_response(
    n=100,
    cell_model=vc,
    parameter_values={'Km_J0': 100.0},      # Base parameters to perturb
    param_perturbation_type='gaussian',
    param_perturbation_params={'rsd': 0.2},
    param_seed=42,                          # Separate seed for parameters
)
```

#### Simulation Robustness

The API includes robust simulation handling with automatic resampling:

```python
X, y = make_dataset_drug_response(
    n=100,
    cell_model=vc,
    resample_size=10,           # Number of alternative samples on failure
    max_retries=3,              # Maximum retries per failed index
    require_all_successful=False, # If False, returns partial results
)
```

### Solver Options

Two solver backends are available:

| Solver | Description | Features |
|--------|-------------|----------|
| `scipy` | Uses `scipy.integrate.odeint` with JIT compilation | Fast for batch simulations |
| `roadrunner` | Uses libRoadRunner for SBML simulation | Mature, robust |

```python
# Scipy solver (default)
X, y = make_dataset_drug_response(
    n=100, cell_model=vc, solver_type='scipy', jit=True
)

# Roadrunner solver
X, y = make_dataset_drug_response(
    n=100, cell_model=vc, solver_type='roadrunner'
)
```

## Example: Complete Workflow

```python
from synthetic import Builder, make_dataset_drug_response
import numpy as np

# 1. Create a virtual cell with custom network topology
vc = Builder.specify(
    degree_cascades=[3, 6, 15],   # 3 degree-1, 6 degree-2, 15 degree-3 cascades
    random_seed=42,
    feedback_density=0.5,
)

# 4. Generate dataset
X, y = make_dataset_drug_response(
    n=1000,
    cell_model=vc,
    target_specie='Oa',
    perturbation_type='gaussian',
    perturbation_params={'rsd': 0.2},
    simulation_params={'start': 0, 'end': 8000, 'points': 101},
    seed=42,
    verbose=True,
)

# 5. Use with sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Model R^2 score: {score:.3f}")
```

## Network Topology

The `degree_cascades` parameter defines the hierarchical network structure:

```python
# Simple network: 1 cascade each across 3 degrees
vc = Builder.specify(degree_cascades=[1, 1, 1])
# Creates: R1_1 -> I1_1, R2_1 -> I2_1, R3_1 -> I3_1, with hierarchical connections

# More complex network
vc = Builder.specify(degree_cascades=[3, 6, 15, 25])
# Creates many parallel cascades with increasing complexity per degree
```

Each cascade creates:
- A Receptor (R) species at each degree
- An Intermediate (I) species at each degree
- The first degree also connects to an Outcome (O) species

## Kinetic Parameter Tuning

The library includes `KineticParameterTuner` for generating biologically plausible kinetic parameters:

```python
from synthetic import KineticParameterTuner

vc = Builder.specify(degree_cascades=[3, 6, 15], random_seed=42)
vc.compile(
    use_kinetic_tuner=True,
    active_percentage_range=(0.3, 0.7),  # Target 30-70% active states
    X_total_multiplier=5.0,               # Km_b = X_total * 5.0
    ki_val=100.0,                        # Constant Ki for inhibitors
    v_max_f_random_range=(5.0, 10.0),    # Forward Vmax range
)

# Access target concentrations
targets = vc.get_target_concentrations()
for species, concentration in targets.items():
    print(f"{species}: {concentration:.2f}")
```

## Advanced Usage

### Accessing Underlying Components

```python
vc = Builder.specify(degree_cascades=[1, 2, 5])

# Access the specification
spec = vc.spec
print(spec.species_list)

# Access the model builder
model = vc.model
print(model.get_antimony_model())

# Access the tuner (if used)
tuner = vc.tuner
print(tuner.get_target_concentrations())
```

### Direct Model Building

For more control, you can use the lower-level API:

```python
from synthetic.Specs.DegreeInteractionSpec import DegreeInteractionSpec
from synthetic.Specs.Drug import Drug
from synthetic.utils.kinetic_tuner import KineticParameterTuner

# Create specification
spec = DegreeInteractionSpec(degree_cascades=[1, 2, 5])
spec.generate_specifications(random_seed=42, feedback_density=0.5)

# Add drug
drug = Drug(
    name="D",
    start_time=5000.0,
    default_value=0.0,
    regulation=["R1_1", "R1_2", "R1_3"],
    regulation_type=["down", "down", "down"],
)
spec.add_drug(drug, value=100.0)

# Generate model
model = spec.generate_network(
    network_name="MyNetwork",
    random_seed=42,
)
model.precompile()

# Apply kinetic tuning
tuner = KineticParameterTuner(model, random_seed=42)
updated_params = tuner.generate_parameters(
    active_percentage_range=(0.3, 0.7),
    X_total_multiplier=5.0,
    ki_val=100.0,
)
for param_name, value in updated_params.items():
    model.set_parameter(param_name, value)
```

## Development

### Running Tests

```bash
pytest
```

Run specific tests:

```bash
pytest tests/test_api.py::TestVirtualCell::test_compile_with_kinetic_tuner
```
