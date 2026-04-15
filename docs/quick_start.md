# Quick Start

## Your First Dataset

Generate a synthetic drug response dataset in three lines:

```python
from synthetic import Builder, make_dataset_drug_response

# 1. Spec & Model Layers: Define topology and automate ODE generation
vc = Builder.specify(degree_cascades=[3, 5], random_seed=42)

# 2. Solver Layer: Generate scikit-learn compatible dataset
X, y = make_dataset_drug_response(n=1000, cell_model=vc, target_specie='Oa')

print(f"Feature matrix shape: {X.shape}")  # (1000, n_features)
print(f"Target vector shape: {y.shape}")    # (1000,)
```

1. **`Builder.specify()`** creates a virtual cell. You can pass a `degree_cascades` list (e.g., `[3, 5]`) or a pre-configured `Spec` object. This handles the conversion from a high-level network topology to a concrete system of Ordinary Differential Equations (ODEs).

2. **`make_dataset_drug_response(1000, ...)`** generates 1000 samples by perturbing initial conditions and kinetic parameters, simulating the ODE model with each perturbation, and extracting the outcome species `Oa` (activated outcome) as the target.

## Customizing Your Model

### Network Size

Control the complexity by changing `degree_cascades`:

```python
# Small network
vc = Builder.specify(degree_cascades=[1, 2])

# Larger network with more degrees
vc = Builder.specify(degree_cascades=[3, 6, 15, 25])
```

### Reproducibility

Set a random seed for reproducible results:

```python
vc = Builder.specify(degree_cascades=[3, 5], random_seed=42)
```

### Parallel Processing

Use `n_cores=-1` to use all available CPUs and `verbose=True` to show progress:

```python
X, y = make_dataset_drug_response(
    n=1000,
    cell_model=vc,
    verbose=True,
    n_cores=-1,
)
```

## Spec-Model-Solver Abstraction

For more transparency and control, you can use the multi-layer API directly. This demonstrates how Synthetic separates network topology from simulation:

```python
from synthetic.Specs.DegreeInteractionSpec import DegreeInteractionSpec
from synthetic.Solver.ScipySolver import ScipySolver

# 1. Spec Layer: Define the network topology
spec = DegreeInteractionSpec(degree_cascades=[1, 2, 5])
spec.generate_specifications(feedback_density=0.5)

# 2. Model Layer: Generate the concrete ODE system
model = spec.generate_network("MyModel")
model.precompile()

# 3. Solver Layer: Simulate the model
solver = ScipySolver()
solver.compile(model.get_antimony_model())
results = solver.simulate(start=0, stop=1000, step=100)
```

## Next Steps

- [Network & Drug Design](network_and_drug_design.md) - Understand network topology and drug mechanisms
- [Solvers & Simulation](solvers_and_simulation.md) - Direct solver usage and timecourse simulation
- [Data Generation](data_generation.md) - Perturbation strategies, extended formats, and feature analysis
- [Advanced Features](advanced_features.md) - Kinetic tuning, HTTP solver
- [Model Export](model_export.md) - Model export and interoperability
- [Benchmarking](benchmarking.md) - Parameter estimation, ML benchmarking
- [API Reference](api_reference.md) - Full API documentation

!!! warning "Using the low-level API?"
    When building models directly with `ModelBuilder`, `Reaction`, or `ReactionArchtype`, you must call `model.precompile()` before accessing parameters or simulating. The high-level `Builder.specify()` API handles this automatically. See [Model Building](model_building.md) for details.
