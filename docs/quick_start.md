# Quick Start

## Installation

```bash
pip install synthetic-models
```

For development:

```bash
git clone https://github.com/synthetic-models/synthetic.git
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
```

## Your First Dataset

Generate a synthetic drug response dataset in three lines:

```python
from synthetic import Builder, make_dataset_drug_response

vc = Builder.specify(degree_cascades=[3, 5], random_seed=42)
X, y = make_dataset_drug_response(n=1000, cell_model=vc, target_specie='Oa')

print(f"Feature matrix shape: {X.shape}")  # (1000, n_features)
print(f"Target vector shape: {y.shape}")    # (1000,)
```

1. **`Builder.specify([3, 5])`** creates a virtual cell with a hierarchical signaling network. The `[3, 5]` argument defines the network topology: 3 cascades at degree 1 (closest to the drug target) and 5 cascades at degree 2 (downstream effectors). A drug is auto-generated that targets the degree 1 receptor species.

2. **`make_dataset_drug_response(1000, ...)`** generates 1000 samples by perturbing initial conditions and kinetic parameters, simulating the ODE model with each perturbation, and extracting the outcome species `Oa` (activated outcome) as the target. The result is a scikit-learn-compatible feature matrix `X` (species concentrations) and target vector `y` (drug response).

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

## Next Steps

- [Network & Drug Design](network_and_drug_design.md) - Understand network topology and drug mechanisms
- [Solvers & Simulation](solvers_and_simulation.md) - Direct solver usage and timecourse simulation
- [Data Generation](data_generation.md) - Perturbation strategies, extended formats, and feature analysis
- [Advanced Workflows](advanced_workflows.md) - Kinetic tuning, parameter estimation, and model export
