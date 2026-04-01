# Synthetic

Synthetic is a library for generating virtual cell data using ODE models based on biochemical laws common in cancer cell signaling networks. It creates datasets compatible with scikit-learn's `make_regression` format.

## Features

- **Hierarchical network generation** with configurable degree cascades and feedback regulation
- **Drug response modeling** with customizable drug mechanisms (up/down regulation)
- **Multiple solver backends** (SciPy, libRoadRunner, HTTP)
- **Biologically plausible parameters** via kinetic parameter tuning
- **Scikit-learn compatible** datasets for ML benchmarking
- **SBML/Antimony export** for interoperability with other tools

## Quick Start

```python
from synthetic import Builder, make_dataset_drug_response

vc = Builder.specify(degree_cascades=[3, 5], random_seed=42)
X, y = make_dataset_drug_response(n=1000, cell_model=vc, target_specie='Oa')
```

Head to the [Quick Start](quick_start.md) guide to get started.

## Documentation

| Section | Description |
|---------|-------------|
| [Quick Start](quick_start.md) | Installation and first dataset in 3 lines |
| [Network & Drug Design](network_and_drug_design.md) | Degree cascades, feedback, drug mechanisms, combination therapy |
| [Solvers & Simulation](solvers_and_simulation.md) | Solver backends, timecourse simulation, HTTP solver |
| [Data Generation](data_generation.md) | Perturbation strategies, extended formats, feature analysis |
| [Advanced Workflows](advanced_workflows.md) | Kinetic tuning, parameter estimation, model export, ML benchmarking |

## Installation

```bash
pip install synthetic-models
```
