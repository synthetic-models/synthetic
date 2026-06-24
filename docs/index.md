# Synthetic

Synthetic is a library for generating virtual cell data using ODE models based on biochemical laws common in cancer cell signaling networks. It creates datasets compatible with scikit-learn's `make_regression` format.

## Why Synthetic?

Real cell signaling data is expensive to generate, difficult to replicate, and often incomplete. Synthetic lets you generate **unlimited, labeled datasets** with full control over network topology, drug mechanisms, and parameter distributions — making it ideal for:

- **ML benchmarking** — train and evaluate models on data with known ground truth
- **Method development** — test new algorithms against biologically plausible signaling cascades
- **Drug response modeling** — simulate combination therapies with configurable up/down regulation
- **Education** — explore how network structure drives emergent behavior in signaling pathways

## How Synthetic Works

Synthetic generates datasets by **simulating ODE models of biochemical signaling networks**. The library covers four capabilities, each in its own section:

1. **[Model Building](model_building.md)** — describe the network. Three levels of control, from a single size parameter (`Builder.specify(degree_cascades=...)`) to writing individual reactions.
2. **[Solving ODEs](solving_odes.md)** — run the simulation. Three solver backends (Scipy, libRoadRunner, HTTP); accepts SBML from anywhere.
3. **[Obtaining Data](obtaining_data.md)** — generate many simulations and package them as a scikit-learn-shaped dataset, with full timecourse and kinetic parameters optionally exposed.
4. **[Use Cases](use_cases.md)** — finished end-to-end examples: parameter estimation, classification, ML model comparison.

For deep customisation of any layer — custom specs, custom rate equations, custom solvers — see [Advanced](advanced.md).

## Installation

The recommended way to install Synthetic is via PyPI:

```bash
pip install synthetic-models
```

To install the latest from GitHub:

```bash
pip install git+https://github.com/synthetic-models/synthetic.git
```

### Optional dependencies

| Extra | Install | Description |
|-------|---------|-------------|
| RoadRunner solver | `pip install synthetic-models[roadrunner]` | libRoadRunner SBML simulation engine — an alternative solver backend with mature SBML support |
| Plotting | `pip install synthetic-models[plotting]` | matplotlib, seaborn, networkx, and Jupyter — for visualizing networks and simulation results |
| Scikit-learn | `pip install synthetic-models[sklearn]` | scikit-learn — for ML pipeline integration and dataset utilities |

Install multiple extras at once:

```bash
pip install synthetic-models[roadrunner,plotting,sklearn]
```

### Install from source

If you can't access PyPI, download a `tar.gz` release from [GitHub Releases](https://github.com/synthetic-models/synthetic/releases) and install locally:

```bash
pip install synthetic-models-<version>.tar.gz
```

### Build from source

This project is developed using [uv](https://docs.astral.sh/uv/), which is the recommended way to build from source:

```bash
git clone https://github.com/synthetic-models/synthetic.git
cd Synthetic
uv sync
```

## Quick Start

!!! tip "Generate your first dataset in 3 lines"

    ```python
    from synthetic import Builder, make_dataset_drug_response

    vc = Builder.specify(degree_cascades=[3, 5], random_seed=42)
    X, y = make_dataset_drug_response(n=1000, cell_model=vc, target_specie='Oa')
    ```

Head to the [Quick Start](quick_start.md) guide for more details.
