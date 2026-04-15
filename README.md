# Synthetic

1. **ODE Model Building** — Construct ODE models using high-level abstractions and syntax. Define network topologies, reaction archetypes, and regulatory interactions without writing equations by hand. Synthetic also provides simulation capabilities (SciPy and RoadRunner) and export options to standard formats like Antimony and SBML. 

2. **Data Generation** — Treat ODE models as ground-truth systems and generate `(X, y)` datasets compatible with scikit-learn's `make_regression`. Perturb initial conditions or parameters, simulate the model, and collect feature–target pairs for benchmarking predictive modeling workflows.

Based on these two primary features, Synthetic can be used to benchmark a wide range of predictive / mathematical modelling workflows including: parameter estimation, identifiability analysis, feature selection, network inference. The library is designed to be flexible and extensible, allowing users to customise model generation and data generation processes to suit their specific research questions.

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

## Core Features

Synthetic is built around two primary workflows: **ODE Model Building** and **Data Generation**.

### 1. ODE Model Building

Synthetic provides a high-level "Spec" layer to define complex network topologies. These specifications are then compiled into a `ModelBuilder` which manages the underlying biochemical reactions.

```python
from synthetic.Specs.DegreeInteractionSpec import DegreeInteractionSpec

# 1. Create a specification for a hierarchical network
# [1, 2, 5] means: 1 cascade at degree 1, 2 at degree 2, 5 at degree 3
spec = DegreeInteractionSpec(degree_cascades=[1, 2, 5])
spec.generate_specifications(random_seed=42, feedback_density=0.5)

# 2. Generate the ODE model (ModelBuilder) from the spec
model = spec.generate_network("MyModel", random_seed=42)
model.precompile()  # Required before accessing parameters or simulating

# 3. Export to standard formats for interoperability
antimony_str = model.get_antimony_model()  # Human-readable format
sbml_str = model.get_sbml_model()          # Standard XML format

# 4. Simulate the model
from synthetic.Solver.ScipySolver import ScipySolver
solver = ScipySolver()
solver.compile(antimony_str, jit=True)
results = solver.simulate(start=0, stop=1000, step=100)
```

### 2. Data Generation

Treat ODE models as ground-truth systems to generate `(X, y)` datasets. This demonstrates the **Spec → Model → Solver** abstractions used for large-scale data generation.

```python
from synthetic.Specs.DegreeInteractionSpec import DegreeInteractionSpec
from synthetic import Builder, make_dataset_drug_response

# 1. Spec & Model Layers: Define topology and automate ODE generation
# Builder.specify(degree_cascades=...) handles the Spec -> Model conversion
vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42)

# 2. Solver Layer: Generate a scikit-learn compatible dataset
# Treats the ODE model as ground-truth for batch simulations
X, y = make_dataset_drug_response(
    n=1000, 
    cell_model=vc, 
    target_specie='Oa',
    perturbation_type='gaussian'
)

# Use directly with machine learning workflows
from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor().fit(X, y)
print(f"Prediction R^2: {regr.score(X, y):.3f}")
```



