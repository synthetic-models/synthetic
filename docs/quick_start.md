# Quick Start

This page shows the fastest path to a usable dataset. The pattern: describe a network, simulate it many times with perturbed parameters, return the results in scikit-learn's `(X, y)` shape.

The library covers [four capabilities](index.md#how-synthetic-works) — model building, solving ODEs, obtaining data, and using the data — each in its own section. This page uses the simplest entry to each: `Builder.specify(degree_cascades=...)` for the model, `make_dataset_drug_response` for the data. The other paths are covered in the Usage section; you can ignore them until you need them.

## Your First Dataset

```python
from synthetic import Builder, make_dataset_drug_response

vc = Builder.specify(degree_cascades=[3, 5], random_seed=42)
X, y = make_dataset_drug_response(n=1000, cell_model=vc, target_specie='Oa')

print(f"Feature matrix shape: {X.shape}")  # (1000, n_features)
print(f"Target vector shape: {y.shape}")    # (1000,)
```

Two things happened:

1. **`Builder.specify(degree_cascades=[3, 5])`** created a virtual cell. The `[3, 5]` argument defines a hierarchical network with 3 cascades at the first level and 5 at the second. A drug that targets the top-level receptors was created automatically.
2. **`make_dataset_drug_response(n=1000, ...)`** simulated that virtual cell 1000 times, perturbing initial conditions and kinetic parameters each time. It returned a scikit-learn-compatible feature matrix `X` (one column per species) and a target vector `y` (the value of the `Oa` species — the activated outcome — in each simulation).

You don't need to know what any of that means yet. **Next: [Model Building](model_building.md)** explains the network `Builder.specify` created and how to configure it (or swap in a different model-building path).

## Customizing Your Model

Three knobs you'll reach for most often:

```python
from synthetic import Builder, make_dataset_drug_response

# Bigger network (more degrees of regulation)
vc = Builder.specify(degree_cascades=[3, 6, 15, 25], random_seed=42)

# Reproducibility — same seed, same network
vc = Builder.specify(degree_cascades=[3, 5], random_seed=42)

# Speed up large sweeps with all CPUs
X, y = make_dataset_drug_response(n=10_000, cell_model=vc, n_cores=-1, verbose=True)
```

The rest of this page assumes you have a compiled `vc` ready to query.

## What You Got

```python
print(vc.get_species_names())           # All species, e.g. ['R1_1', 'R1_1a', ...]
print(vc.get_initial_values())          # Initial concentrations
print(X.columns.tolist())               # Feature columns match species names
```

- `X` is a `pandas.DataFrame` of species concentrations — one column per species, one row per simulated sample.
- `y` is a `pandas.Series` of the `Oa` outcome value, one per row of `X`.
- The two are aligned: `X.iloc[i]` and `y.iloc[i]` come from the same simulation.

That's enough to drop the result into any scikit-learn workflow:

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
model = RandomForestRegressor(random_state=0).fit(X_train, y_train)
print(model.score(X_test, y_test))
```

## The Other Paths

`Builder.specify(degree_cascades=...)` only produces *hierarchical* Michaelis-Menten networks. If that shape is wrong for your problem — or you want to control the network more directly — the other paths into Model Building are:

- **The spec classes** (`DegreeInteractionSpec` directly, or `MichaelisNetworkSpec` for non-hierarchical Michaelis-Menten networks). Both expose species, regulations, and drugs as code objects you can manipulate. See [Model Building § The spec classes](model_building.md#level-2-the-spec-classes).
- **The `ModelBuilder` API** for arbitrary models — different kinetics, custom rate laws, hand-written reactions. See [Model Building § `ModelBuilder` + `Reaction`](model_building.md#level-3-modelbuilder-reaction).
- **Customising any layer of the pipeline** — writing your own spec, your own rate equation, your own solver. See [Advanced](advanced.md).
