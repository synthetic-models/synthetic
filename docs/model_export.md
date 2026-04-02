# Model Export & Interoperability

Export Synthetic models to standard formats for use in other tools.

## SBML Export

SBML (Systems Biology Markup Language) is the standard for sharing biochemical models:

```python
from synthetic import Builder

vc = Builder.specify(degree_cascades=[2, 5, 10], random_seed=42)
model = vc.model

# Save to file
model.save_sbml_model_as('synthetic_model.sbml')

# Get as string
sbml_string = model.get_sbml_model()
```

SBML files can be imported into COPASI, libRoadRunner, and other SBML-compliant tools.

## Antimony Export

Antimony is a human-readable format for model inspection and modification:

```python
# Save to file
model.save_antimony_model_as('synthetic_model.ant')

# Get as string
antimony_string = model.get_antimony_model()
```

## Model Inspection

```python
# Model overview
print(model.head())

# Statistics
print(f"Species: {len(model.get_state_variables())}")
print(f"Parameters: {len(model.get_parameters())}")
print(f"Reactions: {len(model.reactions)}")
```

---

**See also:**

- [Model Building](model_building.md) — constructing models from scratch
- [Solvers & Simulation](solvers_and_simulation.md) — running simulations on exported models
- [API Reference](api_reference.md) — full API docs for `ModelBuilder`
