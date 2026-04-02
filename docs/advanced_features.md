# Advanced Features

## Kinetic Parameter Tuning

The `KineticParameterTuner` generates kinetic parameters that ensure robust signal propagation through the network. Without tuning, random parameter values may produce unresponsive dynamics or 'flat' responses unintentionally.

!!! tip "When to use tuning"
    Use kinetic tuning whenever you want predictable, responsive network dynamics. It is enabled by default in `Builder.specify()`. You only need the manual approach below if you're using the low-level API directly.

### Why Tuning Matters

In hierarchical signaling networks, signal propagation depends on the balance between forward (activation) and backward (deactivation) reaction rates. The tuner solves the full nonlinear kinetic equations to find parameters where each species reaches a target active percentage.

### Usage via VirtualCell

Tuning is enabled by default when compiling:

```python
from synthetic import Builder

vc = Builder.specify(degree_cascades=[3, 6, 15], random_seed=42)
# Kinetic tuning is applied automatically during compile()
```

Customize tuning parameters:

```python
vc = Builder.specify(
    degree_cascades=[3, 6, 15],
    random_seed=42,
    auto_compile=False,
)

vc.compile(
    use_kinetic_tuner=True,
    active_percentage_range=(0.3, 0.7),  # Target 30-70% active states
    X_total_multiplier=5.0,               # Km_b = X_total * 5.0
    ki_val=100.0,                         # Constant Ki for all inhibitors
    v_max_f_random_range=(5.0, 10.0),     # Forward Vmax range
)
```

### Accessing Tuning Results

```python
# Target concentrations the tuner aimed for
targets = vc.get_target_concentrations()
for species, concentration in targets.items():
    print(f"{species}: {concentration:.3f}")

# Regulator-parameter mappings
regulator_map = vc.model.get_regulator_parameter_map()
for regulator, params in regulator_map.items():
    print(f"Regulator {regulator} affects: {list(params.keys())}")
```

### Direct Usage

Use `KineticParameterTuner` without the high-level API:

```python
from synthetic.Specs.DegreeInteractionSpec import DegreeInteractionSpec
from synthetic.Specs.Drug import Drug
from synthetic.utils.kinetic_tuner import KineticParameterTuner

# Create specification
spec = DegreeInteractionSpec(degree_cascades=[3, 6, 15])
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

# Generate and precompile model
model = spec.generate_network("MyNetwork", random_seed=42)
model.precompile()

# Tune parameters
tuner = KineticParameterTuner(model, random_seed=42)
updated_params = tuner.generate_parameters(
    active_percentage_range=(0.3, 0.7),
    X_total_multiplier=5.0,
    ki_val=100.0,
    v_max_f_random_range=(5.0, 10.0),
)

# Apply tuned parameters
for param_name, value in updated_params.items():
    model.set_parameter(param_name, value)

# Get target concentrations for validation
target_concentrations = tuner.get_target_concentrations()
```

## HTTP Solver for Remote Simulation

The `HTTPSolver` sends simulation requests to a remote server via HTTP. This is useful for distributed computing or when the simulation engine runs on a separate machine.

### Client Usage

```python
from synthetic.Solver.HTTPSolver import HTTPSolver

solver = HTTPSolver()

# Connect and validate endpoint
solver.compile("http://localhost:8000/simulate")

# Get default states and parameters
states = solver.get_state_defaults()
params = solver.get_parameter_defaults()

# Override values
solver.set_state_values({"R1_1": 1000.0})
solver.set_parameter_values({"Km_J0": 10.0})

# Run simulation
results = solver.simulate(start=0, stop=60, step=0.5)
```

### Server Setup

The HTTP solver requires a server implementing the simulation API. See `examples/httpsolver_api.md` for the full API specification. A minimal FastAPI server:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Optional

app = FastAPI()

class SimulationRequest(BaseModel):
    start: float
    stop: float
    step: float
    state_values: Optional[Dict[str, float]] = None
    parameter_values: Optional[Dict[str, float]] = None

@app.get("/states")
async def get_states():
    return {"S1": 100.0, "S1a": 0.0}

@app.get("/parameters")
async def get_parameters():
    return {"Vmax_J0": 100.0, "Km_J0": 50.0}

@app.post("/simulate")
async def simulate(req: SimulationRequest):
    # Run simulation with requested parameters
    return {"time": [req.start, req.stop], "S1": [100.0, 90.0], "S1a": [0.0, 10.0]}
```

---

**See also:**

- [Solvers & Simulation](solvers_and_simulation.md) — built-in solver configuration
- [Benchmarking Scenarios](benchmarking.md) — parameter estimation and ML benchmarking
- [API Reference](api_reference.md) — full API docs for `KineticParameterTuner`, `HTTPSolver`, and other classes
- [FAQ](faq.md) — common questions about simulation and parameter tuning
