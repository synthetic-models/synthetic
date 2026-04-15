# Solvers & Simulation

## Using the Built-in Solver

`make_dataset_drug_response` handles solver creation internally. Choose the backend with `solver_type`. 

Synthetic supports several entry points for data generation:
- **`VirtualCell`**: High-level abstraction containing spec and model.
- **`ModelBuilder`**: A concrete ODE system (will automatically create a solver).
- **`Solver`**: A pre-compiled solver (e.g., `ScipySolver` or `RoadrunnerSolver`).

```python
from synthetic import Builder, make_dataset_drug_response

vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42)

# Option 1: Use VirtualCell (Default)
X, y = make_dataset_drug_response(n=100, cell_model=vc)

# Option 2: Use ModelBuilder
X, y = make_dataset_drug_response(n=100, cell_model=vc.model)

# Option 3: Use a pre-compiled Solver directly
solver = vc.model.get_solver(solver_type='scipy', jit=True)
X, y = make_dataset_drug_response(n=100, cell_model=solver)
```

## Direct Solver Usage

For custom simulation workflows, use solvers directly.

=== "ScipySolver"

    Uses `scipy.integrate.odeint` with optional JIT compilation:

    ```python
    from synthetic.Specs.DegreeInteractionSpec import DegreeInteractionSpec
    from synthetic.Solver.ScipySolver import ScipySolver

    # 1. Generate a model from a specification
    spec = DegreeInteractionSpec(degree_cascades=[2, 3, 4])
    spec.generate_specifications()
    model = spec.generate_network("MyModel")
    model.precompile()

    # 2. Get Antimony model string
    antimony = model.get_antimony_model()

    # 3. Compile and simulate
    solver = ScipySolver()
    solver.compile(antimony, jit=True)  # Enable JIT for faster execution
    results = solver.simulate(start=0, stop=10000, step=50)

    print(results.head())
    ```

=== "RoadrunnerSolver"

    Uses libRoadRunner for SBML simulation:

    !!! warning "Requires SBML format"
        RoadrunnerSolver requires SBML, not Antimony. Use `model.get_sbml_model()`.

    ```python
    from synthetic.Solver.RoadrunnerSolver import RoadrunnerSolver

    # 1. Get SBML model string from the same model object
    sbml = model.get_sbml_model()

    # 2. Compile and simulate
    solver = RoadrunnerSolver()
    solver.compile(sbml)
    results = solver.simulate(start=0, stop=10000, step=50)
    ```

### Simulation Output

Both solvers return a pandas DataFrame with a `time` column and one column per species:

```
       time    R1_1   R1_1a    I1_1   I1_1a      O      Oa
0       0.0  100.00    0.00  100.00    0.00  100.0    0.0
1      50.0   95.23    4.77   98.10    1.90   99.5    0.5
2     100.0   91.50    8.50   96.40    3.60   98.8    1.2
...
```

## Timecourse Simulation

Simulate and visualize species dynamics over time.

```python
from synthetic.Specs.DegreeInteractionSpec import DegreeInteractionSpec
from synthetic.Solver.ScipySolver import ScipySolver
import matplotlib.pyplot as plt

# 1. Create model
spec = DegreeInteractionSpec(degree_cascades=[2, 3, 4])
spec.generate_specifications()
model = spec.generate_network("MyModel")
model.precompile()

# 2. Simulate
solver = ScipySolver()
solver.compile(model.get_antimony_model(), jit=False)
timecourse = solver.simulate(start=0, stop=10000, step=50)

# 3. Plot outcome dynamics
fig, ax = plt.subplots()
ax.plot(timecourse['time'], timecourse['O'], label='O (inactive)')
ax.plot(timecourse['time'], timecourse['Oa'], label='Oa (active)')
ax.set_xlabel('Time')
ax.set_ylabel('Concentration')
ax.legend()
plt.show()
```

!!! info "JIT warmup"
    The first call to `ScipySolver.simulate()` with `jit=True` compiles the ODE function via Numba, which can take several seconds. Subsequent simulations are fast. This is a one-time cost per model.

---

**See also:**

- [Data Generation](data_generation.md) — using solvers to generate datasets
- [Benchmarking](benchmarking.md) — parameter estimation with solvers
- [API Reference](api_reference.md) — full API docs for solver classes
