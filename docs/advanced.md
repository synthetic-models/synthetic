# Advanced

Each section on this page is independent — read the one that matches the layer of the pipeline you want to extend. The library's design — three levels of model building, three solver backends, pluggable data-processing — is meant to be extended at every layer.

| Section | What it extends | When you'd reach for it |
|---|---|---|
| [Kinetic parameter tuning](#kinetic-parameter-tuning) | Model parameter generation | Random parameters produced unresponsive dynamics and you want biologically plausible ones |
| [Writing a custom spec](#writing-a-custom-spec) | Model topology | You want a network shape neither `DegreeInteractionSpec` nor `MichaelisNetworkSpec` produces |
| [Writing a custom rate equation](#writing-a-custom-rate-equation) | Reaction kinetics | You need a rate law the predefined archtypes don't cover |
| [Writing a custom solver](#writing-a-custom-solver) | Simulation engine | You want to integrate the ODEs using a backend Synthetic doesn't ship |
| [From regression to classification](#from-regression-to-classification) | ML framing of the dataset | The regression target isn't the right shape for the question you're asking |

---

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

---

## Writing a Custom Spec

When neither `DegreeInteractionSpec` (hierarchical) nor `MichaelisNetworkSpec` (random Michaelis-Menten) produces the topology you want, write a new spec by subclassing `BaseSpec`. The contract has two abstract methods:

- `generate_specifications(**kwargs)` — populate the spec's `species_list` and `regulations`.
- `generate_network(network_name, **kwargs) -> ModelBuilder` — produce a `ModelBuilder` ready to simulate.

The rest of the pipeline (solvers, dataset generation, export) treats the resulting `ModelBuilder` identically, regardless of which spec produced it.

### Example: Linear Cascade Spec

A linear cascade — N species in a strict chain, each activated by the previous one — is a useful network shape neither shipped spec produces:

```python
from synthetic.Specs.BaseSpec import BaseSpec
from synthetic.Specs.Regulation import Regulation
from synthetic.ModelBuilder import ModelBuilder
from synthetic.Reaction import Reaction
from synthetic import ArchtypeCollections


class LinearCascadeSpec(BaseSpec):
    """
    A linear signaling cascade: S0 → S0a → S1 → S1a → ... → SN → SNa.
    No branching, no feedback — a pure feed-forward chain.
    """

    def __init__(self, n_species: int):
        super().__init__()
        self.n_species = n_species

    def generate_specifications(self, random_seed: int = None, **kwargs):
        import random
        if random_seed is not None:
            random.seed(random_seed)

        self.species_list = []
        self.regulations = []

        mm = ArchtypeCollections.michaelis_menten

        for i in range(self.n_species):
            substrate = f"S{i}"
            product = f"S{i}a"
            self.species_list.extend([substrate, product])

            # Each species is activated by the previous species's active form
            if i > 0:
                prev_active = f"S{i-1}a"
                self.regulations.append(Regulation(
                    regulator=prev_active, target=substrate,
                    reg_type='activate',
                ))

    def generate_network(self, network_name: str, **kwargs) -> ModelBuilder:
        model = ModelBuilder(network_name)
        mm = ArchtypeCollections.michaelis_menten

        for i in range(self.n_species):
            substrate = f"S{i}"
            product = f"S{i}a"
            extra_states = (f"S{i-1}a",) if i > 0 else ()

            model.add_reaction(Reaction(
                reaction_archtype=mm,
                reactants=(substrate,),
                products=(product,),
                extra_states=extra_states,
                reaction_name=f'step_{i}',
                parameters_values={'Km': 50.0, 'Vmax': 10.0},
                reactant_values={substrate: 100.0},
                product_values={product: 0.0},
            ))

        model.precompile()
        return model
```

Using it:

```python
spec = LinearCascadeSpec(n_species=4)
spec.generate_specifications(random_seed=42)
model = spec.generate_network("MyChain")

print(model.get_antimony_model())
# 4 species, 4 reactions, no feedback, no drugs
```

The `model` returned here is interchangeable with one produced by `Builder.specify(...)` — pass it to any [solver](solving_odes.md), or wrap it in your own `make_dataset_drug_response`-style loop.

---

## Writing a Custom Rate Equation

If the predefined archtypes in `ArchtypeCollections` don't cover your kinetics, define a new `ReactionArchtype`. The constructor takes:

- `rate_law` — a string expression referencing parameter names and placeholder species (`&S`, `&P`, `&A0`, etc.)
- `parameters` — the parameter names the rate law references
- `reactants` / `products` / `extra_states` — placeholder species

Anything you write in `rate_law` is passed through to the Antimony / SBML backend, so the kinetics are real, not simulated.

### Example: AND-gate Activation

A species is activated only when *both* of two regulators are active simultaneously. This isn't a standard Michaelis-Menten pattern — it needs a multiplicative term in the rate law:

```python
from synthetic import ReactionArchtype, Reaction
from synthetic.ModelBuilder import ModelBuilder

and_gate = ReactionArchtype(
    name='AND-gate activation',
    reactants=('&S',),
    products=('&P',),
    extra_states=('&A0', '&A1'),    # two activators, both required
    parameters=('Vmax', 'Km', 'Ka0', 'Ka1', 'n'),
    # Multiplicative activation: rate is high only when both A0 and A1 are high.
    # The (Ka0*A0)*(Ka1*A1) term is near zero if either A is near zero.
    rate_law='Vmax * &S / (Km + &S) * (Ka0*&A0)^n * (Ka1*&A1)^n / ((Ka0*&A0)^n * (Ka1*&A1)^n + 1)',
    assume_parameters_values={'Vmax': 10, 'Km': 50, 'Ka0': 1.0, 'Ka1': 1.0, 'n': 2},
    assume_reactant_values={'&S': 100},
    assume_product_values={'&P': 0},
    assume_extra_state_values={'&A0': 0.5, '&A1': 0.5},
)

rxn = Reaction(
    reaction_archtype=and_gate,
    reactants=('Substrate',),
    products=('Product',),
    extra_states=('TF_A', 'TF_B'),     # both transcription factors must be active
    reaction_name='and_gate_step',
    parameters_values={'Vmax': 12, 'Km': 40, 'Ka0': 1.5, 'Ka1': 1.5, 'n': 3},
    reactant_values={'Substrate': 100},
    product_values={'Product': 0},
)

model = ModelBuilder('AND_Gated_Pathway')
model.add_reaction(rxn)
model.precompile()
```

The archtype is just a template — instantiate it as many times as needed with different species names and parameter values.

---

## Writing a Custom Solver

If you want to integrate the ODEs using a backend Synthetic doesn't ship — a different integrator, a hardware accelerator, a remote compute service — subclass `Solver`. The contract has two abstract methods:

- `compile(model_string: str, **kwargs)` — accept a model in Antimony or SBML format (or a file path), and prepare an internal model object.
- `simulate(start: float, stop: float, step: float) -> pd.DataFrame` — return a DataFrame with a `time` column and one column per species.

The base class also provides `set_state_values` and `set_parameter_values` for hot-swapping values between simulations — implement these if your backend supports it.

### Example: `scipy.integrate.solve_ivp` Backend

This is a thin wrapper around `scipy.integrate.solve_ivp` that uses Antimony to generate the ODE function. It's a minimal working example — production solvers would do much more.

```python
import re
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from synthetic.Solver.Solver import Solver
from antimony import loadAntimonyModel


class SolveIvpSolver(Solver):
    """Minimal ODE solver using scipy.integrate.solve_ivp and Antimony for the model."""

    def __init__(self):
        super().__init__()
        self._ode_func = None
        self._species_names = []
        self._initial_state = None

    def compile(self, compile_str: str, **kwargs):
        # Load the Antimony model and extract the ODE function + species list.
        loadAntimonyModel(compile_str)
        self._species_names = self._parse_species_names(compile_str)
        self._ode_func = self._build_ode_func(compile_str)
        self._initial_state = self._parse_initial_values(compile_str)
        return True

    def simulate(self, start: float, stop: float, step: float) -> pd.DataFrame:
        t_eval = np.arange(start, stop + step, step)
        sol = solve_ivp(
            self._ode_func,
            t_span=(start, stop),
            y0=self._initial_state,
            t_eval=t_eval,
            method='RK45',
        )
        df = pd.DataFrame(sol.y.T, columns=self._species_names)
        df.insert(0, 'time', sol.t)
        return df

    def set_state_values(self, state_values):
        for i, name in enumerate(self._species_names):
            if name in state_values:
                self._initial_state[i] = state_values[name]
        return True

    # --- helpers (sketched; full implementation needs sympy or a python SBML parser) ---
    def _parse_species_names(self, model_str):
        return re.findall(r'^\s*var\s+(\w+)', model_str, re.MULTILINE)

    def _build_ode_func(self, model_str):
        # Real implementations generate the ODE function via sympy (as ScipySolver does)
        # or load SBML and use a python SBML library. This is a placeholder.
        raise NotImplementedError("Use sympy or a python SBML library here")

    def _parse_initial_values(self, model_str):
        # Parse initial-value lines from Antimony (e.g. "S1 = 100;").
        initials = {}
        for m in re.finditer(r'(\w+)\s*=\s*([\d.eE+-]+)\s*;', model_str):
            if m.group(1) in self._species_names:
                initials[m.group(1)] = float(m.group(2))
        # Fill missing species with 0
        return np.array([initials.get(name, 0.0) for name in self._species_names])
```

Using it:

```python
vc = Builder.specify(degree_cascades=[2, 3, 4], random_seed=42)
antimony = vc.model.get_antimony_model()

solver = SolveIvpSolver()
solver.compile(antimony)
results = solver.simulate(start=0, stop=1000, step=10)
print(results.head())
```

The point isn't to replace `ScipySolver` — it's to show the surface area: two abstract methods, one optional hot-swap method, and the model-string → timecourse pipeline. Any backend that can ingest Antimony or SBML and emit a DataFrame can be plugged in here.

---

## From Regression to Classification

Synthetic's `make_dataset_drug_response` produces a *regression* target — the continuous value of the activated outcome species. Many real ML problems are framed as classification instead: "did this patient respond to the drug, yes or no?" The dataset Synthetic produces is perfectly fine input to a classifier — you just need to threshold the target.

This is a fully *user-side* transformation. There's no library helper for it, because the threshold choice is the scientific question you're trying to ask. For a complete worked example, see [Use Case 2 — Classification](use_cases.md#use-case-2-classification-responder-vs-non-responder). The short version:

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from synthetic import Builder, make_dataset_drug_response

vc = Builder.specify(degree_cascades=[3, 6, 15], random_seed=42)
X, y = make_dataset_drug_response(n=1000, cell_model=vc, target_specie='Oa')

# Threshold y into a binary class. The threshold encodes the question.
y_class = (y < np.median(y)).astype(int)

# Standard scikit-learn from here.
clf = RandomForestClassifier(random_state=42)
scores = cross_val_score(clf, X, y_class, cv=5, scoring='accuracy')
print(f"CV accuracy: {scores.mean():.3f}")
```

The threshold can be any of:

- a clinically meaningful cutoff (e.g., "50% reduction in `Oa` from baseline")
- a percentile of the simulated `y` distribution (e.g., bottom 25%)
- a value from prior experimental data

The rest of the pipeline is unchanged — same `X`, same classifiers, same metrics you would use on any binary classification problem.
