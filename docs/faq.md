# FAQ & Troubleshooting

## Installation

??? question "How do I install optional solver backends?"

    Install extras to add solver backends:

    ```bash
    pip install synthetic-models[roadrunner]
    ```

    See [Installation](index.md#installation) for the full list of available extras (RoadRunner, plotting, scikit-learn).

??? question "Do I need to install all optional dependencies?"

    No. The base install includes `ScipySolver` which is sufficient for most use cases. Install extras only when you need them:

    - **RoadRunner** — for complex SBML models or when you need mature SBML support
    - **Plotting** — for visualization with matplotlib/seaborn
    - **Scikit-learn** — for ML pipeline integration

??? question "How do I install from source?"

    Clone the repo and install with `uv`:

    ```bash
    git clone https://github.com/synthetic-models/synthetic.git
    cd Synthetic
    uv sync
    ```

    See [Build from source](index.md#build-from-source) for details.

## Model Building

??? question "Why does `get_parameters()` or `get_state_variables()` raise an error?"

    You must call `model.precompile()` before accessing parameters or states. Precompile caches the state and parameter dictionaries and enables `set_parameter()` and `set_state()`.

    ```python
    model.precompile()  # Required!
    params = model.get_parameters()
    ```

    When using the high-level API (`Builder.specify()`), this is handled automatically.

??? question "What does 'Parameter X not found in the model' mean?"

    This error occurs when trying to set a parameter that doesn't exist. Common causes:

    1. **You haven't called `precompile()`** — parameter names are only available after precompilation.
    2. **Wrong parameter name** — parameter names include the reaction index (e.g., `Km_J0`, `Vmax_J1`, not just `Km`). Use `model.get_parameters()` to list all valid names.

??? question "How do I debug my model?"

    Print the Antimony model string to inspect all species, reactions, and parameters:

    ```python
    print(model.get_antimony_model())
    ```

    You can also use `model.head()` for a summary overview.

??? question "What are the `&` and `?` prefixes in archtype definitions?"

    These are placeholder prefixes used in `ReactionArchtype`:

    - `&` — main reactants/products and regulators (e.g., `&S`, `&P`, `&A0`)
    - `?` — reverse-only regulators used in reversible reactions (e.g., `?A0`)

    When a `Reaction` is created, these placeholders are replaced with actual species names. See [Model Building](model_building.md#layer-1-reactionarchtype-the-template) for details.

## Simulation

??? question "Why are my simulation results all zeros or NaN?"

    Common causes:

    1. **Parameters not tuned** — random parameter values can produce unresponsive dynamics. Use kinetic tuning: `vc.compile(use_kinetic_tuner=True)` or `Builder.specify()` which applies tuning by default.
    2. **Simulation too short** — the network may not have reached steady state. Increase `stop` time.
    3. **Initial conditions are zero** — check that species have non-zero initial concentrations.

    See [Kinetic Parameter Tuning](advanced_features.md#kinetic-parameter-tuning) for the tuning guide.

??? question "Which solver should I use?"

    | Scenario | Solver |
    |----------|--------|
    | Default / batch simulations | `ScipySolver` |
    | Need JIT speed for 1000+ samples | `ScipySolver` with `jit=True` |
    | Complex SBML models | `RoadrunnerSolver` |
    | Single robust simulations | `RoadrunnerSolver` |
    | Remote/distributed computing | `HTTPSolver` |

    See [Solver Selection Guide](solvers_and_simulation.md#solver-selection-guide) for the full comparison.

??? question "Why is JIT compilation slow on first run?"

    `ScipySolver` uses Numba for JIT compilation. The first simulation compiles the ODE function, which can take several seconds. Subsequent simulations with the same model are fast. This is a one-time cost per model.

??? question "Why does RoadrunnerSolver fail with 'expected SBML'?"

    `RoadrunnerSolver` requires SBML format, not Antimony:

    ```python
    # Wrong:
    solver.compile(model.get_antimony_model())

    # Correct:
    solver.compile(model.get_sbml_model())
    ```

    `ScipySolver` accepts Antimony. See [Solvers & Simulation](solvers_and_simulation.md) for both approaches.

## Data Generation

??? question "How do I get reproducible datasets?"

    Set both `random_seed` and `param_seed`:

    ```python
    vc = Builder.specify(degree_cascades=[3, 6, 15], random_seed=42)

    X, y = make_dataset_drug_response(
        n=1000,
        cell_model=vc,
        random_seed=42,      # For initial condition perturbation
        param_seed=42,       # For kinetic parameter perturbation
    )
    ```

    !!! note
        The `random_seed` in `Builder.specify()` controls network generation. The seeds in `make_dataset_drug_response()` control the data perturbation. Use the same seeds for reproducibility.

??? question "What does 'conserve_rules' perturbation do?"

    The default `conserve_rules` perturbation maintains total species mass. For each species pair (e.g., `R1_1` + `R1_1a`), the total concentration is conserved while the ratio between inactive and active forms is perturbed. This produces biologically plausible initial conditions.

    Use the `perturbation_params` to control the shape of the perturbation:

    ```python
    X, y = make_dataset_drug_response(
        n=1000,
        cell_model=vc,
        perturbation_type='conserve_rules',
        perturbation_params={'shape': 0.5, 'base_shape': 0.01, 'max_shape': 0.5},
    )
    ```

    See [Perturbation Strategies](data_generation.md#perturbation-strategies) for all available types.

??? question "Why is my dataset success rate low?"

    A low success rate means many simulations failed (e.g., numerical instability). Try:

    1. **Enable kinetic tuning** — produces parameters that yield stable dynamics
    2. **Reduce perturbation magnitude** — use a smaller `rsd` value
    3. **Increase simulation points** — more timepoints can help the solver converge
    4. **Switch solver** — Different solvers have different stability properties. Try `RoadrunnerSolver` if `ScipySolver` fails.

    Check the success rate via the metadata:

    ```python
    result = make_dataset_drug_response(
        n=1000,
        cell_model=vc,
        return_timecourse=True,
    )
    print(f"Success rate: {result['metadata']['success_rate']:.2%}")
    ```
