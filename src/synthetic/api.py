"""
High-level API for Synthetic library.

This module provides a simple, intuitive interface for generating synthetic drug
response datasets compatible with scikit-learn's `make_regression` format.

The API uses DegreeInteractionSpec for network topology and KineticParameterTuner
for biologically plausible kinetic parameters.

Example usage:
    from synthetic import Builder, make_dataset_drug_response

    # Create a virtual cell model
    vc = Builder.specify(degree_cascades=[1, 2, 5])

    # Generate sklearn-compatible dataset
    X, y = make_dataset_drug_response(n=1000, cell_model=vc, target_specie='Oa')
"""

from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
import pandas as pd

from synthetic import Solver

from .Specs.BaseSpec import BaseSpec
from .Specs.DegreeInteractionSpec import DegreeInteractionSpec
from .Specs.Drug import Drug
from .ModelBuilder import ModelBuilder
from .Solver.ScipySolver import ScipySolver
from .utils.kinetic_tuner import KineticParameterTuner


class VirtualCell:
    """
    High-level abstraction for a virtual cell model.

    Uses DegreeInteractionSpec for network topology and KineticParameterTuner
    for biologically plausible kinetic parameters.

    This class provides a simple API for creating and managing virtual cell
    models without needing to understand the underlying complexity of the
    Synthetic library's architecture.
    """

    def __init__(
        self,
        spec: Optional[Union[List[int], 'BaseSpec', ModelBuilder]] = None,
        name: str = "VirtualCell",
        random_seed: Optional[int] = None,
        auto_compile: bool = True,
        auto_drug: bool = True,
        drug_name: str = "D",
        drug_start_time: float = 5000.0,
        drug_value: float = 100.0,
        drug_regulation_type: str = "down",
        simulation_end: float = 10000.0,
        **kwargs,
    ):
        """
        Initialize a virtual cell model.

        Args:
            spec: Specification (List[int] for degree cascades, a BaseSpec instance, or ModelBuilder)
            name: Name for the virtual cell
            random_seed: Optional random seed for reproducibility
            auto_compile: Compile immediately after creation (default: True)
            auto_drug: Auto-generate a drug targeting spec-defined species (default: True)
            drug_name: Name for auto-generated drug (default: "D")
            drug_start_time: Time at which auto-drug becomes active (default: 5000.0)
            drug_value: Active concentration for auto-drug (default: 100.0)
            drug_regulation_type: Regulation type for auto-drug: "up" or "down" (default: "down")
            simulation_end: Simulation end time for make_dataset_drug_response (default: 10000.0)
            **kwargs: Additional parameters for backward compatibility or spec-specific options:
                - degree_cascades: List[int] (legacy)
                - feedback_density: float (default: 0.5)
        """

        # Handle legacy degree_cascades/spec parameter resolution
        actual_spec = spec
        if actual_spec is None:
            actual_spec = kwargs.get('degree_cascades')

        if actual_spec is None:
            raise ValueError("Must provide 'spec' (or 'degree_cascades')")

        # Set internal spec representation
        self._spec_instance: Optional[BaseSpec] = None
        self._model_instance: Optional[ModelBuilder] = None
        self._degree_cascades: Optional[List[int]] = None

        if isinstance(actual_spec, list):
            self._degree_cascades = actual_spec
        elif isinstance(actual_spec, BaseSpec):
            self._spec_instance = actual_spec
        elif isinstance(actual_spec, ModelBuilder):
            self._model_instance = actual_spec
        else:
            raise TypeError(f"Unsupported spec type: {type(actual_spec)}")

        self._name = name
        self._random_seed = random_seed
        self._feedback_density = kwargs.get('feedback_density', 0.5)
        self._auto_compile = auto_compile
        self._auto_drug = auto_drug
        self._auto_drug_name: Optional[str] = drug_name if auto_drug else None
        self._drug_start_time = drug_start_time
        self._auto_drug_value = drug_value
        self._auto_drug_regulation_type = drug_regulation_type
        self._simulation_end = simulation_end
        self._drugs: List[Tuple[Drug, Optional[float]]] = []
        self._spec: Optional[BaseSpec] = None
        self._model: Optional[ModelBuilder] = None
        self._tuner: Optional[KineticParameterTuner] = None
        self._solver: Optional['Solver'] = None
        self._compiled = False

        if auto_compile:
            self.compile()

    def add_drug(
        self,
        name: str,
        start_time: float = 500.0,
        default_value: float = 0.0,
        regulation: Optional[List[str]] = None,
        regulation_type: Optional[List[str]] = None,
        value: Optional[float] = None,
    ) -> 'VirtualCell':
        """
        Add a drug to the cell model.

        Drugs can only target degree 1 R species. The drug appears at start_time
        via a piecewise assignment rule and regulates the specified species.

        Args:
            name: Name of the drug
            start_time: Time at which drug becomes active (default: 500.0)
            default_value: Default value when drug is not active (default: 0.0)
            regulation: List of target species (must be degree 1 R species)
            regulation_type: List of regulation types ('up' or 'down')
            value: Optional override value for when drug is active

        Returns:
            self for method chaining
        """
        drug = Drug(
            name=name,
            start_time=start_time,
            default_value=default_value,
            regulation=regulation,
            regulation_type=regulation_type,
        )
        self._drugs.append((drug, value))
        return self

    def list_drugs(self) -> List[Dict[str, Any]]:
        """
        List all drugs in the system (both auto-generated and manually added).

        Returns:
            List of dictionaries with drug information:
            [{'name': 'D', 'targets': ['R1_1'], 'types': ['down'],
              'start_time': 5000.0, 'default_value': 0.0, 'is_auto': True}, ...]
        """
        drugs_info = []
        for drug, value in self._drugs:
            drugs_info.append({
                'name': drug.name,
                'targets': drug.regulation,
                'types': drug.regulation_type,
                'start_time': drug.start_time,
                'default_value': drug.default_value,
                'is_auto': drug.name == self._auto_drug_name
            })
        return drugs_info

    def _generate_auto_drug(self) -> None:
        """
        Generate auto-drug based on specification targets.
        """
        if self._spec is None:
            return

        targets = self._spec.get_auto_drug_targets()
        if not targets:
            return

        # Create and store auto-drug
        drug = Drug(
            name=self._auto_drug_name,
            start_time=self._drug_start_time,
            default_value=0.0,
            regulation=targets,
            regulation_type=[self._auto_drug_regulation_type] * len(targets),
        )
        self._drugs.append((drug, self._auto_drug_value))

    def compile(
        self,
        mean_range_species: Tuple[int, int] = (50, 150),
        rangeScale_params: Tuple[float, float] = (0.8, 1.2),
        rangeMultiplier_params: Tuple[float, float] = (0.9, 1.1),
        use_kinetic_tuner: bool = True,
        active_percentage_range: Tuple[float, float] = (0.3, 0.7),
        X_total_multiplier: float = 5.0,
        ki_val: float = 100.0,
        v_max_f_random_range: Tuple[float, float] = (5.0, 10.0),
        solver_type: str = 'scipy',
        **solver_kwargs,
    ) -> 'VirtualCell':
        """
        Compile the model (lazy initialization).

        Args:
            mean_range_species: Range for initial species values
            rangeScale_params: Range for parameter scaling
            rangeMultiplier_params: Range for parameter multiplier
            use_kinetic_tuner: Whether to use KineticParameterTuner (default: True)
            active_percentage_range: Target active percentage range for tuning
            X_total_multiplier: Multiplier for Km_b calculation
            ki_val: Constant Ki value for inhibitors
            v_max_f_random_range: Range for total forward Vmax
            solver_type: Type of solver ('scipy' or 'roadrunner')
            **solver_kwargs: Additional arguments for solver

        Returns:
            self for method chaining
        """
        # 1. Resolve Spec / Model
        if self._model_instance is not None:
            self._model = self._model_instance
        elif self._spec_instance is not None:
            self._spec = self._spec_instance
        else:
            # Legacy/default DegreeInteractionSpec
            self._spec = DegreeInteractionSpec(degree_cascades=self._degree_cascades)
            self._spec.generate_specifications(
                random_seed=self._random_seed,
                feedback_density=self._feedback_density,
            )

        # 2. Generate auto-drug if enabled
        if self._auto_drug:
            self._generate_auto_drug()

        # 3. Add drugs (auto-drug + any manually added)
        if self._spec is not None:
            for drug, value in self._drugs:
                self._spec.add_drug(drug, value)

        # 4. Generate model (if not already provided)
        if self._model is None:
            self._model = self._spec.generate_network(
                network_name=self._name,
                mean_range_species=mean_range_species,
                rangeScale_params=rangeScale_params,
                rangeMultiplier_params=rangeMultiplier_params,
                random_seed=self._random_seed,
            )

        if not self._model.pre_compiled:
            self._model.precompile()

        # 5. Apply kinetic tuning if requested
        if use_kinetic_tuner:
            self._tuner = KineticParameterTuner(self._model, random_seed=self._random_seed)
            updated_params = self._tuner.generate_parameters(
                active_percentage_range=active_percentage_range,
                X_total_multiplier=X_total_multiplier,
                ki_val=ki_val,
                v_max_f_random_range=v_max_f_random_range,
            )
            for param_name, value in updated_params.items():
                self._model.set_parameter(param_name, value)

        # 6. Apply auto-drug parameters (only if auto-drug enabled)
        if self._auto_drug_name:
            regulator_parameter_map = self._model.get_regulator_parameter_map()
            drug_params = regulator_parameter_map.get(self._auto_drug_name, {})
            for param_name in drug_params:
                self._model.set_parameter(param_name, self._auto_drug_value)

        # 7. Create solver
        self._solver = self._model.get_solver(solver_type=solver_type, **solver_kwargs)

        self._compiled = True
        return self

    def get_species_names(self, exclude_drugs: bool = True) -> List[str]:
        """
        Get list of species names.

        Args:
            exclude_drugs: Whether to exclude drug species (default: True)

        Returns:
            List of species names
        """
        if not self._compiled:
            raise ValueError("Model must be compiled first. Call compile().")
        return list(self._model.get_state_variables().keys())

    def get_initial_values(self, exclude_drugs: bool = True) -> Dict[str, float]:
        """
        Get initial species values from the compiled model.

        Args:
            exclude_drugs: Whether to exclude drug species (default: True)

        Returns:
            Dictionary mapping species names to initial values
        """
        if not self._compiled:
            raise ValueError("Model must be compiled first. Call compile().")

        initial_values = self._model.get_state_variables().copy()

        if exclude_drugs:
            # Remove drugs from initial values (they are in model.variables)
            for drug_name in [d[0].name for d in self._drugs]:
                initial_values.pop(drug_name, None)

        return initial_values

    def get_target_concentrations(self) -> Dict[str, float]:
        """
        Get target active concentrations from the kinetic tuner.

        Returns an empty dict if kinetic tuning was not used during compilation.

        Returns:
            Dictionary mapping species names to target concentrations
        """
        if not self._compiled:
            raise ValueError("Model must be compiled first. Call compile().")
        if self._tuner is None:
            return {}
        return self._tuner.get_target_concentrations()

    @property
    def spec(self) -> Optional[BaseSpec]:
        """
        Access to underlying specification.

        Returns the BaseSpec object. Raises ValueError if model
        has not been compiled and was not initialized from a ModelBuilder.

        Returns:
            BaseSpec object or None
        """
        if self._spec is None and self._model is None:
            raise ValueError("Model must be compiled first. Call compile().")
        return self._spec

    @property
    def model(self) -> ModelBuilder:
        """
        Access to underlying model.

        Returns the ModelBuilder object. Raises ValueError if model
        has not been compiled.

        Returns:
            ModelBuilder object
        """
        if self._model is None:
            raise ValueError("Model must be compiled first. Call compile().")
        return self._model

    @property
    def tuner(self) -> KineticParameterTuner:
        """
        Access to underlying kinetic tuner.

        Returns the KineticParameterTuner object, or None if kinetic
        tuning was not used during compilation.

        Returns:
            KineticParameterTuner object or None
        """
        return self._tuner

    @property
    def solver(self) -> 'Solver':
        """
        Access to underlying solver.

        Returns the Solver object. Raises ValueError if model
        has not been compiled.

        Returns:
            Solver object
        """
        if self._solver is None:
            raise ValueError("Model must be compiled first. Call compile().")
        return self._solver


class _RemoteModelProxy:
    """Lightweight proxy providing ModelBuilder-like parameter access backed by remote data."""

    def __init__(self, parameters: Dict[str, float]):
        self._parameters = parameters

    def get_parameters(self) -> Dict[str, float]:
        return self._parameters.copy()


class RemoteCell:
    """
    High-level abstraction for a remote cell model accessed via HTTP.

    Mirrors the VirtualCell interface but backed by an HTTP endpoint
    instead of a locally built model. Uses HTTPSolver internally to
    fetch states and parameters from the remote server.

    Example:
        rc = RemoteCell("http://localhost:8000/simulate")
        rc.compile()
        X, y = make_dataset_drug_response(
            n=100, cell_model=rc, solver_type='http',
            perturbation_type='lognormal',
        )
    """

    def __init__(
        self,
        endpoint: str,
        simulation_end: float = 10000.0,
        drug_names: Optional[List[str]] = None,
        timeout: float = 300.0,
    ):
        """
        Initialize a remote cell model.

        Args:
            endpoint: HTTP endpoint URL (e.g., "http://localhost:8000/simulate")
            simulation_end: Simulation end time for make_dataset_drug_response (default: 10000.0)
            drug_names: Names of drug species on the server to exclude from features
            timeout: Request timeout in seconds (default: 300.0)
        """
        self._endpoint = endpoint
        self._simulation_end = simulation_end
        self._drug_names = drug_names or []
        self._timeout = timeout
        self._compiled = False
        self._drugs: List[Tuple] = []  # Compatible with VirtualCell interface

        # Internal solver and cached remote data
        self._solver = None
        self._remote_states: Optional[Dict[str, float]] = None
        self._remote_parameters: Optional[Dict[str, float]] = None

    def compile(self) -> 'RemoteCell':
        """
        Connect to the HTTP endpoint and fetch model metadata.

        Validates the endpoint is reachable, then fetches default states
        and parameters from the server's /states and /parameters endpoints.

        Returns:
            self for method chaining

        Raises:
            ValueError: If the endpoint is unreachable
        """
        from .Solver.HTTPSolver import HTTPSolver

        self._solver = HTTPSolver()
        self._solver.compile(self._endpoint, timeout=self._timeout)
        self._remote_states = self._solver.get_state_defaults()
        self._remote_parameters = self._solver.get_parameter_defaults()
        self._compiled = True
        return self

    def get_initial_values(self, exclude_drugs: bool = True) -> Dict[str, float]:
        """
        Get initial species values from the remote server.

        Args:
            exclude_drugs: Whether to exclude drug species (default: True)

        Returns:
            Dictionary mapping species names to initial values

        Raises:
            ValueError: If model is not compiled
        """
        if not self._compiled:
            raise ValueError("Model must be compiled first. Call compile().")
        values = self._remote_states.copy()
        if exclude_drugs:
            for drug_name in self._drug_names:
                values.pop(drug_name, None)
        return values

    @property
    def spec(self):
        """Not available for remote models. Returns None."""
        return None

    @property
    def model(self) -> _RemoteModelProxy:
        """
        Access a lightweight model proxy providing get_parameters().

        Returns:
            _RemoteModelProxy instance

        Raises:
            ValueError: If model is not compiled
        """
        if not self._compiled:
            raise ValueError("Model must be compiled first. Call compile().")
        return _RemoteModelProxy(self._remote_parameters)

    @property
    def solver(self):
        """Access the underlying compiled HTTPSolver."""
        return self._solver


class Builder:
    """
    Factory for creating virtual cell models.

    Provides a static method for creating VirtualCell instances with
    a clean, fluent API.
    """

    @staticmethod
    def specify(
        spec: Optional[Union[List[int], 'BaseSpec', ModelBuilder]] = None,
        name: str = "VirtualCell",
        random_seed: Optional[int] = None,
        auto_compile: bool = True,
        auto_drug: bool = True,
        drug_name: str = "D",
        drug_start_time: float = 5000.0,
        drug_value: float = 100.0,
        drug_regulation_type: str = "down",
        simulation_end: float = 10000.0,
        **kwargs,
    ) -> VirtualCell:
        """
        Create a virtual cell specification.

        Args:
            spec: Specification (List[int] for degree cascades, a BaseSpec instance, or ModelBuilder)
            name: Name for the virtual cell
            random_seed: Optional random seed for reproducibility
            auto_compile: Compile immediately after creation (default: True)
            auto_drug: Auto-generate a drug targeting spec-defined species (default: True)
            drug_name: Name for auto-generated drug (default: "D")
            drug_start_time: Time at which auto-drug becomes active (default: 5000.0)
            drug_value: Active concentration for auto-drug (default: 100.0)
            drug_regulation_type: Regulation type for auto-drug: "up" or "down" (default: "down")
            simulation_end: Simulation end time for make_dataset_drug_response (default: 10000.0)
            **kwargs: Additional parameters for backward compatibility or spec-specific options:
                - degree_cascades: List[int] (legacy)
                - feedback_density: float (default: 0.5)

        Returns:
            VirtualCell instance (compiled if auto_compile=True)
        """
        vc = VirtualCell(
            spec=spec,
            name=name,
            random_seed=random_seed,
            auto_compile=auto_compile,
            auto_drug=auto_drug,
            drug_name=drug_name,
            drug_start_time=drug_start_time,
            drug_value=drug_value,
            drug_regulation_type=drug_regulation_type,
            simulation_end=simulation_end,
            **kwargs,
        )
        return vc

    @staticmethod
    def from_degree_cascades(
        cascades: List[int],
        feedback_density: float = 0.5,
        name: str = "VirtualCell",
        random_seed: Optional[int] = None,
        auto_compile: bool = True,
        **kwargs
    ) -> VirtualCell:
        """
        Create a virtual cell from degree cascades.

        Args:
            cascades: List of cascade counts per degree
            feedback_density: Proportion of feedback connections (0-1)
            name: Name for the virtual cell
            random_seed: Optional random seed for reproducibility
            auto_compile: Compile immediately after creation (default: True)
            **kwargs: Additional parameters for VirtualCell

        Returns:
            VirtualCell instance (compiled if auto_compile=True)
        """
        spec = DegreeInteractionSpec(degree_cascades=cascades)
        return VirtualCell(
            spec=spec,
            feedback_density=feedback_density,
            name=name,
            random_seed=random_seed,
            auto_compile=auto_compile,
            **kwargs
        )

    @staticmethod
    def from_endpoint(
        endpoint: str,
        simulation_end: float = 10000.0,
        drug_names: Optional[List[str]] = None,
        timeout: float = 300.0,
        auto_compile: bool = True,
    ) -> 'RemoteCell':
        """
        Create a remote cell model from an HTTP endpoint.

        The server must implement the HTTP Solver API:
        - POST /simulate: Run simulation with optional state/parameter overrides
        - GET /states: Return default state values
        - GET /parameters: Return default parameter values

        Args:
            endpoint: HTTP endpoint URL (e.g., "http://localhost:8000/simulate")
            simulation_end: Simulation end time for make_dataset_drug_response (default: 10000.0)
            drug_names: Names of drug species on the server to exclude from features
            timeout: Request timeout in seconds (default: 300.0)
            auto_compile: Validate endpoint and fetch metadata immediately (default: True)

        Returns:
            RemoteCell instance (compiled if auto_compile=True)
        """
        rc = RemoteCell(
            endpoint=endpoint,
            simulation_end=simulation_end,
            drug_names=drug_names,
            timeout=timeout,
        )
        if auto_compile:
            rc.compile()
        return rc


def make_dataset_drug_response(
    n: int,
    cell_model: Union[VirtualCell, RemoteCell, ModelBuilder, Solver],
    target_specie: str = 'Oa',
    perturbation_type: str = 'conserve_rules',
    perturbation_params: Optional[Dict[str, Any]] = None,
    parameter_values: Optional[Dict[str, float]] = None,
    param_perturbation_type: str = 'lognormal',
    param_perturbation_params: Optional[Dict[str, Any]] = None,
    simulation_params: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    param_seed: Optional[int] = None,
    solver_type: Optional[str] = None,
    jit: bool = True,
    verbose: bool = False,
    n_cores: int = 1,
    require_all_successful: bool = False,
    return_details: bool = True,
    capture_all_species: bool = True,
    exclude_outcome_from_features: bool = True,
    exclude_activated_from_features: bool = True,
    as_pandas: bool = True,
    return_timecourse: bool = False,
) -> Union[Tuple[pd.DataFrame, pd.Series], Tuple[np.ndarray, np.ndarray], Dict[str, Any]]:
    """
    Generate synthetic drug response dataset.

    Creates a dataset compatible with scikit-learn's make_regression format.
    The feature matrix X contains species concentrations, and the target
    vector y contains the outcome values.

    This function is the primary hinge between Model Building and Data Generation,
    supporting VirtualCell, ModelBuilder, and Solver instances.

    Args:
        n: Number of samples to generate
        cell_model: A compiled model (VirtualCell, ModelBuilder, or Solver)
        target_specie: Name of the outcome species to use as target (default: 'Oa')
        perturbation_type: Type of initial value perturbation
        perturbation_params: Parameters for perturbation distribution
        parameter_values: Dictionary of kinetic parameter values to perturb (optional)
        param_perturbation_type: Type of kinetic parameter perturbation
        param_perturbation_params: Parameters for kinetic parameter perturbation (optional)
        simulation_params: Dictionary with 'start', 'end', 'points' keys
        seed: Random seed for reproducibility
        param_seed: Random seed for parameter perturbations
        solver_type: Optional override for solver type if cell_model is a ModelBuilder
        jit: Whether to use JIT compilation (only for scipy solver)
        verbose: Whether to show progress bar
        n_cores: Number of CPU cores for parallel simulation
        require_all_successful: Whether to require all samples to succeed
        return_details: If True, returns extended data structure
        as_pandas: If True (default), returns pandas DataFrame/Series
        return_timecourse: If True, returns full simulation data

    Returns:
        Dataset in requested format (X, y) or details dictionary
    """
    from .utils.data_generation_helpers import make_data
    from .ModelBuilder import ModelBuilder
    from .Solver.Solver import Solver

    # 1. Extract essentials from the cell_model
    if isinstance(cell_model, VirtualCell):
        if solver_type is not None:
            solver = cell_model.model.get_solver(solver_type=solver_type, jit=jit)
        else:
            solver = cell_model.solver
        if parameter_values is None:
            parameter_values = cell_model.model.get_parameters()
        initial_values = cell_model.get_initial_values(exclude_drugs=True)
        simulation_end = cell_model._simulation_end
        spec = cell_model.spec
    elif isinstance(cell_model, RemoteCell):
        if not cell_model._compiled:
            cell_model.compile()
        solver = cell_model.solver
        if parameter_values is None:
            parameter_values = cell_model.model.get_parameters()
        initial_values = cell_model.get_initial_values(exclude_drugs=True)
        simulation_end = cell_model._simulation_end
        spec = None
    elif isinstance(cell_model, ModelBuilder):
        if not cell_model.pre_compiled:
            cell_model.precompile()
        solver = cell_model.get_solver(solver_type=solver_type or 'scipy', jit=jit)
        if parameter_values is None:
            parameter_values = cell_model.get_parameters()
        initial_values = cell_model.get_state_variables()
        simulation_end = simulation_params.get('end', 10000) if simulation_params else 10000
        spec = None  # Spec not available for raw ModelBuilder
    elif isinstance(cell_model, Solver):
        solver = cell_model
        try:
            if parameter_values is None:
                parameter_values = solver.get_parameter_defaults()
            initial_values = solver.get_state_defaults()
        except NotImplementedError:
            raise ValueError(
                "Provided Solver does not support metadata extraction. "
                "Please provide initial_values and parameter_values explicitly."
            )
        simulation_end = simulation_params.get('end', 10000) if simulation_params else 10000
        spec = None
    else:
        raise TypeError(f"Unsupported cell_model type: {type(cell_model)}")

    # 2. Set default simulation parameters
    if simulation_params is None:
        simulation_params = {'start': 0, 'end': simulation_end, 'points': 101}

    # 3. Handle perturbation parameters
    if perturbation_params is None:
        perturbation_params = {'shape': 0.5, 'base_shape': 0.01, 'max_shape': 0.5}

    if param_perturbation_params is None:
        param_perturbation_params = {'shape': 0.1}

    if perturbation_type == 'conserve_rules' and 'model_spec' not in perturbation_params and 'species_range' not in perturbation_params:
        if spec is not None:
            perturbation_params = perturbation_params.copy()
            perturbation_params['model_spec'] = spec
        else:
            # If no spec, we can't use conserve_rules easily unless user provided species_range
            raise ValueError(
                "conserve_rules perturbation requires a Spec object or explicit 'species_range'. "
                "Either pass a VirtualCell, or change perturbation_type (e.g. to 'lognormal')."
            )

    # 4. Prepare initial values for perturbation (exclude activated forms and outcome base)
    # This logic matches previous implementation for VirtualCell
    initial_values = {k: v for k, v in initial_values.items() if not k.endswith('a')}
    initial_values.pop('O', None)

    # 5. Generate data using make_data
    result = make_data(
        initial_values=initial_values,
        perturbation_type=perturbation_type,
        perturbation_params=perturbation_params,
        n_samples=n,
        solver=solver,
        parameter_values=parameter_values,
        param_perturbation_type=param_perturbation_type,
        param_perturbation_params=param_perturbation_params,
        simulation_params=simulation_params,
        target_method="fold_change_drug",
        seed=seed,
        param_seed=param_seed,
        require_all_successful=require_all_successful,
        return_details=return_details,
        capture_all_species=capture_all_species,
        outcome_var=target_specie,
        verbose=verbose,
        n_cores=n_cores,
    )

    if return_details:
        X = result['features'].copy()
        if exclude_activated_from_features:
            X = X[[col for col in X.columns if not col.endswith('a')]]
        if exclude_outcome_from_features:
            X = X.drop(columns=['O'], errors='ignore')

        y = pd.Series(result['targets'].iloc[:, 0].values, name=target_specie, index=result['targets'].index, dtype=float)

        if not as_pandas:
            X = X.values.astype(np.float64)
            y = y.values.ravel().astype(np.float64)

        if return_timecourse:
            return {
                'X': X,
                'y': y,
                'timecourse': result['timecourse'],
                'parameters': result['parameters'],
                'metadata': result['metadata']
            }
        return X, y
    else:
        raise NotImplementedError("return_details=False is not implemented in this version.")


__all__ = ['VirtualCell', 'RemoteCell', 'Builder', 'make_dataset_drug_response']
