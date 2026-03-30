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
        degree_cascades: List[int],
        name: str = "VirtualCell",
        random_seed: Optional[int] = None,
        feedback_density: float = 0.5,
        auto_compile: bool = True,
        auto_drug: bool = True,
        drug_name: str = "D",
        drug_start_time: float = 5000.0,
        drug_value: float = 100.0,
        drug_regulation_type: str = "down",
        simulation_end: float = 10000.0,
    ):
        """
        Initialize a virtual cell model.

        Args:
            degree_cascades: List of cascade counts per degree, e.g., [1, 2, 5]
            name: Name for the virtual cell
            random_seed: Optional random seed for reproducibility
            feedback_density: Proportion of feedback connections (0-1)
            auto_compile: Compile immediately after creation (default: True)
            auto_drug: Auto-generate a drug targeting degree 1 R species (default: True)
            drug_name: Name for auto-generated drug (default: "D")
            drug_start_time: Time at which auto-drug becomes active (default: 5000.0)
            drug_value: Active concentration for auto-drug (default: 100.0)
            drug_regulation_type: Regulation type for auto-drug: "up" or "down" (default: "down")
            simulation_end: Simulation end time for make_dataset_drug_response (default: 10000.0)
        """
        self._degree_cascades = degree_cascades
        self._name = name
        self._random_seed = random_seed
        self._feedback_density = feedback_density
        self._auto_compile = auto_compile
        self._auto_drug = auto_drug
        self._auto_drug_name: Optional[str] = drug_name if auto_drug else None
        self._drug_start_time = drug_start_time
        self._auto_drug_value = drug_value
        self._auto_drug_regulation_type = drug_regulation_type
        self._simulation_end = simulation_end
        self._drugs: List[Tuple[Drug, Optional[float]]] = []
        self._spec: Optional[DegreeInteractionSpec] = None
        self._model: Optional[ModelBuilder] = None
        self._tuner: Optional[KineticParameterTuner] = None
        self._compiled = False

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
        Generate auto-drug based on degree_cascades[0].

        Creates a drug targeting all degree 1 R species (R1_1, R1_2, etc.)
        with count based on degree_cascades[0].
        """
        cascade_count = self._degree_cascades[0]
        targets = []

        # Generate targets: R1_1, R1_2, ... based on cascade_count
        for i in range(cascade_count):
            targets.append(f"R1_{i + 1}")

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
    ) -> 'VirtualCell':
        """
        Compile the model (lazy initialization).

        Creates the underlying DegreeInteractionSpec, ModelBuilder, and optionally
        applies KineticParameterTuner for biologically plausible parameters.

        If auto_drug is enabled, automatically generates a drug targeting degree 1
        R species and applies its parameters to the model.

        Args:
            mean_range_species: Range for initial species values
            rangeScale_params: Range for parameter scaling
            rangeMultiplier_params: Range for parameter multiplier
            use_kinetic_tuner: Whether to use KineticParameterTuner (default: True)
            active_percentage_range: Target active percentage range for tuning
            X_total_multiplier: Multiplier for Km_b calculation
            ki_val: Constant Ki value for inhibitors
            v_max_f_random_range: Range for total forward Vmax

        Returns:
            self for method chaining
        """
        # Generate auto-drug if enabled
        if self._auto_drug:
            self._generate_auto_drug()

        # Create specification
        self._spec = DegreeInteractionSpec(degree_cascades=self._degree_cascades)
        self._spec.generate_specifications(
            random_seed=self._random_seed,
            feedback_density=self._feedback_density,
        )

        # Add drugs (auto-drug + any manually added)
        for drug, value in self._drugs:
            self._spec.add_drug(drug, value)

        # Generate model
        self._model = self._spec.generate_network(
            network_name=self._name,
            mean_range_species=mean_range_species,
            rangeScale_params=rangeScale_params,
            rangeMultiplier_params=rangeMultiplier_params,
            random_seed=self._random_seed,
        )
        self._model.precompile()

        # Apply kinetic tuning if requested
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

        # Apply auto-drug parameters using regulator_parameter_map
        if self._auto_drug_name:
            regulator_parameter_map = self._model.get_regulator_parameter_map()
            drug_params = regulator_parameter_map.get(self._auto_drug_name, {})
            for param_name in drug_params:
                self._model.set_parameter(param_name, self._auto_drug_value)

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
        return self._model.get_state_variables().keys()

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
    def spec(self) -> DegreeInteractionSpec:
        """
        Access to underlying specification.

        Returns the DegreeInteractionSpec object. Raises ValueError if model
        has not been compiled.

        Returns:
            DegreeInteractionSpec object
        """
        if self._spec is None:
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


class Builder:
    """
    Factory for creating virtual cell models.

    Provides a static method for creating VirtualCell instances with
    a clean, fluent API.
    """

    @staticmethod
    def specify(
        degree_cascades: List[int],
        name: str = "VirtualCell",
        random_seed: Optional[int] = None,
        feedback_density: float = 1,
        auto_compile: bool = True,
        auto_drug: bool = True,
        drug_name: str = "D",
        drug_start_time: float = 5000.0,
        drug_value: float = 100.0,
        drug_regulation_type: str = "down",
        simulation_end: float = 10000.0,
    ) -> VirtualCell:
        """
        Create a virtual cell specification.

        Args:
            degree_cascades: List of cascade counts per degree, e.g., [1, 2, 5]
            name: Name for the virtual cell
            random_seed: Optional random seed for reproducibility
            feedback_density: Proportion of feedback connections (0-1)
            auto_compile: Compile immediately after creation (default: True)
            auto_drug: Auto-generate a drug targeting degree 1 R species (default: True)
            drug_name: Name for auto-generated drug (default: "D")
            drug_start_time: Time at which auto-drug becomes active (default: 5000.0)
            drug_value: Active concentration for auto-drug (default: 100.0)
            drug_regulation_type: Regulation type for auto-drug: "up" or "down" (default: "down")
            simulation_end: Simulation end time for make_dataset_drug_response (default: 10000.0)

        Returns:
            VirtualCell instance (compiled if auto_compile=True)
        """
        vc = VirtualCell(
            degree_cascades=degree_cascades,
            name=name,
            random_seed=random_seed,
            feedback_density=feedback_density,
            auto_compile=auto_compile,
            auto_drug=auto_drug,
            drug_name=drug_name,
            drug_start_time=drug_start_time,
            drug_value=drug_value,
            drug_regulation_type=drug_regulation_type,
            simulation_end=simulation_end,
        )
        if auto_compile:
            vc.compile()
        return vc


def make_dataset_drug_response(
    n: int,
    cell_model: VirtualCell,
    target_specie: str = 'Oa',
    perturbation_type: str = 'conserve_rules',
    perturbation_params: Optional[Dict[str, Any]] = None,
    parameter_values: Optional[Dict[str, float]] = None,
    param_perturbation_type: str = 'lognormal',
    param_perturbation_params: Optional[Dict[str, Any]] = None,
    simulation_params: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    param_seed: Optional[int] = None,
    solver_type: str = 'scipy',
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

    Drugs are NOT included in the X features - they affect the simulation
    but are not part of the feature matrix.

    Args:
        n: Number of samples to generate
        cell_model: VirtualCell instance (must be compiled)
        target_specie: Name of the outcome species to use as target (default: 'Oa')
        perturbation_type: Type of initial value perturbation ('uniform', 'gaussian', 'lognormal', 'lhs')
        perturbation_params: Parameters for perturbation distribution
        parameter_values: Dictionary of kinetic parameter values to perturb (optional)
        param_perturbation_type: Type of kinetic parameter perturbation ('none', 'uniform', 'gaussian', 'lognormal', 'lhs')
        param_perturbation_params: Parameters for kinetic parameter perturbation (optional)
        simulation_params: Dictionary with 'start', 'end', 'points' keys
        seed: Random seed for reproducibility (initial value perturbations)
        param_seed: Random seed for parameter perturbations (uses seed if not provided)
        solver_type: Type of solver ('scipy' or 'roadrunner')
        jit: Whether to use JIT compilation (only for scipy solver)
        verbose: Whether to show progress bar
        require_all_successful: Whether to require all samples to succeed (default: False)
        return_details: If True, returns extended data structure with intermediate datasets (default: True)
        capture_all_species: If True, captures timecourses for all species in returned data.
        as_pandas: If True (default), returns pandas DataFrame for X and Series for y with feature names.
                  If False, returns numpy arrays for X and y.
        return_timecourse: If True, returns dictionary with X, y, timecourse, parameters, and metadata (default: False)

    Returns:
        If return_timecourse=False:
            If as_pandas=True: Tuple of (X, y) where X is pandas DataFrame with feature names
                and y is pandas Series
            If as_pandas=False: Tuple of (X, y) where X is numpy array (n_samples, n_features)
                and y is numpy array (n_samples,)
        If return_timecourse=True: Dictionary with keys:
            - 'X': Feature matrix (basal state values)
            - 'y': Target values
            - 'timecourse': Timecourse simulation data (DataFrame with numpy arrays)
            - 'parameters': Kinetic parameters dataframe (None if not perturbed)
            - 'metadata': Dictionary with metadata about generation process

    Raises:
        ValueError: If cell_model is not compiled or has invalid parameters
    """
    from .utils.data_generation_helpers import make_data

    if not cell_model._compiled:
        raise ValueError("cell_model must be compiled. Call cell_model.compile() first.")

    # Set default simulation parameters (use cell_model's simulation_end)
    if simulation_params is None:
        simulation_params = {'start': 0, 'end': cell_model._simulation_end, 'points': 101}
    
    # Validate simulation parameters
    if 'start' not in simulation_params or 'end' not in simulation_params or 'points' not in simulation_params:
        raise ValueError('simulation_params must contain "start", "end", and "points" keys')

    # Set default perturbation parameters for conserve_rules
    if perturbation_params is None:
        perturbation_params = {'shape': 0.5, 'base_shape': 0.01, 'max_shape': 0.5}
    
    # Auto-extract parameter values from cell model if not provided
    if parameter_values is None:
        parameter_values = cell_model.model.get_parameters()
    
    # Set default parameter perturbation parameters
    if param_perturbation_params is None:
        param_perturbation_params = {'shape': 0.1}
    
    # Pass model_spec to conserve_rules for auto-generation of species ranges
    # Note: This is optional - if model_spec is not provided, conserve_rules will use
    # the species_range dictionary if provided, or auto-generate from initial_values
    if perturbation_type == 'conserve_rules' and 'model_spec' not in perturbation_params and 'species_range' not in perturbation_params:
        perturbation_params = perturbation_params.copy()
        perturbation_params['model_spec'] = cell_model.spec

    # Get initial values (excluding drugs)
    initial_values = cell_model.get_initial_values(exclude_drugs=True)
    # Exclude any species with 'a' at the end of their name (outcome species and activated species)
    # Since they should not be perturbed initially
    initial_values = {k: v for k, v in initial_values.items() if not k.endswith('a')}
    initial_values.pop('O')

    # Create and compile solver
    if solver_type == 'scipy':
        solver = ScipySolver()
        antimony_str = cell_model.model.get_antimony_model()
        solver.compile(antimony_str, jit=jit)
    elif solver_type == 'roadrunner':
        from .Solver.RoadrunnerSolver import RoadrunnerSolver
        solver = RoadrunnerSolver()
        sbml_str = cell_model.model.get_sbml_model()
        solver.compile(sbml_str)
    else:
        raise ValueError(f"Unsupported solver_type: {solver_type}. Use 'scipy' or 'roadrunner'")

    # Generate feature and target data using make_data
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
        # Return extended data structure
        # Use 'features' (perturbed initial values) as X, not 'basal_data' (pre-drug simulation state)
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


__all__ = ['VirtualCell', 'Builder', 'make_dataset_drug_response']
