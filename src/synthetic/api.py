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

from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd

from .Specs.DegreeInteractionSpec import DegreeInteractionSpec
from .Specs.Drug import Drug
from .ModelBuilder import ModelBuilder
from .Solver.ScipySolver import ScipySolver
from .Solver.RoadrunnerSolver import RoadrunnerSolver
from .utils.kinetic_tuner import KineticParameterTuner
from .utils.make_feature_data import make_feature_data
from .utils.make_target_data import make_target_data


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
    ):
        """
        Initialize a virtual cell model.

        Args:
            degree_cascades: List of cascade counts per degree, e.g., [1, 2, 5]
            name: Name for the virtual cell
            random_seed: Optional random seed for reproducibility
            feedback_density: Proportion of feedback connections (0-1)
        """
        self._degree_cascades = degree_cascades
        self._name = name
        self._random_seed = random_seed
        self._feedback_density = feedback_density
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
        # Create specification
        self._spec = DegreeInteractionSpec(degree_cascades=self._degree_cascades)
        self._spec.generate_specifications(
            random_seed=self._random_seed,
            feedback_density=self._feedback_density,
        )

        # Add drugs
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
        feedback_density: float = 0.5,
    ) -> VirtualCell:
        """
        Create a virtual cell specification.

        Args:
            degree_cascades: List of cascade counts per degree, e.g., [1, 2, 5]
            name: Name for the virtual cell
            random_seed: Optional random seed for reproducibility
            feedback_density: Proportion of feedback connections (0-1)

        Returns:
            VirtualCell instance (not yet compiled)
        """
        return VirtualCell(
            degree_cascades=degree_cascades,
            name=name,
            random_seed=random_seed,
            feedback_density=feedback_density,
        )


def make_dataset_drug_response(
    n: int,
    cell_model: VirtualCell,
    target_specie: str = 'Oa',
    perturbation_type: str = 'lognormal',
    perturbation_params: Optional[Dict[str, Any]] = None,
    simulation_params: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    solver_type: str = 'scipy',
    jit: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
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
        perturbation_type: Type of perturbation ('uniform', 'gaussian', 'lognormal', 'lhs')
        perturbation_params: Parameters for perturbation distribution
        simulation_params: Dictionary with 'start', 'end', 'points' keys
        seed: Random seed for reproducibility
        solver_type: Type of solver ('scipy' or 'roadrunner')
        jit: Whether to use JIT compilation (only for scipy solver)
        verbose: Whether to show progress bar

    Returns:
        Tuple of (X, y) where X is shape (n_samples, n_features)
        and y is shape (n_samples,)

    Raises:
        ValueError: If cell_model is not compiled or has invalid parameters
    """
    if not cell_model._compiled:
        raise ValueError("cell_model must be compiled. Call cell_model.compile() first.")

    # Set default simulation parameters
    if simulation_params is None:
        simulation_params = {'start': 0, 'end': 10000, 'points': 101}

    # Validate simulation parameters
    if 'start' not in simulation_params or 'end' not in simulation_params or 'points' not in simulation_params:
        raise ValueError('simulation_params must contain "start", "end", and "points" keys')

    # Set default perturbation parameters
    if perturbation_params is None:
        perturbation_params = {'rsd_shape': 0.2}

    # Get initial values (excluding drugs)
    initial_values = cell_model.get_initial_values(exclude_drugs=True)

    # Generate feature data (perturbations)
    feature_df = make_feature_data(
        initial_values=initial_values,
        perturbation_type=perturbation_type,
        perturbation_params=perturbation_params,
        n_samples=n,
        seed=seed,
    )

    # Create and compile solver
    if solver_type == 'scipy':
        solver = ScipySolver()
        antimony_str = cell_model.model.get_antimony_model()
        solver.compile(antimony_str, jit=jit)
    elif solver_type == 'roadrunner':
        solver = RoadrunnerSolver()
        sbml_str = cell_model.model.get_sbml_model()
        solver.compile(sbml_str)
    else:
        raise ValueError(f"Unsupported solver_type: {solver_type}. Use 'scipy' or 'roadrunner'")

    # Generate target data (simulation results)
    target_df, _ = make_target_data(
        model_spec=cell_model.spec,
        solver=solver,
        feature_df=feature_df,
        simulation_params=simulation_params,
        n_cores=1,
        outcome_var=target_specie,
        verbose=verbose,
    )

    # Convert to numpy arrays (sklearn-compatible format)
    X = feature_df.values.astype(np.float64)
    y = target_df.values.ravel().astype(np.float64)

    return X, y


__all__ = ['VirtualCell', 'Builder', 'make_dataset_drug_response']
