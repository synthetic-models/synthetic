"""
Target data generation component.

This module provides functions for generating target data from timecourses,
including robust error handling for CVODE errors.
"""

import warnings
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from ..Solver.Solver import Solver
from ..Specs.BaseSpec import BaseSpec
from .target_calculators import calculate_target_from_series

logger = logging.getLogger(__name__)


def calculate_targets_from_timecourse(
    timecourse_data: Union[pd.DataFrame, List[np.ndarray]],
    outcome_var: str,
    target_method: str = "last_point",
    simulation_params: Optional[Dict[str, Any]] = None,
    drug_start_time: Optional[float] = None,
) -> pd.DataFrame:
    """
    Calculate target values from timecourse data.

    This is the target generation component that processes timecourse arrays
    to generate target values using specified methods.

    Args:
        timecourse_data: Timecourse data (DataFrame with species columns or list of outcome arrays)
        outcome_var: Variable to extract as target
        target_method: Method for calculating targets ('last_point' or 'fold_change_drug')
        simulation_params: Simulation parameters (required for fold_change_drug)
        drug_start_time: Drug start time (required for fold_change_drug)

    Returns:
        DataFrame with single column containing target values

    Raises:
        ValueError: If target_method is invalid or required parameters missing

    Examples:
        >>> timecourse_df = pd.DataFrame({
        ...     'Cp': [np.array([1, 2, 3]), np.array([4, 5, 6])],
        ...     'Oa': [np.array([0.5, 1.0, 1.5]), np.array([2.0, 2.5, 3.0])]
        ... })
        >>> targets = calculate_targets_from_timecourse(
        ...     timecourse_df, 'Cp', target_method='last_point'
        ... )
        >>> targets
            Cp
        0  3.0
        1  6.0
    """

    if isinstance(timecourse_data, pd.DataFrame):
        # DataFrame format: extract outcome_var and calculate targets
        if outcome_var not in timecourse_data.columns:
            raise ValueError(
                f"outcome_var '{outcome_var}' not found in timecourse_data columns"
            )

        target_values = []
        for idx, row in timecourse_data.iterrows():
            series_array = row[outcome_var]
            series = pd.Series(series_array)
            target = calculate_target_from_series(
                series=series,
                target_method=target_method,
                simulation_params=simulation_params,
                drug_start_time=drug_start_time,
            )
            target_values.append(target)

        target_df = pd.DataFrame(
            {outcome_var: target_values}, index=timecourse_data.index
        )
    else:
        # List format: each item is already outcome_var timecourse
        target_values = []
        for timecourse in timecourse_data:
            series = pd.Series(timecourse)
            target = calculate_target_from_series(
                series=series,
                target_method=target_method,
                simulation_params=simulation_params,
                drug_start_time=drug_start_time,
            )
            target_values.append(target)

        target_df = pd.DataFrame({outcome_var: target_values})

    return target_df


def make_target_data_with_params(
    model_spec: BaseSpec,
    solver: Solver,
    feature_df: pd.DataFrame,
    parameter_df: pd.DataFrame = None,
    simulation_params: Dict[str, Any] = None,
    n_cores: int = 1,
    outcome_var: str = "Cp",
    capture_all_species: bool = False,
    verbose: bool = False,
    target_method: str = "last_point",
) -> Tuple[pd.DataFrame, Union[List[np.ndarray], pd.DataFrame]]:
    """
    Generate target data with optional kinetic parameter perturbation.

    Args:
        model_spec: ModelSpecification object
        solver: Solver object (ScipySolver or RoadrunnerSolver)
        feature_df: DataFrame of perturbed initial values
        parameter_df: DataFrame of perturbed kinetic parameters (optional)
        simulation_params: Dictionary with 'start', 'end', 'points' keys
        n_cores: Number of cores for parallel processing (-1 for all cores)
        outcome_var: Variable to extract as target
        capture_all_species: If True, returns DataFrame with timecourses for all species.
                           If False, returns list of arrays for outcome_var only.
        verbose: Whether to show progress bar
        target_method: Method for calculating target values ('last_point' or 'fold_change_drug').
                      'last_point' (default) returns last time point value.
                      'fold_change_drug' returns fold change from drug start to end time.

    Returns:
        Tuple of (target_df, time_course_data)
        time_course_data is either List[np.ndarray] (if capture_all_species=False)
        or pd.DataFrame (if capture_all_species=True)
    """
    # Import target calculator
    from .target_calculators import calculate_target_from_series

    # Set default simulation parameters
    if simulation_params is None:
        simulation_params = {"start": 0, "end": 500, "points": 100}

    # Validate simulation parameters
    if (
        "start" not in simulation_params
        or "end" not in simulation_params
        or "points" not in simulation_params
    ):
        raise ValueError(
            'Simulation parameters must contain "start", "end" and "points" keys'
        )

    start = simulation_params["start"]
    end = simulation_params["end"]
    points = simulation_params["points"]

    # Calculate drug start time for fold_change_drug method
    # Note: make_target_data_with_params doesn't have drug_start_time parameter,
    # so for fold_change_drug we use the midpoint as drug start
    used_drug_start_time = (start + end) / 2

    # Determine species to capture
    species_to_capture = []
    if capture_all_species:
        # Instead of trying to get species from model_spec (deprecated method),
        # we'll capture all species that appear in simulation results.
        # We'll do an initial simulation to discover available species
        try:
            # Use first row of feature_df to get simulation results format
            test_values = feature_df.iloc[0].to_dict()
            solver.set_state_values(test_values)
            if parameter_df is not None:
                test_param_values = parameter_df.iloc[0].to_dict()
                solver.set_parameter_values(test_param_values)

            test_res = solver.simulate(start, end, points)

            # Capture all columns except 'time' and any drug columns
            all_columns = list(test_res.columns)
            species_to_capture = [
                col for col in all_columns if col != "time" and col != outcome_var
            ]

            # Also include outcome_var if not already included
            if outcome_var not in species_to_capture:
                species_to_capture.append(outcome_var)

        except Exception as e:
            # If test simulation fails, fall back to empty list
            # We'll still try to capture species from actual simulations
            warnings.warn(f"Could not discover species from test simulation: {e}")
            species_to_capture = []

    def simulate_perturbation_single(i: int) -> Tuple[float, np.ndarray]:
        """Simulate a single perturbation and capture only outcome_var."""
        perturbed_values = feature_df.iloc[i].to_dict()

        # Set perturbed initial values into solver
        solver.set_state_values(perturbed_values)

        # Set perturbed kinetic parameters if provided
        if parameter_df is not None:
            parameter_values = parameter_df.iloc[i].to_dict()
            solver.set_parameter_values(parameter_values)

        # Run simulation
        res = solver.simulate(start, end, points)

        # Extract target value and time course
        target_value = calculate_target_from_series(
            series=res[outcome_var],
            target_method=target_method,
            simulation_params=simulation_params,
            drug_start_time=used_drug_start_time,
        )
        time_course = res[outcome_var].values

        return target_value, time_course

    def simulate_perturbation_all_species(
        i: int,
    ) -> Tuple[float, Dict[str, np.ndarray]]:
        """Simulate a single perturbation and capture timecourses for all species."""
        perturbed_values = feature_df.iloc[i].to_dict()

        # Set perturbed initial values into solver
        solver.set_state_values(perturbed_values)

        # Set perturbed kinetic parameters if provided
        if parameter_df is not None:
            parameter_values = parameter_df.iloc[i].to_dict()
            solver.set_parameter_values(parameter_values)

        # Run simulation
        res = solver.simulate(start, end, points)

        # Extract target value
        target_value = calculate_target_from_series(
            series=res[outcome_var],
            target_method=target_method,
            simulation_params=simulation_params,
            drug_start_time=used_drug_start_time,
        )

        # Capture timecourses for all species
        time_courses = {}
        for species in species_to_capture:
            if species in res.columns:
                time_courses[species] = res[species].values

        return target_value, time_courses

    if capture_all_species:
        # Use parallel processing if requested
        if n_cores > 1 or n_cores == -1:
            results = Parallel(n_jobs=n_cores)(
                delayed(simulate_perturbation_all_species)(i)
                for i in tqdm(
                    range(feature_df.shape[0]),
                    desc="Simulating perturbations",
                    disable=not verbose,
                )
            )
            all_targets, time_course_dicts = zip(*results)
            all_targets = list(all_targets)
            time_course_dicts = list(time_course_dicts)
        else:
            # Sequential processing
            all_targets = []
            time_course_dicts = []

            for i in tqdm(
                range(feature_df.shape[0]),
                desc="Simulating perturbations",
                disable=not verbose,
            ):
                target_value, time_courses = simulate_perturbation_all_species(i)
                all_targets.append(target_value)
                time_course_dicts.append(time_courses)

        # Create target DataFrame
        target_df = pd.DataFrame(all_targets, columns=[outcome_var])

        # Create timecourse DataFrame
        timecourse_df = pd.DataFrame(time_course_dicts)

        return target_df, timecourse_df
    else:
        # Use parallel processing if requested
        if n_cores > 1 or n_cores == -1:
            results = Parallel(n_jobs=n_cores)(
                delayed(simulate_perturbation_single)(i)
                for i in tqdm(
                    range(feature_df.shape[0]),
                    desc="Simulating perturbations",
                    disable=not verbose,
                )
            )
            all_targets, time_course_data = zip(*results)
            all_targets = list(all_targets)
            time_course_data = list(time_course_data)
        else:
            # Sequential processing
            all_targets = []
            time_course_data = []

            for i in tqdm(
                range(feature_df.shape[0]),
                desc="Simulating perturbations",
                disable=not verbose,
            ):
                target_value, time_course = simulate_perturbation_single(i)
                all_targets.append(target_value)
                time_course_data.append(time_course)

        # Create target DataFrame
        target_df = pd.DataFrame(all_targets, columns=[outcome_var])

        return target_df, time_course_data


def make_target_data_with_params_robust(
    model_spec: BaseSpec,
    solver: Solver,
    feature_df: pd.DataFrame,
    parameter_df: pd.DataFrame = None,
    simulation_params: Dict[str, Any] = None,
    n_cores: int = 1,
    outcome_var: str = "Cp",
    capture_all_species: bool = False,
    verbose: bool = False,
    target_method: str = "last_point",
    drug_start_time: Optional[float] = None,
    basal_time_offset: int = 2,
    require_all_successful: bool = False,
    return_dict: bool = False,
) -> Union[
    Tuple[pd.DataFrame, Union[List[np.ndarray], pd.DataFrame], pd.Series],
    Dict[str, Any],
]:
    """
    Robust version of make_target_data_with_params that handles CVODE errors
    by removing failed samples and maintaining data alignment.

    This function uses the three-component pattern:
    1. Generates timecourse data with error handling
    2. Calculates target values from timecourses
    3. Returns results in requested format

    Args:
        model_spec: ModelSpecification object
        solver: Solver object (ScipySolver or RoadrunnerSolver)
        feature_df: DataFrame of perturbed initial values
        parameter_df: DataFrame of perturbed kinetic parameters (optional)
        simulation_params: Dictionary with 'start', 'end', 'points' keys
        n_cores: Number of cores (reserved for future use, currently ignored)
        outcome_var: Variable to extract as target
        capture_all_species: If True, returns DataFrame with timecourses for all species.
                           If False, returns list of arrays for outcome_var only.
        verbose: Whether to show progress bar
        target_method: Method for calculating target values ('last_point' or 'fold_change_drug').
                      'last_point' (default) returns last time point value.
                      'fold_change_drug' returns fold change from drug start to end time.
        drug_start_time: Time when drug treatment starts (default: midpoint)
        basal_time_offset: Number of time points before drug for basal capture
        require_all_successful: Whether to require all samples to succeed
        return_dict: If True, returns dictionary format with additional metadata

    Returns:
        If return_dict=False: Tuple of (target_df, time_course_data, success_mask)
        If return_dict=True: Dictionary with keys:
            - 'targets': Target DataFrame
            - 'timecourse': Timecourse data
            - 'basal_data': Basal snapshot DataFrame (None if capture_all_species=False)
            - 'features': Filtered feature DataFrame (successful samples)
            - 'parameters': Filtered parameter DataFrame (None if not provided)
            - 'success_mask': Boolean Series indicating which original samples succeeded

    Examples:
        >>> targets, timecourse, success_mask = make_target_data_with_params_robust(
        ...     model_spec=model_spec,
        ...     solver=solver,
        ...     feature_df=feature_df,
        ...     capture_all_species=True
        ... )

        >>> result = make_target_data_with_params_robust(
        ...     model_spec=model_spec,
        ...     solver=solver,
        ...     feature_df=feature_df,
        ...     capture_all_species=True,
        ...     return_dict=True
        ... )
        >>> targets = result['targets']
        >>> basal_df = result['basal_data']
    """
    logger.info("Running robust simulation with CVODE error handling...")

    # Step 1: Generate timecourse data
    from .make_timecourse_data import generate_timecourse_data

    timecourse_result = generate_timecourse_data(
        model_spec=model_spec,
        solver=solver,
        feature_df=feature_df,
        parameter_df=parameter_df,
        simulation_params=simulation_params,
        outcome_var=outcome_var,
        capture_all_species=capture_all_species,
        verbose=verbose,
        drug_start_time=drug_start_time,
        basal_time_offset=basal_time_offset,
        require_all_successful=require_all_successful,
        n_cores=n_cores,
    )

    # Step 2: Calculate targets from timecourses
    target_df = calculate_targets_from_timecourse(
        timecourse_data=timecourse_result["timecourse"],
        outcome_var=outcome_var,
        target_method=target_method,
        simulation_params=simulation_params,
        drug_start_time=drug_start_time,
    )

    # Return in requested format
    if return_dict:
        return {
            "targets": target_df,
            "timecourse": timecourse_result["timecourse"],
            "basal_data": timecourse_result["basal_data"],
            "features": timecourse_result["features"],
            "parameters": timecourse_result["parameters"],
            "success_mask": timecourse_result["success_mask"],
        }
    else:
        return (
            target_df,
            timecourse_result["timecourse"],
            timecourse_result["success_mask"],
        )


__all__ = [
    "calculate_targets_from_timecourse",
    "make_target_data_with_params",
    "make_target_data_with_params_robust",
]
