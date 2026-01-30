"""
Shared helper functions for data generation utilities.

This module provides utility functions and orchestrators for data generation.
The core components (timecourse and target generation) are in separate modules.
"""

import warnings
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from ..Solver.Solver import Solver

# Import from new component modules
from .make_feature_data import make_feature_data
from .make_target_data import (
    make_target_data_with_params_robust,
)

logger = logging.getLogger(__name__)


def validate_simulation_params(simulation_params: Dict[str, Any]) -> None:
    """
    Validate simulation parameters.

    Args:
        simulation_params: Dictionary with 'start', 'end', 'points' keys

    Raises:
        ValueError: If parameters are invalid
    """
    if "start" not in simulation_params:
        raise ValueError('Simulation parameters must contain "start" key')
    if "end" not in simulation_params:
        raise ValueError('Simulation parameters must contain "end" key')
    if "points" not in simulation_params:
        raise ValueError('Simulation parameters must contain "points" key')

    if simulation_params["start"] >= simulation_params["end"]:
        raise ValueError("Start time must be less than end time")
    if simulation_params["points"] <= 0:
        raise ValueError("Number of points must be positive")


def get_pre_drug_index(
    simulation_params: Dict[str, Any],
    drug_start_time: Optional[float] = None,
    offset: int = 2,
) -> int:
    """
    Calculate the index of the last time point before drug treatment.

    Args:
        simulation_params: Dictionary with 'start', 'end', 'points' keys
        drug_start_time: Time when drug treatment starts (default: midpoint of simulation)
        offset: Number of time points before drug to use (default: 2 for safety)

    Returns:
        Index of the pre-drug time point

    Examples:
        >>> sim_params = {'start': 0, 'end': 10000, 'points': 101}
        >>> idx = get_pre_drug_index(sim_params, drug_start_time=5000)
        >>> idx  # Returns index corresponding to time just before 5000
    """
    start = simulation_params["start"]
    end = simulation_params["end"]
    points = simulation_params["points"]

    # Generate time array
    time_array = np.linspace(start, end, points)

    # Default drug start time to midpoint
    if drug_start_time is None:
        drug_start_time = (start + end) / 2

    # Find indices before drug start time
    pre_drug_indices = np.where(time_array < drug_start_time)[0]

    if len(pre_drug_indices) == 0:
        # All time points are at or after drug start
        return 0

    # Use the last index before drug, minus offset for safety
    pre_drug_index = pre_drug_indices[-1]

    # Apply offset, but ensure we don't go below 0
    safe_index = max(0, pre_drug_index - offset)

    return safe_index


def filter_timecourse_to_drug_period(
    timecourse_data: pd.DataFrame,
    simulation_params: Dict[str, Any],
    drug_start_time: float,
) -> pd.DataFrame:
    """
    Filter timecourse data to only include drug treatment period.

    Args:
        timecourse_data: DataFrame where each row is a simulation,
                        each column is a species, cells are numpy arrays.
                        Expected shape: (n_samples, n_species)
        simulation_params: Dictionary with 'start', 'end', 'points' keys
        drug_start_time: Time when drug treatment starts (required)

    Returns:
        Filtered timecourse DataFrame containing only the drug treatment period.
        Each numpy array in the DataFrame is sliced from the pre-drug index to the end.

    Raises:
        ValueError: If drug_start_time is None
        ValueError: If simulation_params doesn't contain required keys

    Examples:
        >>> sim_params = {'start': 0, 'end': 10000, 'points': 101}
        >>> timecourse_df = pd.DataFrame({  # 2 samples, 2 species
        ...     'A': [np.array([1, 2, 3, 4, 5]), np.array([2, 3, 4, 5, 6])],
        ...     'B': [np.array([0, 1, 2, 3, 4]), np.array([1, 2, 3, 4, 5])]
        ... })
        >>> filtered = filter_timecourse_to_drug_period(
        ...     timecourse_df, sim_params, drug_start_time=4000
        ... )
        >>> filtered.shape  # Still 2 samples, but arrays trimmed to drug period
    """
    # Validate drug_start_time is provided
    if drug_start_time is None:
        raise ValueError(
            "drug_start_time must be provided to filter timecourse to drug treatment period"
        )

    # Validate simulation_params
    validate_simulation_params(simulation_params)

    # Get the index where drug treatment starts
    pre_drug_index = get_pre_drug_index(
        simulation_params=simulation_params,
        drug_start_time=drug_start_time,
        offset=0,  # No offset for filtering - start exactly at drug period
    )

    logger.debug(
        f"Filtering timecourse to drug period: pre_drug_index={pre_drug_index}, "
        f"drug_start_time={drug_start_time}"
    )

    # Filter each time series from pre_drug_index to end
    filtered_timecourse = timecourse_data.map(
        lambda x: x[pre_drug_index:] if isinstance(x, np.ndarray) else x
    )

    return filtered_timecourse


def create_default_simulation_params(
    start: float = 0, end: float = 500, points: int = 100
) -> Dict[str, Any]:
    """
    Create default simulation parameters.

    Args:
        start: Start time
        end: End time
        points: Number of points

    Returns:
        Dictionary with simulation parameters
    """
    return {"start": start, "end": end, "points": points}


def prepare_perturbation_values(feature_df_row: pd.Series) -> Dict[str, float]:
    """
    Prepare perturbation values dictionary from DataFrame row.

    Args:
        feature_df_row: Single row from feature DataFrame

    Returns:
        Dictionary of perturbation values
    """
    return feature_df_row.to_dict()


def check_parameter_set_compatibility(
    parameter_set: List[Dict[str, float]], feature_df: pd.DataFrame
) -> None:
    """
    Check compatibility between parameter set and feature DataFrame.

    Args:
        parameter_set: List of parameter dictionaries
        feature_df: Feature DataFrame

    Raises:
        ValueError: If incompatible
    """
    if len(parameter_set) != feature_df.shape[0]:
        raise ValueError(
            f"Parameter set length ({len(parameter_set)}) must match "
            f"feature dataframe rows ({feature_df.shape[0]})"
        )


def create_feature_target_pipeline(
    make_feature_data_func,
    make_target_data_func,
    initial_values: Dict[str, float],
    perturbation_params: Dict[str, Any],
    n_samples: int,
    model_spec=None,
    solver=None,
    simulation_params: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    **kwargs,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a pipeline that generates both feature and target data.

    Args:
        make_feature_data_func: Function to generate feature data
        make_target_data_func: Function to generate target data
        initial_values: Dictionary of initial values
        perturbation_params: Parameters for perturbation
        n_samples: Number of samples
        model_spec: Model specification (optional)
        solver: Solver object (optional)
        simulation_params: Simulation parameters (optional)
        seed: Random seed for feature generation
        **kwargs: Additional arguments for target data generation

    Returns:
        Tuple of (feature_df, target_df)
    """
    # Generate feature data
    feature_df = make_feature_data_func(
        initial_values=initial_values,
        perturbation_params=perturbation_params,
        n_samples=n_samples,
        seed=seed,
    )

    # Generate target data if model_spec and solver are provided
    if model_spec is not None and solver is not None:
        target_df, _ = make_target_data_func(
            model_spec=model_spec,
            solver=solver,
            feature_df=feature_df,
            simulation_params=simulation_params,
            **kwargs,
        )
    else:
        # Return empty target DataFrame if no model/solver provided
        target_df = pd.DataFrame()

    return feature_df, target_df




# Unified function that returns both feature and target data
def make_data(
    initial_values: Dict[str, float],
    perturbation_type: str,
    perturbation_params: Dict[str, Any],
    n_samples: int,
    solver: Optional[Solver] = None,
    parameter_values: Optional[Dict[str, float]] = None,
    param_perturbation_type: str = "none",
    param_perturbation_params: Optional[Dict[str, Any]] = None,
    simulation_params: Optional[Dict[str, Any]] = None,
    drug_start_time: Optional[float] = None,
    basal_time_offset: int = 2,
    seed: Optional[int] = None,
    param_seed: Optional[int] = None,
    require_all_successful: bool = False,
    return_details: bool = False,
    capture_all_species: bool = False,
    target_method: str = "last_point",
    n_cores: int = 1,
    verbose: bool = False,
    **kwargs,
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Dict[str, Any]]:
    """
    Generate both feature and target data in one call with error handling.

    This function orchestrates the three-component data generation pattern:
    1. Feature data generation (via make_feature_data)
    2. Timecourse data generation (via make_target_data_with_params_robust)
    3. Target data generation (via calculate_targets_from_timecourse)

    Args:
        initial_values: Dictionary of initial values
        perturbation_type: Type of perturbation ('uniform', 'gaussian', 'lognormal', 'lhs')
        perturbation_params: Parameters for perturbation
        n_samples: Number of samples
        solver: Solver object (required for target generation) - can load any SBML/Antimony model
        parameter_values: Dictionary of kinetic parameter values (optional)
        param_perturbation_type: Type of perturbation for kinetic parameters ('none', 'uniform', 'gaussian', 'lognormal', 'lhs')
        param_perturbation_params: Parameters for kinetic parameter perturbation (optional)
        simulation_params: Simulation parameters (optional)
        seed: Random seed for feature generation
        param_seed: Random seed for parameter generation (optional, uses seed if not provided)
        require_all_successful: Whether to require all samples to succeed (default: False)
        return_details: If True, returns extended data structure with intermediate datasets (default: False)
        capture_all_species: If True, captures timecourses for all species in DataFrame format.
                           If False, captures only outcome variable timecourse as list of arrays.
        target_method: Method for calculating target values ('last_point' or 'fold_change_drug').
                      'last_point' (default) returns the last time point value.
                      'fold_change_drug' returns fold change from drug start to end time.
        **kwargs: Additional arguments for target data generation

    Returns:
        If return_details=False: Tuple of (feature_df, target_df) where target_df may contain NaN values for failed simulations
        If return_details=True: Dictionary with keys:
            - 'features': Feature dataframe (initial values)
            - 'targets': Target dataframe (outcome values)
            - 'parameters': Kinetic parameters dataframe (None if not provided)
            - 'timecourse': Timecourse simulation data (DataFrame if capture_all_species=True, list of arrays if False, None if not captured)
            - 'basal_data': Basal snapshot DataFrame (None if capture_all_species=False)
            - 'metadata': Dictionary with metadata about the generation process

    Examples:
        >>> # Load an external SBML model
        >>> from synthetic.Solver.RoadrunnerSolver import RoadrunnerSolver
        >>> solver = RoadrunnerSolver()
        >>> solver.compile(sbml_str)
        >>> 
        >>> X, y = make_data(
        ...     initial_values={'A': 10.0, 'B': 20.0},
        ...     perturbation_type='gaussian',
        ...     perturbation_params={'std': 2.0},
        ...     n_samples=100,
        ...     solver=solver,
        ...     seed=42
        ... )

        >>> result = make_data(
        ...     initial_values=inactive_state_variables,
        ...     perturbation_type='lognormal',
        ...     perturbation_params={'shape': 0.5},
        ...     parameter_values=kinetic_parameters,
        ...     param_perturbation_type='uniform',
        ...     param_perturbation_params={'min': 0.8, 'max': 1.2},
        ...     n_samples=1000,
        ...     solver=solver,
        ...     simulation_params={'start': 0, 'end': 10000, 'points': 101},
        ...     seed=42,
        ...     outcome_var='Oa',
        ...     require_all_successful=True,
        ...     return_details=True,
        ...     capture_all_species=True  # Capture timecourses for all species
        ... )
        >>> feature_df = result['features']
        >>> target_df = result['targets']
        >>> param_df = result['parameters']
        >>> timecourse_df = result['timecourse']  # DataFrame with all species timecourses
    """
    # Component 1: Feature data generation
    feature_df = make_feature_data(
        initial_values=initial_values,
        perturbation_type=perturbation_type,
        perturbation_params=perturbation_params,
        n_samples=n_samples,
        seed=seed,
    )

    # Generate kinetic parameter perturbations if provided
    parameter_df = None
    if parameter_values is not None and param_perturbation_type != "none":
        parameter_df = make_feature_data(
            initial_values=parameter_values,
            perturbation_type=param_perturbation_type,
            perturbation_params=param_perturbation_params,
            n_samples=n_samples,
            seed=param_seed if param_seed is not None else seed,
        )

    # Initialize result variables
    target_df = pd.DataFrame()
    timecourse_data = None
    basal_df = None
    failed_indices = []

    # Components 2 & 3: Timecourse and target data generation
    if solver is not None:
        if simulation_params is None:
            raise ValueError(
                "simulation_params must be provided when generating target data"
            )

        outcome_var = kwargs.get("outcome_var", "Oa")
        verbose = kwargs.get("verbose", False)

        # Use make_target_data_with_params_robust as underlying function
        # This handles timecourse generation and target calculation
        robust_result = make_target_data_with_params_robust(
            solver=solver,
            feature_df=feature_df,
            parameter_df=parameter_df,
            simulation_params=simulation_params,
            outcome_var=outcome_var,
            capture_all_species=capture_all_species,
            verbose=verbose,
            target_method=target_method,
            drug_start_time=drug_start_time,
            basal_time_offset=basal_time_offset,
            require_all_successful=require_all_successful,
            n_cores=n_cores,
            return_dict=True,  # Get full dictionary output
        )

        # Extract results from robust function
        successful_targets = robust_result["targets"]
        successful_timecourse = robust_result["timecourse"]
        successful_basal = robust_result["basal_data"]
        successful_features = robust_result["features"]
        successful_params = robust_result["parameters"]
        success_mask = robust_result["success_mask"]

        # For backward compatibility, we need to preserve all original samples
        # (including failed ones with NaN values)
        target_df = pd.DataFrame(index=feature_df.index, columns=[outcome_var])
        target_df.loc[success_mask] = successful_targets
        target_df.loc[~success_mask] = np.nan

        # Update feature_df with successful resampled values
        if successful_features is not None:
            feature_df.update(successful_features)

        # Update parameter_df with successful resampled values
        if parameter_df is not None and successful_params is not None:
            parameter_df.update(successful_params)

        # Calculate failed indices
        failed_indices = (~success_mask).tolist()

        # Handle timecourse and basal data for return_details
        if return_details:
            if capture_all_species:
                # Reconstruct full timecourse DataFrame with NaN for failed samples
                if successful_timecourse is not None:
                    timecourse_data = pd.DataFrame(
                        index=feature_df.index, columns=successful_timecourse.columns
                    )
                    timecourse_data.loc[success_mask] = successful_timecourse
                    timecourse_data.loc[~success_mask] = np.nan

                # Reconstruct basal DataFrame with NaN for failed samples
                if successful_basal is not None:
                    basal_df = pd.DataFrame(
                        index=feature_df.index, columns=successful_basal.columns
                    )
                    basal_df.loc[success_mask] = successful_basal
                    basal_df.loc[~success_mask] = np.nan
            else:
                # List format: pad with None for failed samples
                timecourse_data = [None] * len(feature_df)
                for i, success in enumerate(success_mask):
                    if success and successful_timecourse is not None:
                        timecourse_data[i] = successful_timecourse[i]
        else:
            # When not returning details, we still need to construct timecourse for internal use
            # but won't return it
            timecourse_data = successful_timecourse

        # Calculate metadata
        success_rate = (
            (len(success_mask) - sum(~success_mask)) / len(success_mask)
            if len(success_mask) > 0
            else 1.0
        )

        # Calculate drug start time and pre_drug_index for metadata
        if simulation_params is not None:
            used_drug_start_time = (
                drug_start_time
                if drug_start_time is not None
                else (simulation_params["start"] + simulation_params["end"]) / 2
            )
            pre_drug_index = get_pre_drug_index(
                simulation_params=simulation_params,
                drug_start_time=used_drug_start_time,
                offset=basal_time_offset,
            )
        else:
            used_drug_start_time = None
            pre_drug_index = None

    # Return in requested format
    if return_details:
        metadata = {
            "failed_indices": failed_indices,
            "success_rate": success_rate
            if solver is not None
            else 1.0,
            "n_samples": n_samples,
            "perturbation_type": perturbation_type,
            "capture_all_species": capture_all_species,
            "target_method": target_method,
            "simulation_params": simulation_params,
            "drug_start_time": used_drug_start_time,
            "pre_drug_index": pre_drug_index,
            "basal_time_offset": basal_time_offset,
        }

        return {
            "features": feature_df,
            "targets": target_df,
            "parameters": parameter_df,
            "timecourse": timecourse_data,
            "basal_data": basal_df,
            "metadata": metadata,
        }
    else:
        return feature_df, target_df


def make_data_extended(
    initial_values: Dict[str, float],
    perturbation_type: str,
    perturbation_params: Dict[str, Any],
    n_samples: int,
    solver: Optional[Solver] = None,
    parameter_values: Optional[Dict[str, float]] = None,
    param_perturbation_type: str = "none",
    param_perturbation_params: Optional[Dict[str, Any]] = None,
    simulation_params: Optional[Dict[str, Any]] = None,
    drug_start_time: Optional[float] = None,
    basal_time_offset: int = 2,
    seed: Optional[int] = None,
    param_seed: Optional[int] = None,
    require_all_successful: bool = False,
    capture_all_species: bool = True,
    target_method: str = "last_point",
    n_cores: int = 1,
    verbose: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """
    Generate feature, target, and intermediate datasets with comprehensive return.

    This function is a convenience wrapper around make_data() with return_details=True.
    By default, captures timecourses for all species in DataFrame format.

    Args:
        initial_values: Dictionary of initial values
        perturbation_type: Type of perturbation ('uniform', 'gaussian', 'lognormal', 'lhs')
        perturbation_params: Parameters for perturbation
        n_samples: Number of samples
        solver: Solver object (required for target generation) - can load any SBML/Antimony model
        parameter_values: Dictionary of kinetic parameter values (optional)
        param_perturbation_type: Type of perturbation for kinetic parameters ('none', 'uniform', 'gaussian', 'lognormal', 'lhs')
        param_perturbation_params: Parameters for kinetic parameter perturbation (optional)
        simulation_params: Simulation parameters (optional)
        seed: Random seed for feature generation
        param_seed: Random seed for parameter generation (optional, uses seed if not provided)
        require_all_successful: Whether to require all samples to succeed (default: False)
        capture_all_species: If True (default), captures timecourses for all species in DataFrame format.
                           If False, captures only outcome variable timecourse as list of arrays.
        target_method: Method for calculating target values ('last_point' or 'fold_change_drug').
                      'last_point' (default) returns the last time point value.
                      'fold_change_drug' returns fold change from drug start to end time.
        **kwargs: Additional arguments for target data generation

    Returns:
        Dictionary with keys:
            - 'features': Feature dataframe (initial values)
            - 'targets': Target dataframe (outcome values)
            - 'parameters': Kinetic parameters dataframe (None if not provided)
            - 'timecourse': Timecourse simulation data (DataFrame if capture_all_species=True, list of arrays if False)
            - 'metadata': Dictionary with metadata about the generation process

    Examples:
        >>> # Load an external SBML model
        >>> from synthetic.Solver.RoadrunnerSolver import RoadrunnerSolver
        >>> solver = RoadrunnerSolver()
        >>> solver.compile(sbml_str)
        >>> 
        >>> result = make_data_extended(
        ...     initial_values={'A': 10.0, 'B': 20.0},
        ...     perturbation_type='gaussian',
        ...     perturbation_params={'std': 2.0},
        ...     n_samples=100,
        ...     solver=solver,
        ...     seed=42
        ... )
        >>> feature_df = result['features']
        >>> target_df = result['targets']
        >>> timecourse_df = result['timecourse']  # DataFrame with all species timecourses

        >>> result = make_data_extended(
        ...     initial_values={'A': 10.0, 'B': 20.0},
        ...     perturbation_type='gaussian',
        ...     perturbation_params={'std': 2.0},
        ...     n_samples=100,
        ...     solver=solver,
        ...     seed=42,
        ...     capture_all_species=False  # Capture only outcome variable timecourse
        ... )
        >>> timecourse_list = result['timecourse']  # List of arrays for outcome variable only
    """
    return make_data(
        initial_values=initial_values,
        perturbation_type=perturbation_type,
        perturbation_params=perturbation_params,
        n_samples=n_samples,
        solver=solver,
        parameter_values=parameter_values,
        param_perturbation_type=param_perturbation_type,
        param_perturbation_params=param_perturbation_params,
        simulation_params=simulation_params,
        drug_start_time=drug_start_time,
        basal_time_offset=basal_time_offset,
        seed=seed,
        param_seed=param_seed,
        require_all_successful=require_all_successful,
        return_details=True,
        capture_all_species=capture_all_species,
        target_method=target_method,
        n_cores=n_cores,
        verbose=verbose,
        **kwargs,
    )


def add_deprecation_warning(
    old_function_name: str, new_function_name: str, stacklevel: int = 2
):
    """
    Add deprecation warning to a function.

    Args:
        old_function_name: Name of deprecated function
        new_function_name: Name of new function to use instead
        stacklevel: Stack level for warning
    """
    warnings.warn(
        f"{old_function_name} is deprecated. Use {new_function_name} instead.",
        DeprecationWarning,
        stacklevel=stacklevel,
    )
