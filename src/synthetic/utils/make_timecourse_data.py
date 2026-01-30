"""
Timecourse data generation component.

This module provides robust timecourse generation with error handling
and basal data collection.
"""

import warnings
import logging
from typing import Dict, Any, Optional, Tuple
import pandas as pd
from tqdm import tqdm

from ..Solver.Solver import Solver

logger = logging.getLogger(__name__)


def generate_timecourse_data(
    solver: Solver,
    feature_df: pd.DataFrame,
    parameter_df: pd.DataFrame = None,
    simulation_params: Dict[str, Any] = None,
    outcome_var: str = "Cp",
    capture_all_species: bool = False,
    verbose: bool = False,
    drug_start_time: Optional[float] = None,
    basal_time_offset: int = 2,
    require_all_successful: bool = False,
    n_cores: int = 1,
) -> Dict[str, Any]:
    """
    Generate timecourse data with error handling.

    This is the core timecourse generation component that handles:
    - Simulation execution with error handling
    - Basal (pre-drug) data collection
    - Support for both single-species and all-species capture

    Args:
        solver: Solver object (ScipySolver or RoadrunnerSolver) - can load any SBML/Antimony model
        feature_df: DataFrame of perturbed initial values
        parameter_df: DataFrame of perturbed kinetic parameters (optional)
        simulation_params: Dictionary with 'start', 'end', 'points' keys
        outcome_var: Variable to extract as target
        capture_all_species: If True, captures timecourses for all species.
                           If False, captures only outcome variable.
        verbose: Whether to show progress bar
        drug_start_time: Time when drug treatment starts (default: midpoint)
        basal_time_offset: Number of time points before drug for basal capture
        require_all_successful: Whether to require all samples to succeed
        n_cores: Number of cores (reserved for future use, currently ignored)

    Returns:
        Dictionary with keys:
            - 'features': Filtered feature DataFrame (successful samples only)
            - 'parameters': Filtered parameter DataFrame (successful samples only, None if not provided)
            - 'timecourse': Timecourse data (DataFrame if capture_all_species=True, list if False)
            - 'basal_data': Basal snapshot DataFrame (None if capture_all_species=False)
            - 'success_mask': Boolean Series indicating which original samples succeeded
            - 'success_indices': List of indices of successful samples

    Examples:
        >>> # Load an external SBML model
        >>> from synthetic.Solver.RoadrunnerSolver import RoadrunnerSolver
        >>> solver = RoadrunnerSolver()
        >>> solver.compile(sbml_str)
        >>> 
        >>> # Generate timecourse data
        >>> result = generate_timecourse_data(
        ...     solver=solver,
        ...     feature_df=feature_df,
        ...     simulation_params={'start': 0, 'end': 500, 'points': 100},
        ...     capture_all_species=True
        ... )
        >>> successful_features = result['features']
        >>> timecourse_df = result['timecourse']
        >>> basal_df = result['basal_data']
    """
    from .data_generation_helpers import (
        validate_simulation_params,
        get_pre_drug_index,
    )

    logger.info("Generating timecourse data with robust error handling...")

    # Set default simulation parameters
    if simulation_params is None:
        raise ValueError(
            "generate_timecourse_data(): simulation_params must be provided with 'start', 'end', and 'points'."
        )

    validate_simulation_params(simulation_params)

    start = simulation_params["start"]
    end = simulation_params["end"]
    points = simulation_params["points"]

    # Calculate drug start time and basal index
    used_drug_start_time = (
        drug_start_time if drug_start_time is not None else (start + end) / 2
    )
    pre_drug_index = get_pre_drug_index(
        simulation_params=simulation_params,
        drug_start_time=used_drug_start_time,
        offset=basal_time_offset,
    )

    # Determine species to capture if capture_all_species is True
    species_to_capture = []
    if capture_all_species:
        try:
            test_feature_values = feature_df.iloc[0].to_dict()
            solver.set_state_values(test_feature_values)

            if parameter_df is not None:
                test_param_values = parameter_df.iloc[0].to_dict()
                solver.set_parameter_values(test_param_values)

            test_res = solver.simulate(start, end, points)
            species_to_capture = [col for col in test_res.columns if col != "time"]

            if outcome_var not in species_to_capture:
                species_to_capture.append(outcome_var)
        except Exception as e:
            warnings.warn(f"Could not discover species from test simulation: {e}")

    from ..SyntheticGenUtils.ParallelUtils import run_parallel_with_error_handling
    
    def simulate_single_sample(
        feature_values: Dict[str, float],
        param_values: Optional[Dict[str, float]] = None,
    ) -> Tuple[Optional[pd.DataFrame], Optional[Dict[str, float]]]:
        """Simulate a single sample and return results with basal snapshot."""
        try:
            solver.set_state_values(feature_values)
            if param_values is not None:
                solver.set_parameter_values(param_values)

            res = solver.simulate(start, end, points)

            # Collect basal snapshot if needed
            basal_snapshot = None
            if capture_all_species and species_to_capture:
                basal_snapshot = {}
                for species in species_to_capture:
                    if species in res.columns:
                        basal_snapshot[species] = float(
                            res[species].iloc[pre_drug_index]
                        )

            return res, basal_snapshot
        except RuntimeError as e:
            if "CV_TOO_MUCH_WORK" in str(e) or "CVODE" in str(e):
                return None, None
            else:
                raise
        except Exception:
            return None, None

    def simulate_single_sample_wrapper(i: int) -> Tuple[int, Optional[pd.DataFrame], Optional[Dict[str, float]]]:
        """Wrapper function for parallel execution that includes index for result aggregation."""
        original_feature_values = feature_df.iloc[i].to_dict()
        original_param_values = (
            parameter_df.iloc[i].to_dict() if parameter_df is not None else None
        )

        # Simulate single sample
        res, basal = simulate_single_sample(
            original_feature_values, original_param_values
        )
        
        return i, res, basal

    # Process simulations (parallel or sequential)
    if n_cores > 1 or n_cores == -1:
        # Parallel processing with error handling
        results = run_parallel_with_error_handling(
            simulation_function=simulate_single_sample_wrapper,
            n_iterations=feature_df.shape[0],
            n_cores=n_cores,
            verbose=verbose,
            description="Simulating perturbations",
            error_prefix="Error simulating perturbation"
        )
        # results contains list of (i, res, basal) tuples where res/basal may be None for failures
    else:
        # Sequential processing
        results = []
        for i in tqdm(
            range(feature_df.shape[0]), 
            desc="Simulating perturbations", 
            disable=not verbose
        ):
            results.append(simulate_single_sample_wrapper(i))

    # Process results (both parallel and sequential paths end up here)
    successful_timecourses = []
    successful_basal_data = []
    successful_indices = []
    successful_features = []
    successful_params = [] if parameter_df is not None else None
    failed_indices = []

    for i, res, basal in results:
        if res is not None:
            # Extract timecourse
            if capture_all_species:
                timecourse_dict = {}
                for species in species_to_capture:
                    if species in res.columns:
                        timecourse_dict[species] = res[species].values
                successful_timecourses.append(timecourse_dict)
                if basal is not None:
                    successful_basal_data.append(basal)
            else:
                successful_timecourses.append(res[outcome_var].values)

            successful_indices.append(i)
            successful_features.append(feature_df.iloc[i].to_dict())
            if parameter_df is not None:
                successful_params.append(parameter_df.iloc[i].to_dict())
        else:
            failed_indices.append(i)

    if failed_indices and verbose:
        logger.info(f"Failed to simulate {len(failed_indices)} samples.")

    # Handle require_all_successful
    if require_all_successful and failed_indices:
        raise RuntimeError(
            f"Failed to simulate {len(failed_indices)} samples. Failed indices: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}"
        )

    # Create success mask
    success_mask = pd.Series([False] * feature_df.shape[0])
    success_mask.iloc[successful_indices] = True

    # Create output DataFrames
    successful_feature_df = pd.DataFrame(successful_features, index=successful_indices)
    successful_parameter_df = None
    if parameter_df is not None:
        successful_parameter_df = pd.DataFrame(
            successful_params, index=successful_indices
        )

    # Create timecourse output
    if capture_all_species and successful_timecourses:
        timecourse_data = pd.DataFrame(successful_timecourses, index=successful_indices)
        basal_data = (
            pd.DataFrame(successful_basal_data, index=successful_indices)
            if successful_basal_data
            else None
        )
    else:
        timecourse_data = successful_timecourses  # List of arrays
        basal_data = None

    logger.info(
        f"Timecourse generation completed: {len(successful_indices)}/{feature_df.shape[0]} samples succeeded"
    )

    return {
        "features": successful_feature_df,
        "parameters": successful_parameter_df,
        "timecourse": timecourse_data,
        "basal_data": basal_data,
        "success_mask": success_mask,
        "success_indices": successful_indices,
    }


__all__ = [
    "generate_timecourse_data",
]
