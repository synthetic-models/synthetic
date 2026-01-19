"""
Shared helper functions for data generation utilities.
"""

import warnings
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from ..Solver.Solver import Solver
from ..Specs.BaseSpec import BaseSpec as ModelSpecification


def validate_simulation_params(simulation_params: Dict[str, Any]) -> None:
    """
    Validate simulation parameters.
    
    Args:
        simulation_params: Dictionary with 'start', 'end', 'points' keys
        
    Raises:
        ValueError: If parameters are invalid
    """
    if 'start' not in simulation_params:
        raise ValueError('Simulation parameters must contain "start" key')
    if 'end' not in simulation_params:
        raise ValueError('Simulation parameters must contain "end" key')
    if 'points' not in simulation_params:
        raise ValueError('Simulation parameters must contain "points" key')
    
    if simulation_params['start'] >= simulation_params['end']:
        raise ValueError('Start time must be less than end time')
    if simulation_params['points'] <= 0:
        raise ValueError('Number of points must be positive')


def extract_species_from_model_spec(
    model_spec,
    include_phospho: bool = True
) -> List[str]:
    """
    Extract species list from model specification.
    
    Args:
        model_spec: ModelSpecification object
        include_phospho: Whether to include phosphorylated versions
        
    Returns:
        List of species names
    """
    species = []
    
    # Try to access species attributes (handling different model spec versions)
    if hasattr(model_spec, 'A_species'):
        species.extend(model_spec.A_species)
    if hasattr(model_spec, 'B_species'):
        species.extend(model_spec.B_species)
    if hasattr(model_spec, 'C_species'):
        species.extend(model_spec.C_species)
    
    if include_phospho:
        # Add phosphorylated versions
        species_with_phospho = []
        for s in species:
            species_with_phospho.append(s)
            species_with_phospho.append(s + 'p')
        return species_with_phospho
    
    return species


def create_default_simulation_params(
    start: float = 0,
    end: float = 500,
    points: int = 100
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
    return {
        'start': start,
        'end': end,
        'points': points
    }


def prepare_perturbation_values(
    feature_df_row: pd.Series
) -> Dict[str, float]:
    """
    Prepare perturbation values dictionary from DataFrame row.
    
    Args:
        feature_df_row: Single row from feature DataFrame
        
    Returns:
        Dictionary of perturbation values
    """
    return feature_df_row.to_dict()


def check_parameter_set_compatibility(
    parameter_set: List[Dict[str, float]],
    feature_df: pd.DataFrame
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
            f'Parameter set length ({len(parameter_set)}) must match '
            f'feature dataframe rows ({feature_df.shape[0]})'
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
    **kwargs
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
        seed=seed
    )
    
    # Generate target data if model_spec and solver are provided
    if model_spec is not None and solver is not None:
        target_df, _ = make_target_data_func(
            model_spec=model_spec,
            solver=solver,
            feature_df=feature_df,
            simulation_params=simulation_params,
            **kwargs
        )
    else:
        # Return empty target DataFrame if no model/solver provided
        target_df = pd.DataFrame()
    
    return feature_df, target_df


def make_target_data_with_params(
    model_spec: ModelSpecification,
    solver: Solver,
    feature_df: pd.DataFrame,
    parameter_df: pd.DataFrame = None,
    simulation_params: Dict[str, Any] = None,
    n_cores: int = 1,
    outcome_var: str = 'Cp',
    capture_all_species: bool = False,
    verbose: bool = False
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
        
    Returns:
        Tuple of (target_df, time_course_data)
        time_course_data is either List[np.ndarray] (if capture_all_species=False) 
        or pd.DataFrame (if capture_all_species=True)
    """
    # Set default simulation parameters
    if simulation_params is None:
        simulation_params = {'start': 0, 'end': 500, 'points': 100}
    
    # Validate simulation parameters
    if 'start' not in simulation_params or 'end' not in simulation_params or 'points' not in simulation_params:
        raise ValueError('Simulation parameters must contain "start", "end" and "points" keys')
    
    start = simulation_params['start']
    end = simulation_params['end']
    points = simulation_params['points']
    
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
            species_to_capture = [col for col in all_columns if col != 'time' and col != outcome_var]
            
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
        target_value = res[outcome_var].iloc[-1]
        time_course = res[outcome_var].values
        
        return target_value, time_course
    
    def simulate_perturbation_all_species(i: int) -> Tuple[float, Dict[str, np.ndarray]]:
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
        target_value = res[outcome_var].iloc[-1]
        
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
                for i in tqdm(range(feature_df.shape[0]), 
                             desc='Simulating perturbations', 
                             disable=not verbose)
            )
            all_targets, time_course_dicts = zip(*results)
            all_targets = list(all_targets)
            time_course_dicts = list(time_course_dicts)
        else:
            # Sequential processing
            all_targets = []
            time_course_dicts = []
            
            for i in tqdm(range(feature_df.shape[0]), 
                         desc='Simulating perturbations', 
                         disable=not verbose):
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
                for i in tqdm(range(feature_df.shape[0]), 
                             desc='Simulating perturbations', 
                             disable=not verbose)
            )
            all_targets, time_course_data = zip(*results)
            all_targets = list(all_targets)
            time_course_data = list(time_course_data)
        else:
            # Sequential processing
            all_targets = []
            time_course_data = []
            
            for i in tqdm(range(feature_df.shape[0]), 
                         desc='Simulating perturbations', 
                         disable=not verbose):
                target_value, time_course = simulate_perturbation_single(i)
                all_targets.append(target_value)
                time_course_data.append(time_course)
        
        # Create target DataFrame
        target_df = pd.DataFrame(all_targets, columns=[outcome_var])
        
        return target_df, time_course_data


def generate_batch_alternatives(base_values: Dict[str, float], 
                               perturbation_type: str,
                               perturbation_params: Dict[str, Any],
                               batch_size: int,
                               base_seed: int,
                               attempt: int) -> pd.DataFrame:
    """
    Generate a batch of alternative values for resampling.
    
    Args:
        base_values: Dictionary of base values to perturb
        perturbation_type: Type of perturbation ('uniform', 'gaussian', 'lognormal', 'lhs')
        perturbation_params: Parameters for perturbation
        batch_size: Number of alternative samples to generate
        base_seed: Base random seed
        attempt: Resampling attempt number (used to generate unique seeds)
        
    Returns:
        DataFrame with batch_size alternative samples
    """
    from .make_feature_data import make_feature_data
    
    # Use a unique seed for each resampling attempt
    alt_seed = base_seed + 1000 * attempt + batch_size
    return make_feature_data(
        initial_values=base_values,
        perturbation_type=perturbation_type,
        perturbation_params=perturbation_params,
        n_samples=batch_size,
        seed=alt_seed
    )


# Unified function that returns both feature and target data
def make_data(
    initial_values: Dict[str, float],
    perturbation_type: str,
    perturbation_params: Dict[str, Any],
    n_samples: int,
    model_spec=None,
    solver=None,
    parameter_values: Optional[Dict[str, float]] = None,
    param_perturbation_type: str = 'none',
    param_perturbation_params: Optional[Dict[str, Any]] = None,
    simulation_params: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    param_seed: Optional[int] = None,
    resample_size: int = 10,
    max_retries: int = 3,
    require_all_successful: bool = False,
    return_details: bool = False,
    capture_all_species: bool = False,
    **kwargs
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Dict[str, Any]]:
    """
    Generate both feature and target data in one call with robust error handling.
    
    Args:
        initial_values: Dictionary of initial values
        perturbation_type: Type of perturbation ('uniform', 'gaussian', 'lognormal', 'lhs')
        perturbation_params: Parameters for perturbation
        n_samples: Number of samples
        model_spec: Model specification (required for target generation)
        solver: Solver object (required for target generation)
        parameter_values: Dictionary of kinetic parameter values (optional)
        param_perturbation_type: Type of perturbation for kinetic parameters ('none', 'uniform', 'gaussian', 'lognormal', 'lhs')
        param_perturbation_params: Parameters for kinetic parameter perturbation (optional)
        simulation_params: Simulation parameters (optional)
        seed: Random seed for feature generation
        param_seed: Random seed for parameter generation (optional, uses seed if not provided)
        resample_size: Number of alternative samples to generate when a simulation fails (default: 10)
        max_retries: Maximum number of resampling attempts per failed index (default: 3)
        require_all_successful: Whether to require all samples to succeed (default: False)
        return_details: If True, returns extended data structure with intermediate datasets (default: False)
        capture_all_species: If True, captures timecourses for all species in DataFrame format.
                           If False, captures only outcome variable timecourse as list of arrays.
        **kwargs: Additional arguments for target data generation
        
    Returns:
        If return_details=False: Tuple of (feature_df, target_df) where target_df may contain NaN values for failed simulations
        If return_details=True: Dictionary with keys:
            - 'features': Feature dataframe (initial values)
            - 'targets': Target dataframe (outcome values)
            - 'parameters': Kinetic parameters dataframe (None if not provided)
            - 'timecourse': Timecourse simulation data (DataFrame if capture_all_species=True, list of arrays if False, None if not captured)
            - 'metadata': Dictionary with metadata about the generation process
        
    Examples:
        >>> X, y = make_data(
        ...     initial_values={'A': 10.0, 'B': 20.0},
        ...     perturbation_type='gaussian',
        ...     perturbation_params={'std': 2.0},
        ...     n_samples=100,
        ...     model_spec=model_spec,
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
        ...     model_spec=degree_spec,
        ...     solver=solver,
        ...     simulation_params={'start': 0, 'end': 10000, 'points': 101},
        ...     seed=42,
        ...     outcome_var='Oa',
        ...     resample_size=10,
        ...     max_retries=3,
        ...     return_details=True,
        ...     capture_all_species=True  # Capture timecourses for all species
        ... )
        >>> feature_df = result['features']
        >>> target_df = result['targets']
        >>> param_df = result['parameters']
        >>> timecourse_df = result['timecourse']  # DataFrame with all species timecourses
    """
    # TODO: ensure that target and feature data generated do not have negative values 
    from .make_feature_data import make_feature_data
    from tqdm import tqdm
    from joblib import Parallel, delayed
    from ..Solver.Solver import Solver
    from ..Specs.BaseSpec import BaseSpec as ModelSpecification
    
    # Generate feature data (initial value perturbations)
    feature_df = make_feature_data(
        initial_values=initial_values,
        perturbation_type=perturbation_type,
        perturbation_params=perturbation_params,
        n_samples=n_samples,
        seed=seed
    )
    
    # Generate kinetic parameter perturbations if provided
    parameter_df = None
    if parameter_values is not None and param_perturbation_type != 'none':
        parameter_df = make_feature_data(
            initial_values=parameter_values,
            perturbation_type=param_perturbation_type,
            perturbation_params=param_perturbation_params,
            n_samples=n_samples,
            seed=param_seed if param_seed is not None else seed
        )
    
    # Initialize timecourse data storage
    timecourse_data = None
    
    # Generate target data if model_spec and solver are provided
    if model_spec is not None and solver is not None:
        # Set default simulation parameters
        if simulation_params is None:
            raise ValueError('simulation_params must be provided when generating target data')
        
        # Validate simulation parameters
        if 'start' not in simulation_params or 'end' not in simulation_params or 'points' not in simulation_params:
            raise ValueError('Simulation parameters must contain "start", "end" and "points" keys')
        
        start = simulation_params['start']
        end = simulation_params['end']
        points = simulation_params['points']
        outcome_var = kwargs.get('outcome_var', 'Oa')
        n_cores = kwargs.get('n_cores', 1)
        verbose = kwargs.get('verbose', False)
        
        # Note: n_cores parameter is passed but not used in make_data() due to:
        # 1. Solver object cannot be pickled for multiprocessing
        # 2. Resampling logic updates DataFrames in-place, which is not parallel-friendly
        # 3. Error handling and retry logic assumes sequential processing
        # For simple parallel processing without resampling, consider using make_target_data_with_params()
        
        # Determine species to capture if capture_all_species is True
        species_to_capture = []
        if capture_all_species:
            # Instead of using deprecated model_spec attributes,
            # we'll discover species from a test simulation
            try:
                # Use first row to test what species are available in simulation results
                test_feature_values = feature_df.iloc[0].to_dict()
                solver.set_state_values(test_feature_values)
                
                if parameter_df is not None:
                    test_param_values = parameter_df.iloc[0].to_dict()
                    solver.set_parameter_values(test_param_values)
                
                # Run test simulation
                test_res = solver.simulate(start, end, points)
                
                # Capture all columns except 'time' and outcome_var
                all_columns = list(test_res.columns)
                species_to_capture = [col for col in all_columns if col != 'time']
                
                # Make sure outcome_var is included
                if outcome_var not in species_to_capture:
                    species_to_capture.append(outcome_var)
                    
            except Exception as e:
                # If test simulation fails, we'll try to get species from actual simulations
                warnings.warn(f"Could not discover species from test simulation: {e}")
                species_to_capture = []
        
        # Check if we need to store simulation data for return_details
        collect_timecourse_data = return_details and capture_all_species
        
        def simulate_with_values_single(feature_values: Dict[str, float], param_values: Optional[Dict[str, float]] = None) -> Tuple[Optional[float], Optional[np.ndarray]]:
            """Simulate with given values and return result and timecourse for outcome_var only."""
            try:
                # Set perturbed initial values into solver
                solver.set_state_values(feature_values)
                
                # Set perturbed kinetic parameters if provided
                if param_values is not None:
                    solver.set_parameter_values(param_values)
                
                # Run simulation
                res = solver.simulate(start, end, points)
                
                # Extract target value and time course
                target_value = res[outcome_var].iloc[-1]
                time_course = res[outcome_var].values
                
                return target_value, time_course
            except RuntimeError as e:
                # Check for CVODE errors
                if "CV_TOO_MUCH_WORK" in str(e) or "CVODE" in str(e):
                    return None, None
                else:
                    raise  # Re-raise unexpected errors
            except Exception as e:
                # Catch other solver errors
                return None, None
        
        def simulate_with_values_all_species(feature_values: Dict[str, float], param_values: Optional[Dict[str, float]] = None) -> Tuple[Optional[float], Optional[Dict[str, np.ndarray]]]:
            """Simulate with given values and return result and timecourses for all species."""
            try:
                # Set perturbed initial values into solver
                solver.set_state_values(feature_values)
                
                # Set perturbed kinetic parameters if provided
                if param_values is not None:
                    solver.set_parameter_values(param_values)
                
                # Run simulation
                res = solver.simulate(start, end, points)
                
                # Extract target value
                target_value = res[outcome_var].iloc[-1]
                
                # Capture timecourses for all species
                time_courses = {}
                for species in species_to_capture:
                    if species in res.columns:
                        time_courses[species] = res[species].values
                
                return target_value, time_courses
            except RuntimeError as e:
                # Check for CVODE errors
                if "CV_TOO_MUCH_WORK" in str(e) or "CVODE" in str(e):
                    return None, None
                else:
                    raise  # Re-raise unexpected errors
            except Exception as e:
                # Catch other solver errors
                return None, None
        
        # Sequential processing with error handling and resampling
        all_targets = []
        failed_indices = []
        
        if return_details:
            if capture_all_species:
                timecourse_data = []  # Will be converted to DataFrame later
            else:
                timecourse_data = []  # List of arrays
        
        # Create progress bar
        pbar = tqdm(range(feature_df.shape[0]), desc='Simulating perturbations', disable=not verbose)
        
        for i in pbar:
            success = False
            target_value = None
            time_course_data = None
            
            # Get original values
            original_feature_values = feature_df.iloc[i].to_dict()
            original_param_values = parameter_df.iloc[i].to_dict() if parameter_df is not None else None
            
            # Try original values first
            if return_details:
                if capture_all_species:
                    target_value, time_course_data = simulate_with_values_all_species(original_feature_values, original_param_values)
                else:
                    target_value, time_course_data = simulate_with_values_single(original_feature_values, original_param_values)
            else:
                # Use the simpler simulation for backward compatibility
                try:
                    # Set perturbed initial values into solver
                    solver.set_state_values(original_feature_values)
                    
                    # Set perturbed kinetic parameters if provided
                    if original_param_values is not None:
                        solver.set_parameter_values(original_param_values)
                    
                    # Run simulation
                    res = solver.simulate(start, end, points)
                    
                    # Extract target value
                    target_value = res[outcome_var].iloc[-1]
                except RuntimeError as e:
                    if "CV_TOO_MUCH_WORK" in str(e) or "CVODE" in str(e):
                        target_value = None
                    else:
                        raise
                except Exception:
                    target_value = None
            
            if target_value is not None:
                success = True
                if return_details and time_course_data is not None:
                    timecourse_data.append(time_course_data)
            else:
                # Try resampling up to max_retries times
                for attempt in range(max_retries):
                    pbar.set_description(f'Simulating perturbations (resampling {i}, attempt {attempt+1}/{max_retries})')
                    
                    # Generate batch alternatives for both feature and parameter values
                    feature_alternatives = generate_batch_alternatives(
                        initial_values, perturbation_type, perturbation_params,
                        resample_size, seed, attempt
                    )
                    
                    if parameter_df is not None:
                        param_alternatives = generate_batch_alternatives(
                            parameter_values, param_perturbation_type, param_perturbation_params,
                            resample_size, param_seed if param_seed is not None else seed, attempt
                        )
                    else:
                        param_alternatives = None
                    
                    # Test each alternative in the batch
                    for j in range(resample_size):
                        alt_feature_values = feature_alternatives.iloc[j].to_dict()
                        alt_param_values = param_alternatives.iloc[j].to_dict() if param_alternatives is not None else None
                        
                        if return_details:
                            if capture_all_species:
                                target_value, time_course_data = simulate_with_values_all_species(alt_feature_values, alt_param_values)
                            else:
                                target_value, time_course_data = simulate_with_values_single(alt_feature_values, alt_param_values)
                        else:
                            try:
                                solver.set_state_values(alt_feature_values)
                                if alt_param_values is not None:
                                    solver.set_parameter_values(alt_param_values)
                                res = solver.simulate(start, end, points)
                                target_value = res[outcome_var].iloc[-1]
                            except RuntimeError as e:
                                if "CV_TOO_MUCH_WORK" in str(e) or "CVODE" in str(e):
                                    target_value = None
                                else:
                                    raise
                            except Exception:
                                target_value = None
                        
                        if target_value is not None:
                            success = True
                            # Update the feature and parameter dataframes with successful alternative
                            feature_df.iloc[i] = pd.Series(alt_feature_values)
                            if parameter_df is not None and param_alternatives is not None:
                                parameter_df.iloc[i] = pd.Series(alt_param_values)
                            
                            if return_details and time_course_data is not None:
                                timecourse_data.append(time_course_data)
                            break
                    
                    if success:
                        break
            
            if success:
                all_targets.append(target_value)
            else:
                all_targets.append(np.nan)
                failed_indices.append(i)
                if return_details:
                    timecourse_data.append(None)  # Placeholder for failed simulation
                pbar.set_description(f'Simulating perturbations (failed: {len(failed_indices)})')
        
        # Update progress bar final message
        if failed_indices:
            pbar.set_description(f'Simulating perturbations (completed, {len(failed_indices)} failed)')
        else:
            pbar.set_description('Simulating perturbations (completed)')
        
        # Handle require_all_successful option
        if require_all_successful and failed_indices:
            raise RuntimeError(
                f"Failed to simulate {len(failed_indices)} samples after {max_retries} retries "
                f"with resample_size={resample_size}. Failed indices: {failed_indices[:10]}{'...' if len(failed_indices) > 10 else ''}"
            )
        
        # Create target DataFrame
        target_df = pd.DataFrame(all_targets, columns=[outcome_var])
        
        # Convert timecourse data to DataFrame if capture_all_species and return_details
        if return_details and capture_all_species and timecourse_data is not None:
            # Filter out None values (failed simulations)
            valid_timecourse_data = [tc for tc in timecourse_data if tc is not None]
            
            # Always create a DataFrame, even if empty
            if valid_timecourse_data:
                timecourse_df = pd.DataFrame(valid_timecourse_data)
                # Pad with None for failed simulations to maintain alignment
                if len(valid_timecourse_data) < len(timecourse_data):
                    # We need to align with original indices
                    aligned_timecourse_data = []
                    tc_idx = 0
                    for tc in timecourse_data:
                        if tc is None:
                            aligned_timecourse_data.append(None)
                        else:
                            aligned_timecourse_data.append(valid_timecourse_data[tc_idx])
                            tc_idx += 1
                    timecourse_data = pd.DataFrame(aligned_timecourse_data)
                else:
                    timecourse_data = timecourse_df
            else:
                # All simulations failed, create DataFrame with NaN values for all samples
                # First try to discover species columns from a test simulation
                species_columns = []
                try:
                    # Try to discover species from a test simulation
                    test_feature_values = feature_df.iloc[0].to_dict()
                    solver.set_state_values(test_feature_values)
                    
                    if parameter_df is not None:
                        test_param_values = parameter_df.iloc[0].to_dict()
                        solver.set_parameter_values(test_param_values)
                    
                    test_res = solver.simulate(start, end, points)
                    species_columns = [col for col in test_res.columns if col != 'time']
                except Exception as e:
                    # If test simulation also fails, try alternative methods to get species
                    warnings.warn(f"Could not discover species for empty DataFrame from simulation: {e}")
                    
                    # Alternative 1: Try to get species from feature_df columns
                    if len(feature_df.columns) > 0:
                        species_columns = list(feature_df.columns)
                    # Alternative 2: If still empty, create generic column names
                    elif not species_columns:
                        species_columns = ['species_{}'.format(i) for i in range(len(initial_values))]
                
                # Create DataFrame with n_samples rows (all NaN)
                nan_data = {col: [np.nan] * n_samples for col in species_columns}
                timecourse_data = pd.DataFrame(nan_data)
    else:
        # Return empty target DataFrame if no model/solver provided
        target_df = pd.DataFrame()
        timecourse_data = None if return_details else None
    
    # Return extended data structure if requested
    if return_details:
        metadata = {
            'failed_indices': failed_indices if 'failed_indices' in locals() else [],
            'success_rate': 1.0 if len(all_targets) == 0 else (1 - len(failed_indices) / len(all_targets)) if 'failed_indices' in locals() else 1.0,
            'n_samples': n_samples,
            'perturbation_type': perturbation_type,
            'capture_all_species': capture_all_species,
            'resampling_used': any(failed_indices) if 'failed_indices' in locals() else False
        }
        
        return {
            'features': feature_df,
            'targets': target_df,
            'parameters': parameter_df,
            'timecourse': timecourse_data,
            'metadata': metadata
        }
    else:
        return feature_df, target_df


def make_data_extended(
    initial_values: Dict[str, float],
    perturbation_type: str,
    perturbation_params: Dict[str, Any],
    n_samples: int,
    model_spec=None,
    solver=None,
    parameter_values: Optional[Dict[str, float]] = None,
    param_perturbation_type: str = 'none',
    param_perturbation_params: Optional[Dict[str, Any]] = None,
    simulation_params: Optional[Dict[str, Any]] = None,
    seed: Optional[int] = None,
    param_seed: Optional[int] = None,
    resample_size: int = 10,
    max_retries: int = 3,
    require_all_successful: bool = False,
    capture_all_species: bool = True,
    n_cores: int = 1,
    verbose: bool = False,
    **kwargs
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
        model_spec: Model specification (required for target generation)
        solver: Solver object (required for target generation)
        parameter_values: Dictionary of kinetic parameter values (optional)
        param_perturbation_type: Type of perturbation for kinetic parameters ('none', 'uniform', 'gaussian', 'lognormal', 'lhs')
        param_perturbation_params: Parameters for kinetic parameter perturbation (optional)
        simulation_params: Simulation parameters (optional)
        seed: Random seed for feature generation
        param_seed: Random seed for parameter generation (optional, uses seed if not provided)
        resample_size: Number of alternative samples to generate when a simulation fails (default: 10)
        max_retries: Maximum number of resampling attempts per failed index (default: 3)
        require_all_successful: Whether to require all samples to succeed (default: False)
        capture_all_species: If True (default), captures timecourses for all species in DataFrame format.
                           If False, captures only outcome variable timecourse as list of arrays.
        **kwargs: Additional arguments for target data generation
        
    Returns:
        Dictionary with keys:
            - 'features': Feature dataframe (initial values)
            - 'targets': Target dataframe (outcome values)
            - 'parameters': Kinetic parameters dataframe (None if not provided)
            - 'timecourse': Timecourse simulation data (DataFrame if capture_all_species=True, list of arrays if False)
            - 'metadata': Dictionary with metadata about the generation process
        
    Examples:
        >>> result = make_data_extended(
        ...     initial_values={'A': 10.0, 'B': 20.0},
        ...     perturbation_type='gaussian',
        ...     perturbation_params={'std': 2.0},
        ...     n_samples=100,
        ...     model_spec=model_spec,
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
        ...     model_spec=model_spec,
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
        model_spec=model_spec,
        solver=solver,
        parameter_values=parameter_values,
        param_perturbation_type=param_perturbation_type,
        param_perturbation_params=param_perturbation_params,
        simulation_params=simulation_params,
        seed=seed,
        param_seed=param_seed,
        resample_size=resample_size,
        max_retries=max_retries,
        require_all_successful=require_all_successful,
        return_details=True,
        capture_all_species=capture_all_species,
        n_cores=n_cores,
        verbose=verbose,
        **kwargs
    )


def add_deprecation_warning(
    old_function_name: str,
    new_function_name: str,
    stacklevel: int = 2
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
        stacklevel=stacklevel
    )