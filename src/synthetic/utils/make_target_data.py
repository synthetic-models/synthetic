"""
Target data generation utilities.

This module provides functions for generating target/simulation data
using model specifications and solvers.
"""

import warnings
from typing import Dict, Any, Optional, List, Tuple, Union
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from models.Solver.Solver import Solver
from models.Solver.ScipySolver import ScipySolver
from models.Solver.RoadrunnerSolver import RoadrunnerSolver
from models.ModelBuilder import ModelBuilder
from models.Utils import ModelSpecification


def make_target_data(
    model_spec: ModelSpecification,
    solver: Solver,
    feature_df: pd.DataFrame,
    simulation_params: Dict[str, Any] = None,
    n_cores: int = 1,
    outcome_var: str = 'Cp',
    verbose: bool = False
) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    """
    Generate target data for a model.
    
    Args:
        model_spec: ModelSpecification object
        solver: Solver object (ScipySolver or RoadrunnerSolver)
        feature_df: DataFrame of perturbed values
        simulation_params: Dictionary with 'start', 'end', 'points' keys
        n_cores: Number of cores for parallel processing (-1 for all cores)
        outcome_var: Variable to extract as target
        verbose: Whether to show progress bar
        
    Returns:
        Tuple of (target_df, time_course_data)
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
    
    def simulate_perturbation(i: int) -> Tuple[float, np.ndarray]:
        """Simulate a single perturbation."""
        perturbed_values = feature_df.iloc[i].to_dict()
        
        # Set perturbed values into solver
        solver.set_state_values(perturbed_values)
        
        # Run simulation
        res = solver.simulate(start, end, points)
        
        # Extract target value and time course
        target_value = res[outcome_var].iloc[-1]
        time_course = res[outcome_var].values
        
        return target_value, time_course
    
    # Use parallel processing if requested
    if n_cores > 1 or n_cores == -1:
        results = Parallel(n_jobs=n_cores)(
            delayed(simulate_perturbation)(i)
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
            target_value, time_course = simulate_perturbation(i)
            all_targets.append(target_value)
            time_course_data.append(time_course)
    
    # Create target DataFrame
    target_df = pd.DataFrame(all_targets, columns=[outcome_var])
    
    return target_df, time_course_data


def make_target_data_diff_spec(
    model_builds: List[ModelBuilder],
    SolverClass: type[Solver],
    feature_df: pd.DataFrame,
    simulation_params: Dict[str, Any] = None,
    n_cores: int = 1,
    verbose: bool = False
) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    """
    Generate target data using different model specifications.
    
    Args:
        model_builds: List of ModelBuilder objects (one per row in feature_df)
        SolverClass: Solver class (ScipySolver or RoadrunnerSolver)
        feature_df: DataFrame of perturbed values
        simulation_params: Dictionary with 'start', 'end', 'points' keys
        n_cores: Number of cores for parallel processing (-1 for all cores)
        verbose: Whether to show progress bar
        
    Returns:
        Tuple of (target_df, time_course_data)
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
    
    def simulate_perturbation(i: int) -> Tuple[float, np.ndarray]:
        """Simulate a single perturbation with specific model specification."""
        perturbed_values = feature_df.iloc[i].to_dict()
        model_build = model_builds[i]
        
        # Create and compile solver
        solver = SolverClass()
        sbml_str = model_build.get_sbml_model()
        ant_str = model_build.get_antimony_model()
        
        if isinstance(solver, RoadrunnerSolver):
            solver.compile(sbml_str)
        elif isinstance(solver, ScipySolver):
            solver.compile(ant_str)
        else:
            raise ValueError('Solver must be either ScipySolver or RoadrunnerSolver')
        
        # Set perturbed values and simulate
        solver.set_state_values(perturbed_values)
        res = solver.simulate(start, end, points)
        
        # Extract target value (Cp) and time course
        target_value = res['Cp'].iloc[-1]
        time_course = res['Cp'].values
        
        return target_value, time_course
    
    # Use parallel processing if requested
    if n_cores > 1 or n_cores == -1:
        results = Parallel(n_jobs=n_cores)(
            delayed(simulate_perturbation)(i)
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
            target_value, time_course = simulate_perturbation(i)
            all_targets.append(target_value)
            time_course_data.append(time_course)
    
    # Create target DataFrame
    target_df = pd.DataFrame(all_targets, columns=['Cp'])
    
    return target_df, time_course_data


def make_target_data_diff_build(
    model_spec: ModelSpecification,
    solver: Solver,
    feature_df: pd.DataFrame,
    parameter_set: List[Dict[str, float]],
    simulation_params: Dict[str, Any] = None,
    outcome_var: str = 'Cp',
    n_cores: int = 1,
    verbose: bool = False
) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    """
    Generate target data using different parameter sets.
    
    Args:
        model_spec: ModelSpecification object
        solver: Solver object
        feature_df: DataFrame of perturbed values
        parameter_set: List of parameter dictionaries (one per row in feature_df)
        simulation_params: Dictionary with 'start', 'end', 'points' keys
        outcome_var: Variable to extract as target
        n_cores: Number of cores for parallel processing (-1 for all cores)
        verbose: Whether to show progress bar
        
    Returns:
        Tuple of (target_df, time_course_data)
    """
    # Set default simulation parameters
    if simulation_params is None:
        simulation_params = {'start': 0, 'end': 500, 'points': 100}
    
    # Validate simulation parameters
    if 'start' not in simulation_params or 'end' not in simulation_params or 'points' not in simulation_params:
        raise ValueError('Simulation parameters must contain "start", "end" and "points" keys')
    
    # Validate parameter set length matches feature dataframe
    if len(parameter_set) != feature_df.shape[0]:
        raise ValueError(f'Parameter set length ({len(parameter_set)}) must match feature dataframe rows ({feature_df.shape[0]})')
    
    start = simulation_params['start']
    end = simulation_params['end']
    points = simulation_params['points']
    
    def simulate_perturbation(i: int) -> Tuple[float, np.ndarray]:
        """Simulate a single perturbation with specific parameter set."""
        perturbed_values = feature_df.iloc[i].to_dict()
        params = parameter_set[i]
        
        # Set perturbed values and parameters
        solver.set_state_values(perturbed_values)
        solver.set_parameter_values(params)
        
        # Run simulation
        res = solver.simulate(start, end, points)
        
        # Extract target value and time course
        target_value = res[outcome_var].iloc[-1]
        time_course = res[outcome_var].values
        
        return target_value, time_course
    
    # Use parallel processing if requested
    if n_cores > 1 or n_cores == -1:
        results = Parallel(n_jobs=n_cores)(
            delayed(simulate_perturbation)(i)
            for i in tqdm(range(feature_df.shape[0]), 
                         desc='Simulating perturbations', 
                         disable=not verbose)
        )
        all_targets, time_course_data = zip(*results)
        all_targets = list(all_targets)
        time_course_data = list(time_course_data)
    else:
        # Sequential processing with error handling
        all_targets = []
        time_course_data = []
        
        for i in tqdm(range(feature_df.shape[0]), 
                     desc='Simulating perturbations', 
                     disable=not verbose):
            try:
                target_value, time_course = simulate_perturbation(i)
                all_targets.append(target_value)
                time_course_data.append(time_course)
            except Exception as e:
                if verbose:
                    print(f'Error simulating perturbation {i}: {e}')
                # Add NaN for failed simulation
                all_targets.append(np.nan)
                time_course_data.append(np.full(points, np.nan))
    
    # Create target DataFrame
    target_df = pd.DataFrame(all_targets, columns=[outcome_var])
    
    return target_df, time_course_data


# Backward compatibility wrappers (with deprecation warnings)
def generate_target_data(
    model_spec: ModelSpecification,
    solver: Solver,
    feature_df: pd.DataFrame,
    simulation_params: Dict[str, Any] = None,
    n_cores: int = 1,
    outcome_var: str = 'Cp',
    verbose: bool = False
) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    """
    DEPRECATED: Use make_target_data instead.
    
    Backward compatibility wrapper for generate_target_data.
    """
    warnings.warn(
        "generate_target_data is deprecated. Use make_target_data instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return make_target_data(
        model_spec=model_spec,
        solver=solver,
        feature_df=feature_df,
        simulation_params=simulation_params,
        n_cores=n_cores,
        outcome_var=outcome_var,
        verbose=verbose
    )


def generate_target_data_diff_spec(
    model_builds: List[ModelBuilder],
    SolverClass: type[Solver],
    feature_df: pd.DataFrame,
    simulation_params: Dict[str, Any] = None,
    n_cores: int = 1,
    verbose: bool = False
) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    """
    DEPRECATED: Use make_target_data_diff_spec instead.
    
    Backward compatibility wrapper for generate_target_data_diff_spec.
    """
    warnings.warn(
        "generate_target_data_diff_spec is deprecated. Use make_target_data_diff_spec instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return make_target_data_diff_spec(
        model_builds=model_builds,
        SolverClass=SolverClass,
        feature_df=feature_df,
        simulation_params=simulation_params,
        n_cores=n_cores,
        verbose=verbose
    )


def generate_target_data_diff_build(
    model_spec: ModelSpecification,
    solver: Solver,
    feature_df: pd.DataFrame,
    parameter_set: List[Dict[str, float]],
    simulation_params: Dict[str, Any] = None,
    outcome_var: str = 'Cp',
    n_cores: int = 1,
    verbose: bool = False
) -> Tuple[pd.DataFrame, List[np.ndarray]]:
    """
    DEPRECATED: Use make_target_data_diff_build instead.
    
    Backward compatibility wrapper for generate_target_data_diff_build.
    """
    warnings.warn(
        "generate_target_data_diff_build is deprecated. Use make_target_data_diff_build instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return make_target_data_diff_build(
        model_spec=model_spec,
        solver=solver,
        feature_df=feature_df,
        parameter_set=parameter_set,
        simulation_params=simulation_params,
        outcome_var=outcome_var,
        n_cores=n_cores,
        verbose=verbose
    )
