"""
Time course data generation utilities.

This module provides functions for generating time course data
from model simulations.
"""

import warnings
from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed

from models.Solver.Solver import Solver
from models.Solver.ScipySolver import ScipySolver
from models.Solver.RoadrunnerSolver import RoadrunnerSolver
from models.ModelBuilder import ModelBuilder
from models.Utils import ModelSpecification


def make_timecourse_data(
    model_spec: ModelSpecification,
    solver: Solver,
    feature_df: pd.DataFrame,
    simulation_params: Dict[str, Any] = None,
    capture_species: Union[str, List[str]] = 'all',
    n_cores: int = 1,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Generate time course data for a model.
    
    Args:
        model_spec: ModelSpecification object
        solver: Solver object
        feature_df: DataFrame of perturbed values
        simulation_params: Dictionary with 'start', 'end', 'points' keys
        capture_species: 'all' or list of species to capture
        n_cores: Number of cores for parallel processing (-1 for all cores)
        verbose: Whether to show progress bar
        
    Returns:
        DataFrame with time course data for each perturbation
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
    if capture_species == 'all':
        species_to_capture = model_spec.A_species + model_spec.B_species + model_spec.C_species
        # Also capture phosphorylated versions
        species_to_capture_with_phospho = []
        for s in species_to_capture:
            species_to_capture_with_phospho.extend([s, s + 'p'])
    else:
        species_to_capture_with_phospho = []
        for s in capture_species:
            species_to_capture_with_phospho.extend([s, s + 'p'])
    
    def simulate_perturbation(i: int) -> Dict[str, np.ndarray]:
        """Simulate a single perturbation and capture time courses."""
        perturbed_values = feature_df.iloc[i].to_dict()
        
        # Set perturbed values and simulate
        solver.set_state_values(perturbed_values)
        res = solver.simulate(start, end, points)
        
        # Capture specified species
        output = {}
        for species in species_to_capture_with_phospho:
            if species in res.columns:
                output[species] = res[species].values
        
        return output
    
    # Use parallel processing if requested
    if n_cores > 1 or n_cores == -1:
        results = Parallel(n_jobs=n_cores)(
            delayed(simulate_perturbation)(i)
            for i in tqdm(range(feature_df.shape[0]), 
                         desc='Simulating perturbations', 
                         disable=not verbose)
        )
        all_outputs = list(results)
    else:
        # Sequential processing
        all_outputs = []
        for i in tqdm(range(feature_df.shape[0]), 
                     desc='Simulating perturbations', 
                     disable=not verbose):
            output = simulate_perturbation(i)
            all_outputs.append(output)
    
    # Create output DataFrame
    output_df = pd.DataFrame(all_outputs)
    
    return output_df


def make_timecourse_data_diff_spec(
    model_builds: List[ModelBuilder],
    SolverClass: type[Solver],
    feature_df: pd.DataFrame,
    simulation_params: Dict[str, Any] = None,
    capture_species: Union[str, List[str]] = 'all',
    n_cores: int = 1,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Generate time course data using different model specifications.
    
    Args:
        model_builds: List of ModelBuilder objects (one per row in feature_df)
        SolverClass: Solver class (ScipySolver or RoadrunnerSolver)
        feature_df: DataFrame of perturbed values
        simulation_params: Dictionary with 'start', 'end', 'points' keys
        capture_species: 'all' or list of species to capture
        n_cores: Number of cores for parallel processing (-1 for all cores)
        verbose: Whether to show progress bar
        
    Returns:
        DataFrame with time course data for each perturbation
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
    
    def simulate_perturbation(i: int) -> Dict[str, np.ndarray]:
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
        
        # Capture specified species
        output = {}
        if capture_species == 'all':
            all_species = model_build.get_state_variables().keys()
            for species in all_species:
                output[species] = res[species].values
        else:
            for species in capture_species:
                output[species] = res[species].values
                phospho_species = species + 'p'
                output[phospho_species] = res[phospho_species].values
        
        return output
    
    # Use parallel processing if requested
    if n_cores > 1 or n_cores == -1:
        results = Parallel(n_jobs=n_cores)(
            delayed(simulate_perturbation)(i)
            for i in tqdm(range(feature_df.shape[0]), 
                         desc='Simulating perturbations', 
                         disable=not verbose)
        )
        all_outputs = list(results)
    else:
        # Sequential processing with error handling
        all_outputs = []
        for i in tqdm(range(feature_df.shape[0]), 
                     desc='Simulating perturbations', 
                     disable=not verbose):
            try:
                output = simulate_perturbation(i)
                all_outputs.append(output)
            except Exception as e:
                if verbose:
                    print(f'Error simulating perturbation {i}: {e}')
                all_outputs.append({})
    
    # Create output DataFrame
    output_df = pd.DataFrame(all_outputs)
    
    # Check if output is empty
    if output_df.empty:
        raise ValueError('Output dataframe is empty, check the model specifications and feature dataframe')
    
    return output_df


def make_timecourse_data_diff_build(
    model_spec: ModelSpecification,
    solver: Solver,
    feature_df: pd.DataFrame,
    parameter_set: List[Dict[str, float]],
    simulation_params: Dict[str, Any] = None,
    capture_species: Union[str, List[str]] = 'all',
    n_cores: int = 1,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Generate time course data using different parameter sets.
    
    Args:
        model_spec: ModelSpecification object
        solver: Solver object
        feature_df: DataFrame of perturbed values
        parameter_set: List of parameter dictionaries (one per row in feature_df)
        simulation_params: Dictionary with 'start', 'end', 'points' keys
        capture_species: 'all' or list of species to capture
        n_cores: Number of cores for parallel processing (-1 for all cores)
        verbose: Whether to show progress bar
        
    Returns:
        DataFrame with time course data for each perturbation
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
    
    def simulate_perturbation(i: int) -> Dict[str, np.ndarray]:
        """Simulate a single perturbation with specific parameter set."""
        perturbed_values = feature_df.iloc[i].to_dict()
        params = parameter_set[i]
        
        # Set perturbed values and parameters, then simulate
        solver.set_state_values(perturbed_values)
        solver.set_parameter_values(params)
        res = solver.simulate(start, end, points)
        
        # Capture specified species
        output = {}
        if capture_species == 'all':
            all_species = model_spec.A_species + model_spec.B_species + model_spec.C_species
            for species in all_species:
                output[species] = res[species].values
                phospho_species = species + 'p'
                output[phospho_species] = res[phospho_species].values
        else:
            for species in capture_species:
                output[species] = res[species].values
                phospho_species = species + 'p'
                output[phospho_species] = res[phospho_species].values
        
        return output
    
    # Use parallel processing if requested
    if n_cores > 1 or n_cores == -1:
        results = Parallel(n_jobs=n_cores)(
            delayed(simulate_perturbation)(i)
            for i in tqdm(range(feature_df.shape[0]), 
                         desc='Simulating perturbations', 
                         disable=not verbose)
        )
        all_outputs = list(results)
    else:
        # Sequential processing with error handling
        all_outputs = []
        for i in tqdm(range(feature_df.shape[0]), 
                     desc='Simulating perturbations', 
                     disable=not verbose):
            try:
                output = simulate_perturbation(i)
                all_outputs.append(output)
            except Exception as e:
                if verbose:
                    print(f'Error simulating perturbation {i}: {e}')
                all_outputs.append({})
    
    # Create output DataFrame
    output_df = pd.DataFrame(all_outputs)
    
    return output_df


# Simplified versions for v3 API
def make_timecourse_data_v3(
    all_species: Dict[str, Any],
    solver: Solver,
    feature_df: pd.DataFrame,
    simulation_params: Dict[str, Any] = None,
    n_cores: int = 1,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Generate time course data (v3 API).
    
    Simplified version that works with dictionary of all species.
    
    Args:
        all_species: Dictionary of all species (keys are species names)
        solver: Solver object
        feature_df: DataFrame of perturbed values
        simulation_params: Dictionary with 'start', 'end', 'points' keys
        n_cores: Number of cores for parallel processing (-1 for all cores)
        verbose: Whether to show progress bar
        
    Returns:
        DataFrame with time course data for each perturbation
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
    
    def simulate_perturbation(i: int) -> Dict[str, np.ndarray]:
        """Simulate a single perturbation."""
        perturbed_values = feature_df.iloc[i].to_dict()
        
        # Set perturbed values and simulate
        solver.set_state_values(perturbed_values)
        res = solver.simulate(start, end, points)
        
        # Capture all species from the dictionary
        output = {}
        for species in all_species.keys():
            if species in res.columns:
                output[species] = res[species].values
        
        return output
    
    # Use parallel processing if requested
    if n_cores > 1 or n_cores == -1:
        results = Parallel(n_jobs=n_cores)(
            delayed(simulate_perturbation)(i)
            for i in tqdm(range(feature_df.shape[0]), 
                         desc='Simulating perturbations', 
                         disable=not verbose)
        )
        all_outputs = list(results)
    else:
        # Sequential processing
        all_outputs = []
        for i in tqdm(range(feature_df.shape[0]), 
                     desc='Simulating perturbations', 
                     disable=not verbose):
            output = simulate_perturbation(i)
            all_outputs.append(output)
    
    # Create output DataFrame
    output_df = pd.DataFrame(all_outputs)
    
    return output_df


def make_timecourse_data_diff_build_v3(
    all_species: Dict[str, Any],
    solver: Solver,
    feature_df: pd.DataFrame,
    parameter_set: List[Dict[str, float]],
    simulation_params: Dict[str, Any] = None,
    n_cores: int = 1,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Generate time course data using different parameter sets (v3 API).
    
    Args:
        all_species: Dictionary of all species (keys are species names)
        solver: Solver object
        feature_df: DataFrame of perturbed values
        parameter_set: List of parameter dictionaries (one per row in feature_df)
        simulation_params: Dictionary with 'start', 'end', 'points' keys
        n_cores: Number of cores for parallel processing (-1 for all cores)
        verbose: Whether to show progress bar
        
    Returns:
        DataFrame with time course data for each perturbation
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
    
    def simulate_perturbation(i: int) -> Dict[str, np.ndarray]:
        """Simulate a single perturbation with specific parameter set."""
        perturbed_values = feature_df.iloc[i].to_dict()
        params = parameter_set[i]
        
        # Set perturbed values and parameters, then simulate
        solver.set_state_values(perturbed_values)
        solver.set_parameter_values(params)
        res = solver.simulate(start, end, points)
        
        # Capture all species from the dictionary
        output = {}
        for species in all_species.keys():
            if species in res.columns:
                output[species] = res[species].values
        
        return output
    
    # Use parallel processing if requested
    if n_cores > 1 or n_cores == -1:
        results = Parallel(n_jobs=n_cores)(
            delayed(simulate_perturbation)(i)
            for i in tqdm(range(feature_df.shape[0]), 
                         desc='Simulating perturbations', 
                         disable=not verbose)
        )
        all_outputs = list(results)
    else:
        # Sequential processing with error handling
        all_outputs = []
        for i in tqdm(range(feature_df.shape[0]), 
                     desc='Simulating perturbations', 
                     disable=not verbose):
            try:
                output = simulate_perturbation(i)
                all_outputs.append(output)
            except Exception as e:
                if verbose:
                    print(f'Error simulating perturbation {i}: {e}')
                all_outputs.append({})
    
    # Create output DataFrame
    output_df = pd.DataFrame(all_outputs)
    
    return output_df


# Backward compatibility wrappers (with deprecation warnings)
def generate_model_timecourse_data(
    model_spec: ModelSpecification,
    solver: Solver,
    feature_df: pd.DataFrame,
    simulation_params: Dict[str, Any] = None,
    capture_species: Union[str, List[str]] = 'all',
    n_cores: int = 1,
    verbose: bool = False
) -> pd.DataFrame:
    """
    DEPRECATED: Use make_timecourse_data instead.
    
    Backward compatibility wrapper for generate_model_timecourse_data.
    """
    warnings.warn(
        "generate_model_timecourse_data is deprecated. Use make_timecourse_data instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return make_timecourse_data(
        model_spec=model_spec,
        solver=solver,
        feature_df=feature_df,
        simulation_params=simulation_params,
        capture_species=capture_species,
        n_cores=n_cores,
        verbose=verbose
    )


def generate_model_timecourse_data_diff_spec(
    model_builds: List[ModelBuilder],
    SolverClass: type[Solver],
    feature_df: pd.DataFrame,
    simulation_params: Dict[str, Any] = None,
    capture_species: Union[str, List[str]] = 'all',
    n_cores: int = 1,
    verbose: bool = False
) -> pd.DataFrame:
    """
    DEPRECATED: Use make_timecourse_data_diff_spec instead.
    
    Backward compatibility wrapper for generate_model_timecourse_data_diff_spec.
    """
    warnings.warn(
        "generate_model_timecourse_data_diff_spec is deprecated. Use make_timecourse_data_diff_spec instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return make_timecourse_data_diff_spec(
        model_builds=model_builds,
        SolverClass=SolverClass,
        feature_df=feature_df,
        simulation_params=simulation_params,
        capture_species=capture_species,
        n_cores=n_cores,
        verbose=verbose
    )


def generate_model_timecourse_data_diff_build(
    model_spec: ModelSpecification,
    solver: Solver,
    feature_df: pd.DataFrame,
    parameter_set: List[Dict[str, float]],
    simulation_params: Dict[str, Any] = None,
    capture_species: Union[str, List[str]] = 'all',
    n_cores: int = 1,
    verbose: bool = False
) -> pd.DataFrame:
    """
    DEPRECATED: Use make_timecourse_data_diff_build instead.
    
    Backward compatibility wrapper for generate_model_timecourse_data_diff_build.
    """
    warnings.warn(
        "generate_model_timecourse_data_diff_build is deprecated. Use make_timecourse_data_diff_build instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return make_timecourse_data_diff_build(
        model_spec=model_spec,
        solver=solver,
        feature_df=feature_df,
        parameter_set=parameter_set,
        simulation_params=simulation_params,
        capture_species=capture_species,
        n_cores=n_cores,
        verbose=verbose
    )


def generate_model_timecourse_data_v3(
    all_species: Dict[str, Any],
    solver: Solver,
    feature_df: pd.DataFrame,
    simulation_params: Dict[str, Any] = None,
    n_cores: int = 1,
    verbose: bool = False
) -> pd.DataFrame:
    """
    DEPRECATED: Use make_timecourse_data_v3 instead.
    
    Backward compatibility wrapper for generate_model_timecourse_data_v3.
    """
    warnings.warn(
        "generate_model_timecourse_data_v3 is deprecated. Use make_timecourse_data_v3 instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return make_timecourse_data_v3(
        all_species=all_species,
        solver=solver,
        feature_df=feature_df,
        simulation_params=simulation_params,
        n_cores=n_cores,
        verbose=verbose
    )


def generate_model_timecourse_data_diff_build_v3(
    all_species: Dict[str, Any],
    solver: Solver,
    feature_df: pd.DataFrame,
    parameter_set: List[Dict[str, float]],
    simulation_params: Dict[str, Any] = None,
    n_cores: int = 1,
    verbose: bool = False
) -> pd.DataFrame:
    """
    DEPRECATED: Use make_timecourse_data_diff_build_v3 instead.
    
    Backward compatibility wrapper for generate_model_timecourse_data_diff_build_v3.
    """
    warnings.warn(
        "generate_model_timecourse_data_diff_build_v3 is deprecated. Use make_timecourse_data_diff_build_v3 instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    return make_timecourse_data_diff_build_v3(
        all_species=all_species,
        solver=solver,
        feature_df=feature_df,
        parameter_set=parameter_set,
        simulation_params=simulation_params,
        n_cores=n_cores,
        verbose=verbose
    )
