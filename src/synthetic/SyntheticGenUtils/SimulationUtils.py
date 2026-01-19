"""
Simulation utilities for synthetic data generation.

Contains reusable simulation workflow and solver-specific logic extracted 
from SyntheticGen.py to eliminate code duplication.
"""

import pandas as pd
from typing import Dict, Any, List
from models.Solver.Solver import Solver
from models.Solver.RoadrunnerSolver import RoadrunnerSolver
from models.Solver.ScipySolver import ScipySolver


def compile_solver(solver: Solver, model_build=None, sbml_str: str = None, ant_str: str = None) -> None:
    """
    Compile solver with appropriate model format based on solver type.
    
    Args:
        solver: Solver instance (RoadrunnerSolver or ScipySolver)
        model_build: ModelBuilder object (alternative to direct strings)
        sbml_str: SBML model string
        ant_str: Antimony model string
        
    Raises:
        ValueError: If solver type is not supported or model strings are missing
    """
    # Get model strings from model_build if provided
    if model_build is not None:
        sbml_str = model_build.get_sbml_model()
        ant_str = model_build.get_antimony_model()
    
    if sbml_str is None or ant_str is None:
        raise ValueError('Either model_build or both sbml_str and ant_str must be provided')
    
    # Choose appropriate model format based on solver type
    if isinstance(solver, RoadrunnerSolver):
        solver.compile(sbml_str)
    elif isinstance(solver, ScipySolver):
        solver.compile(ant_str)
    else:
        raise ValueError('Solver must be either ScipySolver or RoadrunnerSolver')


def set_simulation_values(solver: Solver, 
                         perturbed_values: Dict[str, float],
                         parameter_values: Dict[str, float] = None) -> None:
    """
    Set state and parameter values for simulation.
    
    Args:
        solver: Solver instance
        perturbed_values: Dictionary of state variable values
        parameter_values: Dictionary of parameter values (optional)
    """
    # Set state values
    solver.set_state_values(perturbed_values)
    
    # Set parameter values if provided
    if parameter_values:
        solver.set_parameter_values(parameter_values)


def extract_simulation_results(res: pd.DataFrame, 
                             outcome_var: str = 'Cp',
                             capture_time_course: bool = False,
                             capture_species: str = 'all',
                             all_species_dict: Dict = None) -> Any:
    """
    Extract results from simulation output based on capture options.
    
    Args:
        res: Simulation result DataFrame
        outcome_var: Variable to extract as primary result
        capture_time_course: Whether to capture time course data
        capture_species: Species to capture ('all' or list)
        all_species_dict: Dictionary of all species (required for 'all' capture)
        
    Returns:
        Extracted simulation results (scalar, array, or dictionary)
    """
    if capture_time_course:
        # Return last value and time course
        last_value = res[outcome_var].iloc[-1]
        time_course = res[outcome_var].values
        return last_value, time_course
    
    elif capture_species == 'all':
        if all_species_dict is None:
            raise ValueError('all_species_dict is required when capture_species="all"')
        
        output = {}
        for species in all_species_dict.keys():
            if species in res.columns:
                output[species] = res[species].values
        return output
    
    elif isinstance(capture_species, list):
        output = {}
        for species in capture_species:
            if species in res.columns:
                output[species] = res[species].values
                # Also capture phosphorylated version if it exists
                phosphorylated = species + 'p'
                if phosphorylated in res.columns:
                    output[phosphorylated] = res[phosphorylated].values
        return output
    
    else:
        # Default: return only the last value of outcome_var
        return res[outcome_var].iloc[-1]


def create_simulation_function(solver: Solver,
                             simulation_params: Dict[str, Any],
                             outcome_var: str = 'Cp',
                             capture_time_course: bool = False,
                             capture_species: str = 'all',
                             all_species_dict: Dict = None) -> callable:
    """
    Create a reusable simulation function with configured parameters.
    
    Args:
        solver: Solver instance
        simulation_params: Simulation parameters (start, end, points)
        outcome_var: Outcome variable to extract
        capture_time_course: Whether to capture time course
        capture_species: Species to capture
        all_species_dict: Dictionary of all species
        
    Returns:
        Configured simulation function
    """
    start = simulation_params['start']
    end = simulation_params['end']
    points = simulation_params['points']
    
    def simulate_function():
        """Run simulation with pre-configured parameters."""
        res = solver.simulate(start, end, points)
        return extract_simulation_results(res, outcome_var, capture_time_course, 
                                        capture_species, all_species_dict)
    
    return simulate_function


def validate_solver_type(solver, expected_types: List[type] = None) -> None:
    """
    Validate that solver is of expected type(s).
    
    Args:
        solver: Solver instance to validate
        expected_types: List of expected solver types
        
    Raises:
        ValueError: If solver is not of expected type
    """
    if expected_types is None:
        expected_types = [RoadrunnerSolver, ScipySolver]
    
    if not any(isinstance(solver, solver_type) for solver_type in expected_types):
        raise ValueError(f'Solver must be one of {expected_types}')


def get_simulation_timepoints(simulation_params: Dict[str, Any]) -> List[float]:
    """
    Generate simulation timepoints based on parameters.
    
    Args:
        simulation_params: Dictionary with start, end, points
        
    Returns:
        List of timepoints
    """
    start = simulation_params['start']
    end = simulation_params['end']
    points = simulation_params['points']
    
    return list(range(int(start), int(end) + 1, int((end - start) / points)))
