"""
Parallel processing utilities for synthetic data generation.

Contains reusable parallel and sequential processing logic extracted from 
SyntheticGen.py to eliminate code duplication.
"""

from typing import Callable, List, Any, Tuple
from joblib import Parallel, delayed
from tqdm import tqdm


def run_parallel_simulation(simulation_function: Callable, 
                          n_iterations: int,
                          n_cores: int = 1,
                          verbose: bool = False,
                          description: str = "Simulating perturbations") -> List[Any]:
    """
    Run simulations in parallel using joblib.
    
    Args:
        simulation_function: Function to execute for each iteration
        n_iterations: Number of iterations to run
        n_cores: Number of cores to use (-1 for all available)
        verbose: Whether to show progress bar
        description: Description for progress bar
        
    Returns:
        List of simulation results
    """
    if n_cores > 1 or n_cores == -1:
        results = Parallel(n_jobs=n_cores)(
            delayed(simulation_function)(i) 
            for i in tqdm(range(n_iterations), desc=description, disable=not verbose)
        )
        return list(results)
    else:
        return run_sequential_simulation(simulation_function, n_iterations, verbose, description)


def run_sequential_simulation(simulation_function: Callable,
                            n_iterations: int,
                            verbose: bool = False,
                            description: str = "Simulating perturbations") -> List[Any]:
    """
    Run simulations sequentially with optional progress bar.
    
    Args:
        simulation_function: Function to execute for each iteration
        n_iterations: Number of iterations to run
        verbose: Whether to show progress bar
        description: Description for progress bar
        
    Returns:
        List of simulation results
    """
    results = []
    for i in tqdm(range(n_iterations), desc=description, disable=not verbose):
        results.append(simulation_function(i))
    return results


def handle_simulation_error(error_message: str, iteration: int, exception: Exception) -> None:
    """
    Handle simulation errors consistently across functions.
    
    Args:
        error_message: Custom error message prefix
        iteration: Current iteration number
        exception: Exception that occurred
    """
    print(f'{error_message} {iteration}: {exception}')


def run_parallel_with_error_handling(simulation_function: Callable,
                                   n_iterations: int,
                                   n_cores: int = 1,
                                   verbose: bool = False,
                                   description: str = "Simulating perturbations",
                                   error_prefix: str = "Error simulating perturbation") -> List[Any]:
    """
    Run simulations in parallel with built-in error handling.
    
    Args:
        simulation_function: Function to execute for each iteration
        n_iterations: Number of iterations to run
        n_cores: Number of cores to use (-1 for all available)
        verbose: Whether to show progress bar
        description: Description for progress bar
        error_prefix: Prefix for error messages
        
    Returns:
        List of simulation results (failed iterations return None)
    """
    def safe_simulation_function(i):
        try:
            return simulation_function(i)
        except Exception as e:
            handle_simulation_error(error_prefix, i, e)
            return None
    
    if n_cores > 1 or n_cores == -1:
        results = Parallel(n_jobs=n_cores)(
            delayed(safe_simulation_function)(i) 
            for i in tqdm(range(n_iterations), desc=description, disable=not verbose)
        )
        # Filter out None results (failed simulations)
        return [result for result in results if result is not None]
    else:
        return run_sequential_with_error_handling(simulation_function, n_iterations, verbose, description, error_prefix)


def run_sequential_with_error_handling(simulation_function: Callable,
                                     n_iterations: int,
                                     verbose: bool = False,
                                     description: str = "Simulating perturbations",
                                     error_prefix: str = "Error simulating perturbation") -> List[Any]:
    """
    Run simulations sequentially with built-in error handling.
    
    Args:
        simulation_function: Function to execute for each iteration
        n_iterations: Number of iterations to run
        verbose: Whether to show progress bar
        description: Description for progress bar
        error_prefix: Prefix for error messages
        
    Returns:
        List of simulation results (failed iterations are skipped)
    """
    results = []
    for i in tqdm(range(n_iterations), desc=description, disable=not verbose):
        try:
            result = simulation_function(i)
            results.append(result)
        except Exception as e:
            handle_simulation_error(error_prefix, i, e)
    return results


def split_parallel_results(results: List[Tuple], num_expected_results: int = 2) -> Tuple[List, ...]:
    """
    Split parallel results that return tuples into separate lists.
    
    Args:
        results: List of tuples from parallel processing
        num_expected_results: Number of expected elements in each tuple
        
    Returns:
        Tuple of lists containing unpacked results
    """
    if not results:
        return tuple([] for _ in range(num_expected_results))
    
    # Verify all results have the expected number of elements
    for i, result in enumerate(results):
        if not isinstance(result, tuple) or len(result) != num_expected_results:
            raise ValueError(f"Result {i} has unexpected format: {result}")
    
    # Unpack the results
    unpacked_results = zip(*results)
    return tuple(list(result) for result in unpacked_results)
