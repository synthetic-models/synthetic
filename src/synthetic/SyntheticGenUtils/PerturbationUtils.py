"""
Perturbation generation utilities for synthetic data generation.

Contains reusable perturbation generation logic extracted from SyntheticGen.py
to eliminate code duplication.
"""

import numpy as np
import pandas as pd
from scipy.stats import qmc
from typing import Dict, Any, List


def apply_uniform_perturbation(initial_values: Dict[str, float],
                                min_: float, max_: float,
                                rng: np.random.Generator = None) -> Dict[str, float]:
    """
    Generate uniform perturbation for all species.
    
    Args:
        initial_values: Dictionary of initial species values
        min_: Minimum multiplier for uniform distribution
        max_: Maximum multiplier for uniform distribution
        rng: Random number generator (optional)
        
    Returns:
        Dictionary of perturbed values
    """
    perturbed_values = {}
    
    if rng is None:
        rng = np.random.default_rng()
    
    for species, initial_value in initial_values.items():
        perturbed_values[species] = initial_value * rng.uniform(min_, max_)
    
    return perturbed_values


def apply_gaussian_perturbation(initial_values: Dict[str, float],
                                 perturbation_params: Dict[str, float],
                                 rng: np.random.Generator = None) -> Dict[str, float]:
    """
    Generate Gaussian perturbation for all species.
    
    Args:
        initial_values: Dictionary of initial species values
        perturbation_params: Parameters with 'std' or 'rsd' key
        rng: Random number generator (optional)
        
    Returns:
        Dictionary of perturbed values
    """
    perturbed_values = {}
    
    if rng is None:
        rng = np.random.default_rng()
    
    for species, initial_value in initial_values.items():
        mu = initial_value
        
        # Calculate standard deviation
        if 'std' in perturbation_params:
            sigma = perturbation_params['std']
        elif 'rsd' in perturbation_params:
            # Relative standard deviation: std = rsd * mean
            sigma = perturbation_params['rsd'] * initial_value
        else:
            # Default to small perturbation if no parameters provided
            sigma = 0.1 * initial_value
        
        perturbed_values[species] = rng.normal(mu, sigma)
    
    return perturbed_values


def apply_lognormal_perturbation(initial_values: Dict[str, float],
                                 perturbation_params: Dict[str, float],
                                 rng: np.random.Generator = None) -> Dict[str, float]:
    """
    Generate lognormal perturbation for all species.
    
    Args:
        initial_values: Dictionary of initial species values (must be > 0)
        perturbation_params: Parameters with 'shape' or 'rsd_shape' key
            - 'shape': shape parameter (sigma) of the underlying normal distribution
            - 'rsd_shape': relative shape parameter (shape / log(initial_value))
        rng: Random number generator (optional)
        
    Returns:
        Dictionary of perturbed values
        
    Raises:
        ValueError: If initial values are not positive or parameters are invalid
    """
    perturbed_values = {}
    
    if rng is None:
        rng = np.random.default_rng()
    
    # Validate initial values are positive for lognormal
    for species, initial_value in initial_values.items():
        if initial_value <= 0:
            raise ValueError(f'Initial value for {species} must be positive for lognormal distribution')
    
    # Validate perturbation parameters
    if 'shape' not in perturbation_params and 'rsd_shape' not in perturbation_params:
        raise ValueError('For lognormal perturbation, parameters must contain "shape" or "rsd_shape"')
    
    if 'shape' in perturbation_params and perturbation_params['shape'] <= 0:
        raise ValueError('Shape parameter must be positive')
    if 'rsd_shape' in perturbation_params and perturbation_params['rsd_shape'] <= 0:
        raise ValueError('Relative shape parameter must be positive')
    
    for species, initial_value in initial_values.items():
        # For lognormal, median = exp(mean of underlying normal)
        # We want median = initial_value, so mean = log(initial_value)
        mean = np.log(initial_value)
        
        # Calculate shape parameter (sigma of underlying normal)
        if 'shape' in perturbation_params:
            sigma = perturbation_params['shape']
        elif 'rsd_shape' in perturbation_params:
            # Relative shape: sigma = rsd_shape * log(initial_value)
            # But we need to handle case where initial_value = 1 -> log(1) = 0
            if initial_value == 1:
                raise ValueError(f'Cannot use rsd_shape with initial_value = 1 for {species}')
            sigma = perturbation_params['rsd_shape'] * np.log(initial_value)
        else:
            # Default shape if no parameters provided
            sigma = 0.1
        
        # Generate lognormal sample: lognormal(mean, sigma)
        # numpy's lognormal uses mean and sigma of underlying normal distribution
        perturbed_values[species] = rng.lognormal(mean, sigma)
    
    return perturbed_values


def generate_lhs_perturbation(n_samples: int, n_features: int,
                            min_: float, max_: float,
                            seed: int = None) -> pd.DataFrame:
    """
    Generate Latin Hypercube Sampling perturbation.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features/dimensions
        min_: Minimum value for scaling
        max_: Maximum value for scaling
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with LHS samples
    """
    # Initialize LHS sampler
    sampler = qmc.LatinHypercube(d=n_features, seed=seed)
    lhs_samples = sampler.random(n=n_samples)
    
    # Scale to [min_, max_] across all dimensions
    scaled_samples = qmc.scale(lhs_samples, [min_] * n_features, [max_] * n_features)
    
    return scaled_samples


def get_all_species(model_spec=None, initial_values: Dict = None) -> List[str]:
    """
    Get list of all species from model specification or initial values.
    
    Args:
        model_spec: ModelSpecification object
        initial_values: Dictionary of initial values
        
    Returns:
        List of species names
    """
    if model_spec is not None:
        return model_spec.A_species + model_spec.B_species
    elif initial_values is not None:
        return list(initial_values.keys())
    else:
        raise ValueError('Either model_spec or initial_values must be provided')


def validate_initial_values(initial_values: Dict[str, float]) -> None:
    """
    Validate that initial values dictionary is properly formatted.
    
    Args:
        initial_values: Dictionary to validate
        
    Raises:
        ValueError: If initial values are invalid
    """
    if not initial_values:
        raise ValueError('Initial values dictionary cannot be empty')
    
    for species, value in initial_values.items():
        if not isinstance(species, str):
            raise ValueError('Species names must be strings')
        
        if not isinstance(value, (int, float, np.number)):
            raise ValueError(f'Value for {species} must be numeric')
        
        if value < 0:
            raise ValueError(f'Value for {species} must be non-negative')


def set_random_seed(seed: int = None) -> np.random.Generator:
    """
    Set random seed and return random number generator.
    
    Args:
        seed: Random seed (None for non-reproducible)
        
    Returns:
        Random number generator
    """
    if seed is not None:
        return np.random.default_rng(seed)
    else:
        return np.random.default_rng()


def generate_perturbation_samples(perturbation_type: str,
                                initial_values: Dict[str, float],
                                perturbation_params: Dict[str, float],
                                n_samples: int,
                                seed: int = None) -> List[Dict[str, float]]:
    """
    Generate multiple perturbation samples.
    
    Args:
        perturbation_type: Type of perturbation ('uniform', 'gaussian', 'lhs', 'lognormal')
        initial_values: Dictionary of initial values
        perturbation_params: Parameters for perturbation
        n_samples: Number of samples to generate
        seed: Random seed
        
    Returns:
        List of dictionaries with perturbed values
    """
    rng = set_random_seed(seed)
    all_perturbed_values = []
    
    if perturbation_type == 'lhs':
        # LHS handles multiple samples differently
        n_features = len(initial_values)
        species_list = list(initial_values.keys())
        lhs_samples = generate_lhs_perturbation(n_samples, n_features,
                                              perturbation_params['min'], 
                                              perturbation_params['max'], seed)
        
        for i in range(n_samples):
            perturbed_values = {}
            for j, species in enumerate(species_list):
                perturbed_values[species] = lhs_samples[i, j]
            all_perturbed_values.append(perturbed_values)
    
    else:
        # Uniform, Gaussian, and Lognormal generate samples individually
        for _ in range(n_samples):
            if perturbation_type == 'uniform':
                perturbed_values = apply_uniform_perturbation(
                    initial_values, perturbation_params['min'], 
                    perturbation_params['max'], rng)
            elif perturbation_type == 'gaussian':
                perturbed_values = apply_gaussian_perturbation(
                    initial_values, perturbation_params, rng)
            elif perturbation_type == 'lognormal':
                perturbed_values = apply_lognormal_perturbation(
                    initial_values, perturbation_params, rng)
            else:
                raise ValueError(f'Unsupported perturbation type: {perturbation_type}')
            
            all_perturbed_values.append(perturbed_values)
    
    return all_perturbed_values


def convert_perturbations_to_dataframe(perturbations: List[Dict[str, float]],
                                     columns: List[str] = None) -> pd.DataFrame:
    """
    Convert list of perturbation dictionaries to DataFrame.
    
    Args:
        perturbations: List of dictionaries with perturbed values
        columns: Column names (optional, uses keys from first dict if None)
        
    Returns:
        DataFrame with perturbation data
    """
    if not perturbations:
        return pd.DataFrame()
    
    if columns is None:
        columns = list(perturbations[0].keys())
    
    # Ensure all dictionaries have the same keys
    for i, pert_dict in enumerate(perturbations):
        for col in columns:
            if col not in pert_dict:
                perturbations[i][col] = np.nan
    
    return pd.DataFrame(perturbations, columns=columns)


def generate_gaussian_perturbation_dataframe(
    initial_values: Dict[str, float],
    perturbation_params: Dict[str, float],
    n_samples: int,
    seed: int = None
) -> pd.DataFrame:
    """
    Generate DataFrame of Gaussian perturbed samples.
    
    Args:
        initial_values: Dictionary of initial values (keys = parameter names, values = starting points)
        perturbation_params: Parameters for Gaussian perturbation
            - 'std': absolute standard deviation
            - 'rsd': relative standard deviation (std/mean)
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns as keys from initial_values and rows as samples
        
    Raises:
        ValueError: If perturbation parameters are invalid
    """
    # Validate perturbation parameters
    if 'std' not in perturbation_params and 'rsd' not in perturbation_params:
        raise ValueError('For Gaussian perturbation, parameters must contain "std" or "rsd"')
    
    if 'std' in perturbation_params and perturbation_params['std'] < 0:
        raise ValueError('Standard deviation must be non-negative')
    if 'rsd' in perturbation_params and perturbation_params['rsd'] < 0:
        raise ValueError('Relative standard deviation must be non-negative')
    
    # Generate perturbation samples
    perturbations = generate_perturbation_samples(
        perturbation_type='gaussian',
        initial_values=initial_values,
        perturbation_params=perturbation_params,
        n_samples=n_samples,
        seed=seed
    )
    
    # Convert to DataFrame
    return convert_perturbations_to_dataframe(perturbations)


def generate_lognormal_perturbation_dataframe(
    initial_values: Dict[str, float],
    perturbation_params: Dict[str, float],
    n_samples: int,
    seed: int = None
) -> pd.DataFrame:
    """
    Generate DataFrame of lognormal perturbed samples.
    
    Args:
        initial_values: Dictionary of initial values (keys = parameter names, values = starting points)
            Must be positive for lognormal distribution
        perturbation_params: Parameters for lognormal perturbation
            - 'shape': shape parameter (sigma) of the underlying normal distribution
            - 'rsd_shape': relative shape parameter (shape / log(initial_value))
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns as keys from initial_values and rows as samples
        
    Raises:
        ValueError: If initial values are not positive or parameters are invalid
    """
    # Validate initial values are positive for lognormal
    for species, value in initial_values.items():
        if value <= 0:
            raise ValueError(f'Initial value for {species} must be positive for lognormal distribution')
    
    # Validate perturbation parameters
    if 'shape' not in perturbation_params and 'rsd_shape' not in perturbation_params:
        raise ValueError('For lognormal perturbation, parameters must contain "shape" or "rsd_shape"')
    
    if 'shape' in perturbation_params and perturbation_params['shape'] <= 0:
        raise ValueError('Shape parameter must be positive')
    if 'rsd_shape' in perturbation_params and perturbation_params['rsd_shape'] <= 0:
        raise ValueError('Relative shape parameter must be positive')
    
    # Generate perturbation samples
    perturbations = generate_perturbation_samples(
        perturbation_type='lognormal',
        initial_values=initial_values,
        perturbation_params=perturbation_params,
        n_samples=n_samples,
        seed=seed
    )
    
    # Convert to DataFrame
    return convert_perturbations_to_dataframe(perturbations)


def generate_uniform_perturbation_dataframe(
    initial_values: Dict[str, float],
    perturbation_params: Dict[str, float],
    n_samples: int,
    seed: int = None
) -> pd.DataFrame:
    """
    Generate DataFrame of uniform perturbed samples.
    
    Args:
        initial_values: Dictionary of initial values (keys = parameter names, values = starting points)
        perturbation_params: Parameters for uniform perturbation
            - 'min': minimum multiplier
            - 'max': maximum multiplier
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns as keys from initial_values and rows as samples
        
    Raises:
        ValueError: If perturbation parameters are invalid
    """
    # Validate perturbation parameters
    if 'min' not in perturbation_params or 'max' not in perturbation_params:
        raise ValueError('For uniform perturbation, parameters must contain "min" and "max"')
    
    if perturbation_params['min'] > perturbation_params['max']:
        raise ValueError('Minimum must be less than or equal to maximum')
    
    # Generate perturbation samples
    perturbations = generate_perturbation_samples(
        perturbation_type='uniform',
        initial_values=initial_values,
        perturbation_params=perturbation_params,
        n_samples=n_samples,
        seed=seed
    )
    
    # Convert to DataFrame
    return convert_perturbations_to_dataframe(perturbations)


def generate_lhs_perturbation_dataframe(
    initial_values: Dict[str, float],
    perturbation_params: Dict[str, float],
    n_samples: int,
    seed: int = None
) -> pd.DataFrame:
    """
    Generate DataFrame of Latin Hypercube Sampling perturbed samples.
    
    Args:
        initial_values: Dictionary of initial values (keys = parameter names, values = starting points)
        perturbation_params: Parameters for LHS perturbation
            - 'min': minimum value for scaling
            - 'max': maximum value for scaling
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with columns as keys from initial_values and rows as samples
        
    Raises:
        ValueError: If perturbation parameters are invalid
    """
    # Validate perturbation parameters
    if 'min' not in perturbation_params or 'max' not in perturbation_params:
        raise ValueError('For LHS perturbation, parameters must contain "min" and "max"')
    
    if perturbation_params['min'] > perturbation_params['max']:
        raise ValueError('Minimum must be less than or equal to maximum')
    
    # Generate perturbation samples
    perturbations = generate_perturbation_samples(
        perturbation_type='lhs',
        initial_values=initial_values,
        perturbation_params=perturbation_params,
        n_samples=n_samples,
        seed=seed
    )
    
    # Convert to DataFrame
    return convert_perturbations_to_dataframe(perturbations)
