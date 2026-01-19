"""
Feature data generation utilities using new PerturbationUtils functions.

This module provides functions for generating feature/perturbation data
using the enhanced perturbation functions from PerturbationUtils.
"""

import warnings
import logging
from typing import Dict, Any, Optional, Union, List
import pandas as pd
import numpy as np

from ..SyntheticGenUtils.PerturbationUtils import (
    generate_gaussian_perturbation_dataframe,
    generate_lognormal_perturbation_dataframe,
    generate_uniform_perturbation_dataframe,
    generate_lhs_perturbation_dataframe,
    generate_perturbation_samples,
    convert_perturbations_to_dataframe,
    validate_initial_values
)
from .conserve_rules_utils import resolve_species_range

logger = logging.getLogger(__name__)


def make_feature_data(
    initial_values: Dict[str, float],
    perturbation_type: str,
    perturbation_params: Dict[str, Any],
    n_samples: int,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate feature data using specified perturbation type.
    
    This function provides a unified interface for generating feature data
    using the enhanced perturbation functions from PerturbationUtils.
    
    Args:
        initial_values: Dictionary mapping parameter names to initial values
        perturbation_type: Type of perturbation ('uniform', 'gaussian', 'lognormal', 'lhs', 'conserve_rules')
        perturbation_params: Parameters for the perturbation distribution
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with shape (n_samples, n_parameters)
        
    Raises:
        ValueError: If parameters are invalid or unsupported perturbation type
        
    Examples:
        >>> initial_values = {'A': 10.0, 'B': 20.0}
        >>> params = {'std': 2.0}
        >>> df = make_feature_data(initial_values, 'gaussian', params, n_samples=100, seed=42)
    """
    # Validate initial values
    validate_initial_values(initial_values)
    
    # Validate perturbation parameters
    validate_feature_data_params(perturbation_type, perturbation_params)
    
    # Map perturbation type to appropriate generator
    generator_map = {
        'uniform': generate_uniform_perturbation_dataframe,
        'gaussian': generate_gaussian_perturbation_dataframe,
        'lognormal': generate_lognormal_perturbation_dataframe,
        'lhs': generate_lhs_perturbation_dataframe,
        'conserve_rules': make_feature_data_conserve_rules
    }
    
    if perturbation_type not in generator_map:
        raise ValueError(
            f"Unsupported perturbation type: {perturbation_type}. "
            f"Supported types: {list(generator_map.keys())}"
        )
    
    # Get the appropriate generator function
    generator = generator_map[perturbation_type]
    
    # Generate the dataframe
    if perturbation_type == 'conserve_rules':
        # Special handling for conserve_rules to pass model_spec if available
        model_spec = perturbation_params.pop('model_spec', None)
        df = generator(
            initial_values=initial_values,
            perturbation_params=perturbation_params,
            n_samples=n_samples,
            seed=seed,
            model_spec=model_spec
        )
        # Restore model_spec if it was removed
        if model_spec is not None:
            perturbation_params['model_spec'] = model_spec
    else:
        df = generator(
            initial_values=initial_values,
            perturbation_params=perturbation_params,
            n_samples=n_samples,
            seed=seed
        )
    
    return df


def make_feature_data_uniform(
    initial_values: Dict[str, float],
    min_val: float,
    max_val: float,
    n_samples: int,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate feature data using uniform perturbation.
    
    Convenience function for uniform perturbation generation.
    
    Args:
        initial_values: Dictionary mapping parameter names to initial values
        min_val: Minimum multiplier for uniform distribution
        max_val: Maximum multiplier for uniform distribution
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with shape (n_samples, n_parameters)
    """
    return make_feature_data(
        initial_values=initial_values,
        perturbation_type='uniform',
        perturbation_params={'min': min_val, 'max': max_val},
        n_samples=n_samples,
        seed=seed
    )


def make_feature_data_gaussian(
    initial_values: Dict[str, float],
    n_samples: int,
    std: Optional[float] = None,
    rsd: Optional[float] = None,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate feature data using Gaussian perturbation.
    
    Convenience function for Gaussian perturbation generation.
    
    Args:
        initial_values: Dictionary mapping parameter names to initial values
        std: Absolute standard deviation (use either std or rsd)
        rsd: Relative standard deviation (std = rsd * mean)
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with shape (n_samples, n_parameters)
        
    Raises:
        ValueError: If neither std nor rsd is provided
    """
    if std is not None:
        params = {'std': std}
    elif rsd is not None:
        params = {'rsd': rsd}
    else:
        raise ValueError("Either 'std' or 'rsd' must be provided")
    
    return make_feature_data(
        initial_values=initial_values,
        perturbation_type='gaussian',
        perturbation_params=params,
        n_samples=n_samples,
        seed=seed
    )


def make_feature_data_lognormal(
    initial_values: Dict[str, float],
    n_samples: int,
    shape: Optional[float] = None,
    rsd_shape: Optional[float] = None,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate feature data using lognormal perturbation.
    
    Convenience function for lognormal perturbation generation.
    
    Args:
        initial_values: Dictionary mapping parameter names to initial values
            (must be positive for lognormal distribution)
        shape: Shape parameter (use either shape or rsd_shape)
        rsd_shape: Relative shape parameter
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with shape (n_samples, n_parameters)
        
    Raises:
        ValueError: If neither shape nor rsd_shape is provided
    """
    if shape is not None:
        params = {'shape': shape}
    elif rsd_shape is not None:
        params = {'rsd_shape': rsd_shape}
    else:
        raise ValueError("Either 'shape' or 'rsd_shape' must be provided")
    
    return make_feature_data(
        initial_values=initial_values,
        perturbation_type='lognormal',
        perturbation_params=params,
        n_samples=n_samples,
        seed=seed
    )


def make_feature_data_lhs(
    initial_values: Dict[str, float],
    min_val: float,
    max_val: float,
    n_samples: int,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    Generate feature data using Latin Hypercube Sampling.
    
    Convenience function for LHS perturbation generation.
    
    Args:
        initial_values: Dictionary mapping parameter names to initial values
        min_val: Minimum value for scaling
        max_val: Maximum value for scaling
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with shape (n_samples, n_parameters)
    """
    return make_feature_data(
        initial_values=initial_values,
        perturbation_type='lhs',
        perturbation_params={'min': min_val, 'max': max_val},
        n_samples=n_samples,
        seed=seed
    )


def make_feature_data_conserve_rules(
    initial_values: Dict[str, float],
    perturbation_params: Dict[str, Any],
    n_samples: int,
    seed: Optional[int] = None,
    model_spec=None
) -> pd.DataFrame:
    """
    Generate feature data using conserve_rules perturbation.
    
    This function uses lognormal distribution with species-specific shape parameters
    based on biological conservation principles. Species closer to the central pathway
    or output are more conserved (lower shape), while outer species have higher
    variance (higher shape).
    
    Args:
        initial_values: Dictionary mapping parameter names to initial values
        perturbation_params: Parameters containing 'species_range' dictionary
            - 'species_range': Dict mapping species to their shape parameters
            - Alternatively, can contain 'base_shape', 'max_shape', 'num_cascades'
              for auto-generation of species_range
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        model_spec: Model specification object (optional, for auto-generating species_range)
        
    Returns:
        DataFrame with shape (n_samples, n_parameters)
        
    Raises:
        ValueError: If species_range cannot be determined
        
    Examples:
        >>> # Using pre-generated species_range
        >>> species_range = {'R1_1': 0.01, 'I1_1': 0.01, 'R2_1': 0.5, 'I2_1': 0.5}
        >>> df = make_feature_data_conserve_rules(
        ...     initial_values={'R1_1': 10.0, 'I1_1': 5.0, 'R2_1': 8.0, 'I2_1': 4.0},
        ...     perturbation_params={'species_range': species_range},
        ...     n_samples=100,
        ...     seed=42
        ... )
        
        >>> # Auto-generating species_range from model_spec
        >>> df = make_feature_data_conserve_rules(
        ...     initial_values=initial_values,
        ...     perturbation_params={'base_shape': 0.01, 'max_shape': 0.5, 'num_cascades': 5},
        ...     n_samples=100,
        ...     seed=42,
        ...     model_spec=degree_spec
        ... )
    """
    # Validate initial values
    validate_initial_values(initial_values)
    
    # Extract shape parameters once to avoid redundant calculations
    base_shape = perturbation_params.get('base_shape', 0.01)
    max_shape = perturbation_params.get('max_shape', 0.5)
    middle_shape = (base_shape + max_shape) / 2
    
    # Get or generate species_range
    if 'species_range' in perturbation_params:
        species_range = perturbation_params['species_range']
        logger.info("Using provided species_range dictionary")
    else:
        # Auto-generate species_range from model_spec or initial_values
        num_cascades = perturbation_params.get('num_cascades')
        
        species_range = resolve_species_range(
            model_spec=model_spec,
            initial_values=initial_values,
            base_shape=base_shape,
            max_shape=max_shape,
            num_cascades=num_cascades
        )
        logger.info(f"Auto-generated species_range with {len(species_range)} entries")
    
    # Validate that all initial_values have corresponding shape parameters
    missing_species = set(initial_values.keys()) - set(species_range.keys())
    if missing_species:
        logger.warning(f"Species without shape parameters: {missing_species}")
        for species in missing_species:
            species_range[species] = middle_shape
    
    # Generate perturbations using per-species shape parameters
    rng = np.random.default_rng(seed)
    all_perturbed_values = []
    
    for _ in range(n_samples):
        perturbed_values = {}
        for species, initial_value in initial_values.items():
            # Get shape parameter for this species (use middle_shape as fallback)
            shape = species_range.get(species, middle_shape)
            
            # Generate lognormal sample with species-specific shape
            mean = np.log(initial_value)
            sigma = shape
            perturbed_values[species] = rng.lognormal(mean, sigma)
        
        all_perturbed_values.append(perturbed_values)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_perturbed_values, columns=list(initial_values.keys()))
    
    return df


def validate_feature_data_params(
    perturbation_type: str,
    perturbation_params: Dict[str, Any]
) -> None:
    """
    Validate perturbation parameters.
    
    Args:
        perturbation_type: Type of perturbation
        perturbation_params: Parameters to validate
        
    Raises:
        ValueError: If parameters are invalid
    """
    if perturbation_type == 'uniform':
        if 'min' not in perturbation_params or 'max' not in perturbation_params:
            raise ValueError("Uniform perturbation requires 'min' and 'max' parameters")
        if perturbation_params['min'] > perturbation_params['max']:
            raise ValueError("Minimum must be less than or equal to maximum")
    
    elif perturbation_type == 'gaussian':
        if 'std' not in perturbation_params and 'rsd' not in perturbation_params:
            raise ValueError("Gaussian perturbation requires 'std' or 'rsd' parameter")
        if 'std' in perturbation_params and perturbation_params['std'] < 0:
            raise ValueError("Standard deviation must be non-negative")
    
    elif perturbation_type == 'lognormal':
        if 'shape' not in perturbation_params and 'rsd_shape' not in perturbation_params:
            raise ValueError("Lognormal perturbation requires 'shape' or 'rsd_shape' parameter")
        if 'shape' in perturbation_params and perturbation_params['shape'] <= 0:
            raise ValueError("Shape parameter must be positive")
    
    elif perturbation_type == 'lhs':
        if 'min' not in perturbation_params or 'max' not in perturbation_params:
            raise ValueError("LHS perturbation requires 'min' and 'max' parameters")
    
    elif perturbation_type == 'conserve_rules':
        if 'species_range' not in perturbation_params:
            # Check if auto-generation parameters are provided
            required_auto_params = ['base_shape', 'max_shape']
            has_auto_params = any(p in perturbation_params for p in required_auto_params)
            if not has_auto_params:
                raise ValueError(
                    "Conserve rules perturbation requires either 'species_range' dictionary "
                    "or auto-generation parameters ('base_shape', 'max_shape', optionally 'num_cascades')"
                )
    else:
        raise ValueError(f"Unsupported perturbation type: {perturbation_type}")


# Backward compatibility wrapper (with deprecation warning)
def generate_feature_data_v3(
    model_spec=None,
    initial_values: Dict[str, float] = None,
    perturbation_type: str = None,
    perturbation_params: Dict[str, Any] = None,
    n: int = None,
    seed: Optional[int] = None
) -> pd.DataFrame:
    """
    DEPRECATED: Use make_feature_data instead.
    
    Backward compatibility wrapper for generate_feature_data_v3.
    """
    warnings.warn(
        "generate_feature_data_v3 is deprecated. Use make_feature_data instead.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # For backward compatibility, we need to handle the model_spec parameter
    # even though it's not used in the new implementation
    if initial_values is None:
        raise ValueError("initial_values must be provided")
    
    return make_feature_data(
        initial_values=initial_values,
        perturbation_type=perturbation_type,
        perturbation_params=perturbation_params,
        n_samples=n,
        seed=seed
    )