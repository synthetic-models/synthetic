"""
Validation utilities for synthetic data generation functions.

Contains reusable validation logic extracted from SyntheticGen.py to eliminate
code duplication and improve maintainability.
"""

from typing import Dict, Any


def validate_simulation_params(simulation_params: Dict[str, Any]) -> None:
    """
    Validate simulation parameters for consistency across different functions.
    
    Args:
        simulation_params: Dictionary containing 'start', 'end', and 'points' keys
        
    Raises:
        ValueError: If required parameters are missing
    """
    required_keys = ['start', 'end', 'points']
    missing_keys = [key for key in required_keys if key not in simulation_params]
    
    if missing_keys:
        raise ValueError(f'Simulation parameters must contain {missing_keys} keys')


def validate_perturbation_params(perturbation_type: str, perturbation_params: Dict[str, Any]) -> None:
    """
    Validate perturbation parameters based on the perturbation type.
    
    Args:
        perturbation_type: Type of perturbation ('uniform', 'gaussian', 'lhs')
        perturbation_params: Parameters for the perturbation
        
    Raises:
        ValueError: If parameters don't match the perturbation type
    """
    if perturbation_type not in ['uniform', 'gaussian', 'lhs']:
        raise ValueError('Perturbation type must be "uniform", "gaussian", or "lhs"')

    if perturbation_type in ['uniform', 'lhs']:
        required_keys = ['min', 'max']
        missing_keys = [key for key in required_keys if key not in perturbation_params]
        if missing_keys:
            raise ValueError(f'For {perturbation_type} perturbation, parameters must contain {missing_keys}')

    elif perturbation_type == 'gaussian':
        required_keys = ['std', 'rsd']
        if not any(key in perturbation_params for key in required_keys):
            raise ValueError('For gaussian perturbation, parameters must contain "std" or "rsd"')


def validate_perturbation_type(perturbation_type: str, valid_types: list = None) -> None:
    """
    Validate that a perturbation type is supported.
    
    Args:
        perturbation_type: Type of perturbation to validate
        valid_types: List of valid perturbation types (default: ['uniform', 'gaussian', 'lhs'])
        
    Raises:
        ValueError: If perturbation type is not supported
    """
    if valid_types is None:
        valid_types = ['uniform', 'gaussian', 'lhs']
    
    if perturbation_type not in valid_types:
        raise ValueError(f'Perturbation type must be one of {valid_types}')


def validate_model_spec_has_species(model_spec) -> None:
    """
    Validate that model specification has required species lists.
    
    Args:
        model_spec: ModelSpecification object
        
    Raises:
        ValueError: If model spec lacks required species attributes
    """
    required_attrs = ['A_species', 'B_species', 'C_species']
    missing_attrs = [attr for attr in required_attrs if not hasattr(model_spec, attr)]
    
    if missing_attrs:
        raise ValueError(f'Model specification missing required attributes: {missing_attrs}')


def validate_feature_dataframe_shape(feature_df, expected_rows: int = None) -> None:
    """
    Validate that a feature dataframe has the expected shape.
    
    Args:
        feature_df: DataFrame to validate
        expected_rows: Expected number of rows (optional)
        
    Raises:
        ValueError: If dataframe is empty or has unexpected shape
    """
    if feature_df.empty:
        raise ValueError('Feature dataframe is empty')
    
    if expected_rows is not None and len(feature_df) != expected_rows:
        raise ValueError(f'Expected {expected_rows} rows, got {len(feature_df)}')


def validate_seed_parameter(seed) -> None:
    """
    Validate that seed parameter is appropriate.
    
    Args:
        seed: Random seed value to validate
        
    Raises:
        ValueError: If seed is not None or an integer
    """
    if seed is not None and not isinstance(seed, int):
        raise ValueError('Random seed must be None or an integer')
