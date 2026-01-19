"""
Parameter randomization utilities for ModelBuilder objects.
Provides fine-grained control over random parameter generation.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ..ModelBuilder import ModelBuilder
from .parameter_mapper import get_parameter_reaction_map, find_parameter_by_role


class ParameterRandomizer:
    """
    Generate new random parameters for existing ModelBuilder objects.
    
    Features:
    - Set different ranges for different parameter types (Vmax, Km, Kc, etc.)
    - Randomize all parameters or only those affecting specific state variables
    - Reproducible randomization with seed support
    - Validation of parameter ranges
    """
    
    def __init__(self, model_builder: ModelBuilder):
        """
        Initialize randomizer for a ModelBuilder object.
        
        Args:
            model_builder: The ModelBuilder object to randomize
        """
        if not model_builder.pre_compiled:
            raise ValueError("Model must be pre-compiled before randomization")
        
        self.model = model_builder
        self.parameter_ranges = {}
        self._rng = np.random.default_rng()
        self._param_map = get_parameter_reaction_map(model_builder)
        
        # Initialize default ranges based on parameter types
        self._initialize_default_ranges()
    
    def _initialize_default_ranges(self):
        """Initialize sensible default ranges for common parameter types."""
        default_ranges = {
            'vmax': (0.1, 20.0),      # Maximum rates (wider to accommodate test values)
            'km': (1.0, 200.0),       # Michaelis constants (wider to accommodate test values)
            'kc': (0.01, 1.0),        # Constitutive rates
            'ka': (0.001, 0.1),       # Activation constants
            'ki': (0.001, 0.1),       # Inhibition constants
            'kic': (0.001, 0.1),      # Competitive inhibition
            'k': (0.001, 1.0),        # Generic rate constants
        }
        
        for param_type, (min_val, max_val) in default_ranges.items():
            self.parameter_ranges[param_type] = (min_val, max_val)
    
    def set_range_for_param_type(
        self, 
        param_type: str, 
        min_val: float, 
        max_val: float
    ) -> None:
        """
        Set range for specific parameter types.
        
        Args:
            param_type: Parameter type to set range for (e.g., 'Vmax', 'Km', 'Kc')
                        Case-insensitive matching
            min_val: Minimum value for this parameter type
            max_val: Maximum value for this parameter type
            
        Raises:
            ValueError: If min_val >= max_val or values are not positive
        """
        if min_val >= max_val:
            raise ValueError(f"min_val ({min_val}) must be less than max_val ({max_val})")
        if min_val <= 0:
            raise ValueError(f"Parameter values must be positive, got min_val={min_val}")
        
        param_type_lower = param_type.lower()
        self.parameter_ranges[param_type_lower] = (min_val, max_val)
    
    def get_param_type_from_name(self, param_name: str) -> str:
        """
        Extract parameter type from parameter name.
        
        Args:
            param_name: Parameter name (e.g., 'Km_J0', 'Vmax_J1')
            
        Returns:
            Lowercase parameter type (e.g., 'km', 'vmax')
        """
        # Extract base parameter type before underscore
        parts = param_name.split('_')
        if len(parts) < 2:
            return 'unknown'
        
        param_type = parts[0].lower()
        
        # Map to known parameter types
        if 'vmax' in param_type:
            return 'vmax'
        elif 'km' in param_type:
            return 'km'
        elif param_type.startswith('kc'):
            return 'kc'
        elif param_type.startswith('ka'):
            return 'ka'
        elif param_type.startswith('ki'):
            if 'c' in param_type:
                return 'kic'
            return 'ki'
        elif param_type.startswith('k'):
            return 'k'
        else:
            return 'unknown'
    
    def _get_range_for_param(self, param_name: str) -> Tuple[float, float]:
        """
        Get the appropriate range for a parameter.
        
        Args:
            param_name: Parameter name
            
        Returns:
            (min_val, max_val) tuple
        """
        param_type = self.get_param_type_from_name(param_name)
        
        # Try exact match first
        if param_type in self.parameter_ranges:
            return self.parameter_ranges[param_type]
        
        # Try generic 'k' range for any rate constant
        if 'k' in param_type and 'k' in self.parameter_ranges:
            return self.parameter_ranges['k']
        
        # Default to Vmax range if nothing else matches
        return self.parameter_ranges.get('vmax', (0.1, 10.0))
    
    def randomize_all_parameters(self, seed: Optional[int] = None) -> ModelBuilder:
        """
        Generate new random parameters for entire model.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            New ModelBuilder object with randomized parameters
        """
        # Set random seed if provided
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        # Create copy of model
        new_model = self.model.copy()
        
        # Precompile the new model to initialize states and parameters
        new_model.precompile()
        
        # Get all parameters from the original model
        all_params = self.model.get_parameters()
        
        # Randomize each parameter
        for param_name, current_value in all_params.items():
            min_val, max_val = self._get_range_for_param(param_name)
            new_value = self._rng.uniform(min_val, max_val)
            new_model.set_parameter(param_name, new_value)
        
        return new_model
    
    def randomize_parameters_for_state(
        self, 
        state_var: str, 
        seed: Optional[int] = None
    ) -> ModelBuilder:
        """
        Randomize only parameters affecting a specific state variable.
        
        Args:
            state_var: State variable name (e.g., 'O', 'R1')
            seed: Random seed for reproducibility
            
        Returns:
            New ModelBuilder object with randomized parameters for specified state
        """
        # Set random seed if provided
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        # Find parameters affecting this state
        from .parameter_mapper import get_parameters_for_state
        state_params = get_parameters_for_state(self.model, state_var)
        params_to_randomize = state_params['all']
        
        if not params_to_randomize:
            raise ValueError(f"No parameters found affecting state variable '{state_var}'")
        
        # Create copy of model
        new_model = self.model.copy()
        
        # Precompile the new model to initialize states and parameters
        new_model.precompile()
        
        # Randomize only the specified parameters
        for param_name in params_to_randomize:
            min_val, max_val = self._get_range_for_param(param_name)
            new_value = self._rng.uniform(min_val, max_val)
            new_model.set_parameter(param_name, new_value)
        
        return new_model
    
    def randomize_parameters_by_role(
        self, 
        role: str, 
        seed: Optional[int] = None
    ) -> ModelBuilder:
        """
        Randomize only parameters matching a specific role.
        
        Args:
            role: Parameter role to randomize (e.g., 'Vmax', 'Km')
            seed: Random seed for reproducibility
            
        Returns:
            New ModelBuilder object with randomized parameters of specified role
        """
        # Set random seed if provided
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        # Find parameters with this role
        params_to_randomize = find_parameter_by_role(self.model, role)
        
        if not params_to_randomize:
            raise ValueError(f"No parameters found with role '{role}'")
        
        # Create copy of model
        new_model = self.model.copy()
        
        # Precompile the new model to initialize states and parameters
        new_model.precompile()
        
        # Randomize only the specified parameters
        for param_name in params_to_randomize:
            min_val, max_val = self._get_range_for_param(param_name)
            new_value = self._rng.uniform(min_val, max_val)
            new_model.set_parameter(param_name, new_value)
        
        return new_model
    
    def validate_parameter_ranges(self) -> Dict[str, bool]:
        """
        Check if current parameter values are within configured ranges.
        
        Returns:
            Dictionary mapping parameter names to validation status
        """
        all_params = self.model.get_parameters()
        validation_results = {}
        
        for param_name, value in all_params.items():
            min_val, max_val = self._get_range_for_param(param_name)
            is_valid = min_val <= value <= max_val
            validation_results[param_name] = is_valid
            
            if not is_valid:
                print(f"Warning: Parameter {param_name} = {value} "
                      f"outside range [{min_val}, {max_val}]")
        
        return validation_results
    
    def get_parameter_statistics(self) -> Dict[str, Dict]:
        """
        Get statistics about current parameter values.
        
        Returns:
            Dictionary with statistics by parameter type
        """
        all_params = self.model.get_parameters()
        
        stats_by_type = {}
        
        for param_name, value in all_params.items():
            param_type = self.get_param_type_from_name(param_name)
            
            if param_type not in stats_by_type:
                stats_by_type[param_type] = {
                    'count': 0,
                    'values': [],
                    'min': float('inf'),
                    'max': float('-inf'),
                    'mean': 0.0
                }
            
            stats = stats_by_type[param_type]
            stats['count'] += 1
            stats['values'].append(value)
            stats['min'] = min(stats['min'], value)
            stats['max'] = max(stats['max'], value)
        
        # Calculate means
        for param_type, stats in stats_by_type.items():
            if stats['values']:
                stats['mean'] = sum(stats['values']) / len(stats['values'])
                del stats['values']  # Remove raw values to keep output clean
        
        return stats_by_type
