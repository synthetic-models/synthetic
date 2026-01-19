"""
Initial condition randomization utilities for ModelBuilder objects.
Provides control over random initialization of state variables.
"""

from typing import Dict, List, Optional, Tuple, Union, Pattern
import re
import numpy as np
from ..ModelBuilder import ModelBuilder


class InitialConditionRandomizer:
    """
    Randomize initial conditions for existing ModelBuilder objects.
    
    Features:
    - Set different ranges for different state variables
    - Pattern-based range setting (e.g., 'R*' for all receptors)
    - Reproducible randomization with seed support
    - Validation of initial condition ranges
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
        self.state_ranges = {}  # Specific state ranges
        self.pattern_ranges = []  # Pattern-based ranges
        self._rng = np.random.default_rng()
        
        # Initialize with default ranges
        self._initialize_default_ranges()
    
    def _initialize_default_ranges(self):
        """Initialize sensible default ranges for common state types."""
        # Default range for all states - allow 0 since many states start at 0
        self.set_range_for_pattern("*", 0.0, 100.0)
        
        # Special ranges for activated forms
        self.set_range_for_pattern("*a", 0.0, 10.0)
    
    def set_range_for_state(
        self, 
        state_name: str, 
        min_val: float, 
        max_val: float
    ) -> None:
        """
        Set range for specific state variable.
        
        Args:
            state_name: Exact state variable name (e.g., 'R1', 'Oa')
            min_val: Minimum initial value
            max_val: Maximum initial value
            
        Raises:
            ValueError: If min_val >= max_val or state doesn't exist
        """
        if min_val >= max_val:
            raise ValueError(f"min_val ({min_val}) must be less than max_val ({max_val})")
        
        # Check if state exists
        if state_name not in self.model.states:
            raise ValueError(f"State variable '{state_name}' not found in model")
        
        self.state_ranges[state_name] = (min_val, max_val)
    
    def set_range_for_pattern(
        self, 
        pattern: str, 
        min_val: float, 
        max_val: float
    ) -> None:
        """
        Set range for state variables matching pattern.
        
        Args:
            pattern: Pattern to match (e.g., 'R*' for all receptors, '*a' for activated forms)
                     Supports '*' wildcard
            min_val: Minimum initial value
            max_val: Maximum initial value
            
        Raises:
            ValueError: If min_val >= max_val
        """
        if min_val >= max_val:
            raise ValueError(f"min_val ({min_val}) must be less than max_val ({max_val})")
        
        # Convert pattern to regex
        regex_pattern = pattern.replace('*', '.*')
        compiled_pattern = re.compile(f'^{regex_pattern}$')
        
        self.pattern_ranges.append({
            'pattern': pattern,
            'regex': compiled_pattern,
            'min_val': min_val,
            'max_val': max_val
        })
    
    def _get_range_for_state(self, state_name: str) -> Tuple[float, float]:
        """
        Get the appropriate range for a state variable.
        
        Priority order:
        1. Exact state name match in state_ranges
        2. Pattern matches (first matching pattern wins)
        3. Default (* pattern) if no other matches
        
        Args:
            state_name: State variable name
            
        Returns:
            (min_val, max_val) tuple
        """
        # Check exact match first
        if state_name in self.state_ranges:
            return self.state_ranges[state_name]
        
        # Check pattern matches
        for pattern_info in self.pattern_ranges:
            if pattern_info['regex'].match(state_name):
                return (pattern_info['min_val'], pattern_info['max_val'])
        
        # Default to first pattern (should be * if initialized)
        if self.pattern_ranges:
            return (self.pattern_ranges[0]['min_val'], self.pattern_ranges[0]['max_val'])
        
        # Fallback default
        return (1.0, 100.0)
    
    def get_range_for_state(self, state_name: str) -> Tuple[float, float]:
        """
        Public method to get range for a state variable.
        
        Args:
            state_name: State variable name
            
        Returns:
            (min_val, max_val) tuple
            
        Raises:
            ValueError: If state doesn't exist
        """
        if state_name not in self.model.states:
            raise ValueError(f"State variable '{state_name}' not found in model")
        
        return self._get_range_for_state(state_name)
    
    def randomize_initial_conditions(self, seed: Optional[int] = None) -> ModelBuilder:
        """
        Generate new random initial conditions.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            New ModelBuilder object with randomized initial conditions
        """
        # Set random seed if provided
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        # Create copy of model
        new_model = self.model.copy()
        
        # Precompile the new model to initialize states
        new_model.precompile()
        
        # Randomize each state variable
        for state_name, current_value in self.model.states.items():
            min_val, max_val = self._get_range_for_state(state_name)
            new_value = self._rng.uniform(min_val, max_val)
            new_model.set_state(state_name, new_value)
        
        return new_model
    
    def randomize_subset_initial_conditions(
        self, 
        state_pattern: str, 
        seed: Optional[int] = None
    ) -> ModelBuilder:
        """
        Randomize only initial conditions matching a pattern.
        
        Args:
            state_pattern: Pattern to match (e.g., 'R*', '*a')
            seed: Random seed for reproducibility
            
        Returns:
            New ModelBuilder object with randomized initial conditions for matching states
        """
        # Set random seed if provided
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        
        # Convert pattern to regex
        regex_pattern = state_pattern.replace('*', '.*')
        compiled_pattern = re.compile(f'^{regex_pattern}$')
        
        # Find matching states
        matching_states = [
            state_name for state_name in self.model.states.keys()
            if compiled_pattern.match(state_name)
        ]
        
        if not matching_states:
            raise ValueError(f"No states found matching pattern '{state_pattern}'")
        
        # Create copy of model
        new_model = self.model.copy()
        
        # Precompile the new model to initialize states
        new_model.precompile()
        
        # Randomize only matching states
        for state_name in matching_states:
            min_val, max_val = self._get_range_for_state(state_name)
            new_value = self._rng.uniform(min_val, max_val)
            new_model.set_state(state_name, new_value)
        
        return new_model
    
    def validate_initial_condition_ranges(self) -> Dict[str, bool]:
        """
        Check if current initial conditions are within configured ranges.
        
        Returns:
            Dictionary mapping state names to validation status
        """
        validation_results = {}
        
        for state_name, value in self.model.states.items():
            min_val, max_val = self._get_range_for_state(state_name)
            is_valid = min_val <= value <= max_val
            validation_results[state_name] = is_valid
            
            if not is_valid:
                print(f"Warning: State {state_name} = {value} "
                      f"outside range [{min_val}, {max_val}]")
        
        return validation_results
    
    def get_initial_condition_statistics(self) -> Dict[str, Dict]:
        """
        Get statistics about current initial condition values.
        
        Returns:
            Dictionary with statistics grouped by pattern
        """
        stats_by_pattern = {}
        
        for state_name, value in self.model.states.items():
            # Determine which pattern this state matches
            matched_pattern = None
            
            # Check exact matches
            if state_name in self.state_ranges:
                matched_pattern = state_name
            
            # Check pattern matches
            if matched_pattern is None:
                for pattern_info in self.pattern_ranges:
                    if pattern_info['regex'].match(state_name):
                        matched_pattern = pattern_info['pattern']
                        break
            
            # Use 'other' for unmatched states
            if matched_pattern is None:
                matched_pattern = 'other'
            
            # Initialize pattern stats if needed
            if matched_pattern not in stats_by_pattern:
                stats_by_pattern[matched_pattern] = {
                    'count': 0,
                    'values': [],
                    'min': float('inf'),
                    'max': float('-inf'),
                    'mean': 0.0,
                    'states': []
                }
            
            # Update stats
            stats = stats_by_pattern[matched_pattern]
            stats['count'] += 1
            stats['values'].append(value)
            stats['min'] = min(stats['min'], value)
            stats['max'] = max(stats['max'], value)
            stats['states'].append(state_name)
        
        # Calculate means and clean up
        for pattern, stats in stats_by_pattern.items():
            if stats['values']:
                stats['mean'] = sum(stats['values']) / len(stats['values'])
                del stats['values']  # Remove raw values
        
        return stats_by_pattern
    
    def get_state_categories(self) -> Dict[str, List[str]]:
        """
        Categorize states based on naming patterns.
        
        Returns:
            Dictionary with lists of states by category:
            {
                'receptors': ['R1', 'R2', 'R3'],
                'activated': ['R1a', 'R2a', 'R3a'],
                'intermediates': ['I1_1', 'I1_2'],
                'outcomes': ['O', 'Oa'],
                'other': ['D']  # drugs
            }
        """
        categories = {
            'receptors': [],
            'activated': [],
            'intermediates': [],
            'outcomes': [],
            'drugs': [],
            'other': []
        }
        
        for state_name in self.model.states.keys():
            # Check for receptor patterns
            if re.match(r'^R\d+$', state_name):
                categories['receptors'].append(state_name)
            elif re.match(r'^R\d+a$', state_name):
                categories['activated'].append(state_name)
            elif re.match(r'^I\d+_\d+$', state_name) or re.match(r'^I\d+_\d+a$', state_name):
                categories['intermediates'].append(state_name)
            elif state_name == 'O' or state_name == 'Oa':
                categories['outcomes'].append(state_name)
            elif state_name.startswith('D'):
                categories['drugs'].append(state_name)
            else:
                categories['other'].append(state_name)
        
        # Remove empty categories
        return {k: v for k, v in categories.items() if v}
    
    def set_category_ranges(
        self, 
        category: str, 
        min_val: float, 
        max_val: float
    ) -> None:
        """
        Set range for an entire category of states.
        
        Args:
            category: State category ('receptors', 'activated', 'intermediates', 
                     'outcomes', 'drugs', 'other')
            min_val: Minimum initial value
            max_val: Maximum initial value
            
        Raises:
            ValueError: If min_val >= max_val or invalid category
        """
        if min_val >= max_val:
            raise ValueError(f"min_val ({min_val}) must be less than max_val ({max_val})")
        
        categories = self.get_state_categories()
        if category not in categories:
            raise ValueError(f"Invalid category '{category}'. "
                           f"Valid categories: {list(categories.keys())}")
        
        # Set range for each state in category
        for state_name in categories[category]:
            self.set_range_for_state(state_name, min_val, max_val)
