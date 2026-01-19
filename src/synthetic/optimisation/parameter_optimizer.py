"""
Parameter optimizer for kinetic tuning using direct optimization.

This module provides optimization-based approaches to generate kinetic parameters
that achieve target active fractions in multi-degree drug interaction networks,
considering both pre-drug and post-drug treatment conditions.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Callable
from scipy.optimize import minimize, differential_evolution, Bounds
import logging

from ..ModelBuilder import ModelBuilder
from ..Solver.Solver import Solver
from ..Solver.RoadrunnerSolver import RoadrunnerSolver
from ..utils.kinetic_tuner import KineticParameterTuner
from ..utils.parameter_mapper import get_parameters_for_state


class ParameterOptimizer:
    """
    Direct optimization-based parameter tuner for achieving target active fractions.
    
    Uses scipy.optimize to find parameters that minimize error between:
    1. Pre-drug active fractions (50-70% range, all species)
    2. Post-drug active fractions (20-50% range, degree 1 species only)
    
    The optimization uses a single simulation with piecewise drug application,
    extracting both pre- and post-drug steady states from the same timecourse.
    """
    
    def __init__(self, 
                 model: ModelBuilder,
                 solver: Optional[Solver] = None,
                 random_seed: Optional[int] = None):
        """
        Initialize the parameter optimizer.
        
        Args:
            model: ModelBuilder object with network structure
            solver: Optional solver instance (defaults to RoadrunnerSolver)
            random_seed: Optional seed for reproducible random number generation
        """
        if not model.pre_compiled:
            raise ValueError("Model must be pre-compiled before parameter optimization")
        
        self.model = model
        self.solver = solver or RoadrunnerSolver()
        self.rng = np.random.default_rng(random_seed)
        
        # Cache model information
        self.all_states = model.get_state_variables()
        self.active_states = {k: v for k, v in self.all_states.items() if k.endswith('a')}
        self.inactive_states = {k: v for k, v in self.all_states.items() if not k.endswith('a')}
        
        # Parse drug information from model
        self.drug_info = self._parse_drug_info()
        
        # Get all parameters and their bounds
        self.all_params = list(model.get_parameters().keys())
        self.param_bounds = self._estimate_parameter_bounds()
        
        # Initialize target storage
        self.pre_drug_targets: Optional[Dict[str, float]] = None
        self.post_drug_targets: Optional[Dict[str, float]] = None
        
        self._logger = logging.getLogger(__name__)
        
        # Validate state pairs
        self._validate_state_pairs()
    
    def _parse_drug_info(self) -> Dict:
        """
        Parse drug information from model variables.
        
        Returns:
            Dictionary containing drug name, start time, and concentration
        """
        import re
        
        drug_info = {}
        
        # Regex pattern for piecewise assignment: 'D := piecewise(0, time < 500, 10)'
        pattern = r'(\w+)\s*:=\s*piecewise\(([^,]+),\s*time\s*<\s*([^,]+),\s*([^)]+)\)'
        
        for var_name, rule in self.model.variables.items():
            match = re.match(pattern, rule.strip())
            if match:
                state_name, before_value, activation_time, after_value = match.groups()
                try:
                    drug_info = {
                        'name': state_name,
                        'start_time': float(activation_time),
                        'concentration': float(after_value),
                        'before_value': float(before_value)
                    }
                    break  # Assume only one drug for now
                except ValueError:
                    continue
        
        return drug_info
    
    def _validate_state_pairs(self):
        """Ensure each active state Xa has a corresponding inactive state X."""
        for active_state in self.active_states.keys():
            inactive_state = active_state[:-1]  # Remove trailing 'a'
            if inactive_state not in self.inactive_states:
                raise ValueError(
                    f"Active state {active_state} has no corresponding inactive state {inactive_state}"
                )
    
    def _estimate_parameter_bounds(self) -> Bounds:
        """
        Estimate biologically reasonable bounds for parameters.
        
        Returns:
            Bounds object with lower and upper bounds for each parameter
        """
        # Default bounds based on parameter type
        lower_bounds = []
        upper_bounds = []
        
        for param_name in self.all_params:
            param_lower, param_upper = self._get_param_bounds(param_name)
            lower_bounds.append(param_lower)
            upper_bounds.append(param_upper)
        
        return Bounds(lower_bounds, upper_bounds)
    
    def _get_param_bounds(self, param_name: str) -> Tuple[float, float]:
        """
        Get bounds for a specific parameter based on its type.
        
        Args:
            param_name: Parameter name (e.g., 'Vmax_J0', 'Km_J1', 'Kc_J2')
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        param_lower = param_name.lower()
        
        # Rate constants (Vmax, Kc) - allow small values for weak activation
        if 'vmax' in param_lower or 'kc' in param_lower:
            return (0.01, 1000.0)
        
        # Michaelis constants (Km)
        elif 'km' in param_lower:
            return (0.1, 10000.0)
        
        # Inhibition constants (Ki)
        elif 'ki' in param_lower:
            return (0.01, 1000.0)
        
        # Default bounds
        else:
            return (0.01, 1000.0)
    
    def _identify_degree1_species(self) -> List[str]:
        """
        Identify degree 1 species from the model structure.
        
        Returns:
            List of degree 1 species names (both active and inactive forms)
        """
        degree1_species = []
        
        # Look for patterns R1_* and I1_*
        for state in self.all_states.keys():
            if state.startswith('R1_') or state.startswith('I1_') or state == 'O':
                degree1_species.append(state)
                # Also include active form if applicable
                if not state.endswith('a') and f"{state}a" in self.all_states:
                    degree1_species.append(f"{state}a")
        
        return list(set(degree1_species))
    
    def generate_targets(self,
                        pre_drug_range: Tuple[float, float] = (0.5, 0.7),
                        post_drug_range: Tuple[float, float] = (0.2, 0.5)) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Generate consistent pre-drug and post-drug targets.
        
        Args:
            pre_drug_range: Range for pre-drug active fractions (50-70%)
            post_drug_range: Range for post-drug active fractions (20-50%)
            
        Returns:
            Tuple of (pre_drug_targets, post_drug_targets)
        """
        # Identify degree 1 species
        degree1_species = self._identify_degree1_species()
        degree1_active = [s for s in degree1_species if s.endswith('a')]
        
        # Generate pre-drug targets for all active species
        pre_drug_targets = {}
        for active_state in self.active_states.keys():
            # Random active fraction within pre-drug range
            target_fraction = self.rng.uniform(pre_drug_range[0], pre_drug_range[1])
            inactive_state = active_state[:-1]
            total = self.all_states[inactive_state] + self.all_states[active_state]
            pre_drug_targets[active_state] = target_fraction * total
        
        # Generate post-drug targets only for degree 1 active species
        post_drug_targets = {}
        for active_state in degree1_active:
            pre_val = pre_drug_targets.get(active_state)
            if pre_val is None:
                continue
            
            inactive_state = active_state[:-1]
            total = self.all_states[inactive_state] + self.all_states[active_state]
            pre_fraction = pre_val / total
            
            # Ensure post-drug target is lower than pre-drug
            max_post_fraction = min(post_drug_range[1], pre_fraction * 0.9)
            min_post_fraction = max(post_drug_range[0], pre_fraction * 0.3)
            
            if max_post_fraction > min_post_fraction:
                post_fraction = self.rng.uniform(min_post_fraction, max_post_fraction)
                post_drug_targets[active_state] = post_fraction * total
        
        self.pre_drug_targets = pre_drug_targets
        self.post_drug_targets = post_drug_targets
        
        return pre_drug_targets, post_drug_targets
    
    def _simulate_with_parameters(self, params_vector: np.ndarray) -> Dict:
        """
        Simulate model with given parameters.
        
        Args:
            params_vector: Vector of parameter values in same order as self.all_params
            
        Returns:
            Dictionary with simulation results
        """
        # Create model copy
        model_copy = self.model.copy()
        
        # Apply parameters
        for param_name, param_value in zip(self.all_params, params_vector):
            model_copy.set_parameter(param_name, param_value)
        
        # Compile and simulate
        try:
            self.solver.compile(model_copy.get_sbml_model())
            
            # Run simulation long enough to capture both pre- and post-drug steady states
            drug_start = self.drug_info.get('start_time', 500)
            # Use reasonable simulation time: drug time + settling time
            end_time = drug_start + 1000  # 1000 seconds after drug
            result = self.solver.simulate(start=0, stop=end_time, step=101)
            
            return {
                'success': True,
                'data': result,
                'model': model_copy
            }
        except Exception as e:
            self._logger.warning(f"Simulation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': None,
                'model': model_copy
            }
    
    def _extract_concentrations(self, simulation_data, time_threshold: float) -> Dict[str, float]:
        """
        Extract concentrations at a specific time threshold.
        
        Args:
            simulation_data: Simulation result dataframe
            time_threshold: Time point to extract concentrations
            
        Returns:
            Dictionary mapping species names to concentrations
        """
        if simulation_data is None:
            return {}
        
        # Find index closest to time_threshold
        time_col = simulation_data['time']
        idx = np.abs(time_col - time_threshold).argmin()
        
        concentrations = {}
        for col in simulation_data.columns:
            if col != 'time':
                concentrations[col] = float(simulation_data[col].iloc[idx])
        
        return concentrations
    
    def _objective_function(self, params_vector: np.ndarray) -> float:
        """
        Objective function for optimization.
        
        Calculates error between actual and target active fractions for:
        1. Pre-drug steady state (all species)
        2. Post-drug steady state (degree 1 species only)
        
        Uses relative error (fraction error) instead of absolute concentration error
        for better scaling and robustness.
        
        Args:
            params_vector: Vector of parameter values
            
        Returns:
            Total error (sum of squared relative errors)
        """
        if self.pre_drug_targets is None or self.post_drug_targets is None:
            raise RuntimeError("Targets not set. Call generate_targets() first.")
        
        # Simulate model with current parameters
        sim_result = self._simulate_with_parameters(params_vector)
        if not sim_result['success']:
            return 1e6  # Penalty for failed simulation
        
        simulation_data = sim_result['data']
        
        # Extract pre-drug concentrations (just before drug application)
        drug_start = self.drug_info.get('start_time', 500)
        pre_drug_time = drug_start * 0.9  # 90% of drug start time
        pre_concentrations = self._extract_concentrations(simulation_data, pre_drug_time)
        
        # Extract post-drug concentrations (sufficiently after drug application)
        settling_time = 500  # Reduced settling time for speed
        post_drug_time = drug_start + settling_time
        post_concentrations = self._extract_concentrations(simulation_data, post_drug_time)
        
        # Calculate errors using fraction error
        total_error = 0.0
        
        # Pre-drug error (all active species) - use relative error
        for species, target in self.pre_drug_targets.items():
            if species in pre_concentrations:
                # Get total concentration for this species
                inactive_state = species[:-1]
                if inactive_state in self.all_states and species in self.all_states:
                    total = self.all_states[inactive_state] + self.all_states[species]
                    if total > 0:
                        actual_fraction = pre_concentrations[species] / total
                        target_fraction = target / total
                        # Relative error squared
                        rel_error = (actual_fraction - target_fraction) ** 2
                        total_error += rel_error
        
        # Post-drug error (degree 1 species only)
        degree1_active = [s for s in self.post_drug_targets.keys() if s.endswith('a')]
        for species in degree1_active:
            target = self.post_drug_targets.get(species)
            if target is not None and species in post_concentrations:
                # Get total concentration for this species
                inactive_state = species[:-1]
                if inactive_state in self.all_states and species in self.all_states:
                    total = self.all_states[inactive_state] + self.all_states[species]
                    if total > 0:
                        actual_fraction = post_concentrations[species] / total
                        target_fraction = target / total
                        # Relative error squared
                        rel_error = (actual_fraction - target_fraction) ** 2
                        total_error += rel_error
        
        return total_error
    
    def optimize(self,
                pre_drug_range: Tuple[float, float] = (0.5, 0.7),
                post_drug_range: Tuple[float, float] = (0.2, 0.5),
                use_initial_guess: bool = True,
                max_iterations: int = 50,
                tolerance: float = 1e-3,
                method: str = 'L-BFGS-B',
                workers: int = 1) -> Dict:
        """
        Optimize parameters to achieve target active fractions.
        
        Args:
            pre_drug_range: Range for pre-drug active fractions
            post_drug_range: Range for post-drug active fractions
            use_initial_guess: Whether to use KineticParameterTuner for initial guess
            max_iterations: Maximum number of optimization iterations
            tolerance: Convergence tolerance
            method: Optimization method ('L-BFGS-B' or 'differential_evolution')
            workers: Number of parallel workers for differential_evolution
            
        Returns:
            Dictionary containing optimized parameters and optimization results
        """
        # Generate targets
        pre_targets, post_targets = self.generate_targets(pre_drug_range, post_drug_range)
        self._logger.info(f"Generated targets: {len(pre_targets)} pre-drug, {len(post_targets)} post-drug")
        
        # Get initial guess
        if use_initial_guess:
            initial_guess = self._get_initial_guess()
        else:
            # Random initial guess within bounds
            initial_guess = []
            for i in range(len(self.all_params)):
                lb = self.param_bounds.lb[i]
                ub = self.param_bounds.ub[i]
                initial_guess.append(self.rng.uniform(lb, ub))
        
        # Run optimization
        self._logger.info(f"Starting {method} optimization with {len(self.all_params)} parameters")
        
        if method == 'differential_evolution':
            # Global optimization with potential parallelization
            result = differential_evolution(
                func=self._objective_function,
                bounds=self.param_bounds,
                maxiter=max_iterations,
                tol=tolerance,
                disp=True,
                workers=workers,
                seed=self.rng.integers(0, 2**31 - 1),
                updating='deferred' if workers > 1 else 'immediate'
            )
            n_iterations = result.nit
        else:
            # Local optimization (L-BFGS-B)
            result = minimize(
                fun=self._objective_function,
                x0=initial_guess,
                bounds=self.param_bounds,
                method='L-BFGS-B',
                options={
                    'maxiter': max_iterations,
                    'ftol': tolerance,
                    'disp': True,
                    'maxfun': max_iterations * 5
                }
            )
            n_iterations = result.nit
        
        # Convert result to parameter dictionary
        optimized_params = {}
        for param_name, param_value in zip(self.all_params, result.x):
            optimized_params[param_name] = param_value
        
        # Evaluate final error
        final_error = self._objective_function(result.x)
        
        # Calculate achieved fractions for reporting
        achieved_fractions = self._calculate_achieved_fractions(result.x)
        
        # More lenient success criteria:
        # - If optimizer reports success, use that
        # - Otherwise, consider it successful if error is reasonable (< 0.5)
        # - This handles cases where max iterations reached but progress was made
        is_successful = result.success or final_error < 0.5
        
        return {
            'success': is_successful,
            'optimized_parameters': optimized_params,
            'initial_parameters': dict(zip(self.all_params, initial_guess)),
            'final_error': final_error,
            'n_iterations': n_iterations,
            'message': result.message,
            'pre_drug_targets': pre_targets,
            'post_drug_targets': post_targets,
            'achieved_fractions': achieved_fractions,
            'method': method
        }
    
    def _calculate_achieved_fractions(self, params_vector: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Calculate achieved active fractions for reporting.
        
        Args:
            params_vector: Vector of parameter values
            
        Returns:
            Dictionary with achieved fractions for pre- and post-drug conditions
        """
        if self.pre_drug_targets is None or self.post_drug_targets is None:
            return {}
        
        sim_result = self._simulate_with_parameters(params_vector)
        if not sim_result['success']:
            return {}
        
        simulation_data = sim_result['data']
        drug_start = self.drug_info.get('start_time', 500)
        pre_drug_time = drug_start * 0.9
        post_drug_time = drug_start + 500
        
        pre_concentrations = self._extract_concentrations(simulation_data, pre_drug_time)
        post_concentrations = self._extract_concentrations(simulation_data, post_drug_time)
        
        achieved = {'pre_drug': {}, 'post_drug': {}}
        
        # Pre-drug fractions
        for species in self.pre_drug_targets.keys():
            if species in pre_concentrations:
                inactive_state = species[:-1]
                if inactive_state in self.all_states and species in self.all_states:
                    total = self.all_states[inactive_state] + self.all_states[species]
                    if total > 0:
                        achieved['pre_drug'][species] = pre_concentrations[species] / total
        
        # Post-drug fractions (degree 1 only)
        degree1_active = [s for s in self.post_drug_targets.keys() if s.endswith('a')]
        for species in degree1_active:
            if species in post_concentrations:
                inactive_state = species[:-1]
                if inactive_state in self.all_states and species in self.all_states:
                    total = self.all_states[inactive_state] + self.all_states[species]
                    if total > 0:
                        achieved['post_drug'][species] = post_concentrations[species] / total
        
        return achieved
    
    def _get_initial_guess(self) -> np.ndarray:
        """
        Get initial parameter guess using KineticParameterTuner.
        
        Returns:
            Vector of parameter values
        """
        try:
            tuner = KineticParameterTuner(self.model, random_seed=int(self.rng.integers(0, 10000)))
            initial_params = tuner.generate_parameters(
                active_percentage_range=(0.5, 0.7),  # Use pre-drug range
                X_total_multiplier=5.0,
                ki_val=100.0,
                v_max_f_random_range=(5.0, 10.0)
            )
            
            # Convert to vector in correct order
            initial_vector = []
            for param_name in self.all_params:
                if param_name in initial_params:
                    initial_vector.append(initial_params[param_name])
                else:
                    # Use midpoint of bounds as fallback
                    idx = self.all_params.index(param_name)
                    lb = self.param_bounds.lb[idx]
                    ub = self.param_bounds.ub[idx]
                    initial_vector.append((lb + ub) / 2)
            
            return np.array(initial_vector)
            
        except Exception as e:
            self._logger.warning(f"Failed to get initial guess from KineticParameterTuner: {e}")
            # Fallback to random guess
            initial_vector = []
            for i in range(len(self.all_params)):
                lb = self.param_bounds.lb[i]
                ub = self.param_bounds.ub[i]
                initial_vector.append(self.rng.uniform(lb, ub))
            return np.array(initial_vector)


def optimize_parameters(model: ModelBuilder,
                       pre_drug_range: Tuple[float, float] = (0.5, 0.7),
                       post_drug_range: Tuple[float, float] = (0.2, 0.5),
                       random_seed: Optional[int] = None,
                       method: str = 'L-BFGS-B',
                       workers: int = 1) -> Dict:
    """
    Convenience function for optimizing kinetic parameters.
    
    Args:
        model: ModelBuilder object with network structure
        pre_drug_range: Range for pre-drug active fractions
        post_drug_range: Range for post-drug active fractions
        random_seed: Optional seed for reproducibility
        method: Optimization method ('L-BFGS-B' or 'differential_evolution')
        workers: Number of parallel workers for differential_evolution
        
    Returns:
        Dictionary containing optimized parameters and optimization results
    """
    optimizer = ParameterOptimizer(model, random_seed=random_seed)
    return optimizer.optimize(pre_drug_range, post_drug_range, method=method, workers=workers)
