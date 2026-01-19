"""
Optimisation module for kinetic parameter tuning.

This module provides optimization-based approaches to generate kinetic parameters
that achieve target active fractions in multi-degree drug interaction networks.

Main components:
- ParameterOptimizer: Direct optimization using scipy with pre/post drug targets
- utilities: Helper functions for target generation and error calculation
"""

from .parameter_optimizer import ParameterOptimizer, optimize_parameters

__all__ = [
    'ParameterOptimizer',
    'optimize_parameters',
]
