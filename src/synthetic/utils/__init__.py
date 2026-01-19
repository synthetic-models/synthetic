"""
Utility modules for ModelBuilder parameter and initial condition control.
"""

from .parameter_mapper import (
    get_parameter_reaction_map,
    find_parameter_by_role,
    explain_reaction_parameters,
    get_parameters_for_state
)

from .parameter_randomizer import ParameterRandomizer
from .initial_condition_randomizer import InitialConditionRandomizer
from .kinetic_tuner import KineticParameterTuner

# Data generation utilities
from .make_feature_data import (
    make_feature_data,
    make_feature_data_uniform,
    make_feature_data_gaussian,
    make_feature_data_lognormal,
    make_feature_data_lhs,
    validate_feature_data_params,
    generate_feature_data_v3  # deprecated, but exported for compatibility
)

from .make_target_data import (
    make_target_data,
    make_target_data_diff_spec,
    make_target_data_diff_build,
    generate_target_data,  # deprecated, but exported for compatibility
    generate_target_data_diff_spec,  # deprecated, but exported for compatibility
    generate_target_data_diff_build  # deprecated, but exported for compatibility
)

from .make_timecourse_data import (
    make_timecourse_data,
    make_timecourse_data_diff_spec,
    make_timecourse_data_diff_build,
    make_timecourse_data_v3,
    make_timecourse_data_diff_build_v3,
    generate_model_timecourse_data,  # deprecated, but exported for compatibility
    generate_model_timecourse_data_diff_spec,  # deprecated, but exported for compatibility
    generate_model_timecourse_data_diff_build,  # deprecated, but exported for compatibility
    generate_model_timecourse_data_v3,  # deprecated, but exported for compatibility
    generate_model_timecourse_data_diff_build_v3  # deprecated, but exported for compatibility
)

from .data_generation_helpers import (
    validate_simulation_params,
    extract_species_from_model_spec,
    create_default_simulation_params,
    prepare_perturbation_values,
    check_parameter_set_compatibility,
    create_feature_target_pipeline,
    make_target_data_with_params,
    make_data,
    make_data_extended,
    generate_batch_alternatives
)

__all__ = [
    # Original utilities
    'get_parameter_reaction_map',
    'find_parameter_by_role',
    'explain_reaction_parameters',
    'get_parameters_for_state',
    'ParameterRandomizer',
    'InitialConditionRandomizer',
    'KineticParameterTuner',
    
    # New data generation utilities - make_* functions
    'make_feature_data',
    'make_feature_data_uniform',
    'make_feature_data_gaussian',
    'make_feature_data_lognormal',
    'make_feature_data_lhs',
    'validate_feature_data_params',
    
    'make_target_data',
    'make_target_data_diff_spec',
    'make_target_data_diff_build',
    
    'make_timecourse_data',
    'make_timecourse_data_diff_spec',
    'make_timecourse_data_diff_build',
    'make_timecourse_data_v3',
    'make_timecourse_data_diff_build_v3',
    
    # Helper functions
    'validate_simulation_params',
    'extract_species_from_model_spec',
    'create_default_simulation_params',
    'prepare_perturbation_values',
    'check_parameter_set_compatibility',
    'create_feature_target_pipeline',
    'make_target_data_with_params',
    'make_data',
    'make_data_extended',
    'generate_batch_alternatives',
    
    # Deprecated functions (for backward compatibility)
    'generate_feature_data_v3',
    'generate_target_data',
    'generate_target_data_diff_spec',
    'generate_target_data_diff_build',
    'generate_model_timecourse_data',
    'generate_model_timecourse_data_diff_spec',
    'generate_model_timecourse_data_diff_build',
    'generate_model_timecourse_data_v3',
    'generate_model_timecourse_data_diff_build_v3'
]
