# SyntheticGenUtils - Phase 1 Refactoring

## Overview

This package contains utility functions extracted from `SyntheticGen.py` to eliminate code duplication while maintaining full backward compatibility. The refactoring identified repeated patterns across multiple function versions and extracted them into reusable utilities.

## Structure

```
SyntheticGenUtils/
├── __init__.py          # Package exports and version info
├── ValidationUtils.py    # Parameter validation functions
├── ParallelUtils.py      # Parallel processing utilities
├── DataProcessingUtils.py # DataFrame and data processing functions
├── SimulationUtils.py    # Simulation workflow utilities
├── PerturbationUtils.py  # Perturbation generation functions
└── README.md            # Documentation
```

## Key Features

### 1. Elimination of Code Duplication
- **Validation logic**: Common parameter validation now centralized
- **Parallel processing**: Consistent parallel/sequential simulation handling
- **Data processing**: Unified DataFrame creation and manipulation
- **Simulation workflow**: Reusable solver configuration and result extraction
- **Perturbation generation**: Generic perturbation pattern implementations

### 2. Full Backward Compatibility
- **No existing functions modified**: All original functions remain unchanged
- **New unified API**: Optional unified functions with version selection
- **Gradual adoption**: Can use utilities or continue with existing patterns

### 3. Enhanced Maintainability
- **Single responsibility**: Each utility focuses on specific functionality
- **Error handling**: Consistent error handling across all functions
- **Type hints**: Better code clarity and IDE support

## Utility Functions

### ValidationUtils.py
- `validate_simulation_params()` - Validate simulation parameters
- `validate_perturbation_params()` - Validate perturbation parameters
- `validate_perturbation_type()` - Validate perturbation type support
- `validate_model_spec_has_species()` - Validate model specification
- `validate_feature_dataframe_shape()` - Validate DataFrame shape
- `validate_seed_parameter()` - Validate random seed parameter

### ParallelUtils.py
- `run_parallel_simulation()` - Run simulations in parallel
- `run_sequential_simulation()` - Run simulations sequentially
- `handle_simulation_error()` - Consistent error handling
- `run_parallel_with_error_handling()` - Parallel processing with error handling
- `split_parallel_results()` - Split tuple results from parallel processing

### DataProcessingUtils.py
- `create_feature_dataframe()` - Create DataFrame from feature data
- `create_target_dataframe()` - Create DataFrame from target data
- `process_time_course_data()` - Process time course data
- `extract_simulation_output()` - Extract simulation results
- `normalize_dynamic_features()` - Normalize dynamic features

### SimulationUtils.py
- `compile_solver()` - Configure solver based on type
- `set_simulation_values()` - Set state and parameter values
- `extract_simulation_results()` - Extract results based on capture options
- `create_simulation_function()` - Create reusable simulation function
- `validate_solver_type()` - Validate solver compatibility

### PerturbationUtils.py
- `generate_uniform_perturbation()` - Uniform perturbation generation
- `generate_gaussian_perturbation()` - Gaussian perturbation generation
- `generate_lhs_perturbation()` - LHS perturbation generation
- `get_all_species()` - Get species list from model spec or initial values
- `validate_initial_values()` - Validate initial values dictionary
- `generate_perturbation_samples()` - Generate multiple perturbation samples

## Unified Functions

New unified functions in `SyntheticGen.py` provide a cleaner API with version selection:

### `unified_generate_feature_data(version_number='v3', **kwargs)`
- Select different feature generation versions: 'v1', 'v2', 'v3'

### `unified_generate_target_data(version_number='default', **kwargs)`
- Select different target generation versions: 'default', 'diff_spec', 'diff_build'

### `unified_generate_model_timecourse_data(version_number='default', **kwargs)`
- Select different time course versions: 'default', 'diff_spec', 'diff_build', 'v3', 'diff_build_v3'

## Usage Examples

### Pattern 1: Direct Utility Imports
```python
from models.SyntheticGenUtils.ValidationUtils import validate_simulation_params
from models.SyntheticGenUtils.PerturbationUtils import generate_uniform_perturbation

# Use utilities directly
validate_simulation_params({'start': 0, 'end': 500, 'points': 100})
```

### Pattern 2: Package-level Imports
```python
from models.SyntheticGenUtils import validate_simulation_params, generate_uniform_perturbation
```

### Pattern 3: Unified Functions
```python
from models.SyntheticGen import unified_generate_feature_data

# Use unified API
feature_df = unified_generate_feature_data(
    version_number='v3',
    model_spec=model_spec,
    initial_values=initial_values,
    perturbation_type='uniform',
    perturbation_params={'min': 0.5, 'max': 2.0},
    n=100
)
```

### Pattern 4: Backward Compatibility (No Changes)
```python
# Existing code continues to work unchanged
from models.SyntheticGen import generate_feature_data_v3, generate_target_data

feature_df = generate_feature_data_v3(...)
target_df, time_course = generate_target_data(...)
```

## Future Development

### Phase 2 Possibilities
1. **Refactor existing functions**: Gradually replace repeated code with utility calls
2. **Add new functionality**: Build new features using the utility foundation
3. **Performance optimization**: Optimize utilities while maintaining interfaces
4. **Extended validation**: More comprehensive parameter validation

### Guidelines for New Development
- Use existing utilities when implementing new functions
- Add to utilities when identifying new repeated patterns
- Maintain backward compatibility when extending APIs
- Follow the established naming and parameter conventions

## Testing

Run the demo script to verify functionality:
```bash
python test_utils_demo.py
```

## Benefits

1. **Reduced code duplication**: Common patterns now centralized
2. **Easier maintenance**: Changes to shared logic affect all functions
3. **Better testing**: Utilities can be tested independently
4. **Gradual migration**: No breaking changes required
5. **Future-proof**: Solid foundation for additional features

## Notes

- This is Phase 1 (non-invasive refactoring)
- Existing downstream processes remain unaffected
- New development can choose the appropriate pattern
- Utilities are designed for gradual adoption
