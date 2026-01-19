"""
Example usage of parameter utility functions.

This script demonstrates the three utility functions created for the task:
1. Parameter mapping and explanation
2. Parameter randomization with controlled ranges
3. Initial condition randomization
"""

import numpy as np
from models.ModelBuilder import ModelBuilder
from models.Reaction import Reaction
from models.ReactionArchtype import ReactionArchtype
from models.ArchtypeCollections import michaelis_menten

# Import the utility functions
from models.utils.parameter_mapper import (
    get_parameter_reaction_map,
    find_parameter_by_role,
    explain_reaction_parameters,
    get_parameters_for_state
)
from models.utils.parameter_randomizer import ParameterRandomizer
from models.utils.initial_condition_randomizer import InitialConditionRandomizer


def create_test_model():
    """Create a simple test model for demonstration."""
    model = ModelBuilder("TestModel")
    
    # Add a simple reaction: R1 -> R1a
    reaction1 = Reaction(
        michaelis_menten,
        ('R1',),
        ('R1a',),
        reactant_values=100.0,
        parameters_values=(10.0, 50.0)  # Vmax=10.0, Km=50.0
    )
    
    # Add another reaction: O -> Oa
    reaction2 = Reaction(
        michaelis_menten,
        ('O',),
        ('Oa',),
        reactant_values=200.0,
        parameters_values=(20.0, 80.0)  # Vmax=20.0, Km=80.0
    )
    
    model.add_reaction(reaction1)
    model.add_reaction(reaction2)
    model.precompile()
    
    return model


def demonstrate_parameter_mapping(model):
    """Demonstrate parameter mapping utilities."""
    print("=" * 60)
    print("PARAMETER MAPPING UTILITIES")
    print("=" * 60)
    
    # 1. Get parameter-reaction map
    print("\n1. Parameter-Reaction Map:")
    param_map = get_parameter_reaction_map(model)
    for param_name, info in param_map.items():
        print(f"  {param_name}:")
        print(f"    Type: {info['parameter_type']}")
        print(f"    Reaction {info['reaction_index']}: {info['reactants']} -> {info['products']}")
    
    # 2. Find parameters by role
    print("\n2. Find Parameters by Role:")
    vmax_params = find_parameter_by_role(model, 'Vmax')
    km_params = find_parameter_by_role(model, 'Km')
    print(f"  Vmax parameters: {vmax_params}")
    print(f"  Km parameters: {km_params}")
    
    # 3. Find parameters affecting specific state
    print("\n3. Parameters Affecting Specific State:")
    state_params = get_parameters_for_state(model, 'O')
    print(f"  Parameters affecting 'O':")
    print(f"    As reactant: {state_params['as_reactant']}")
    print(f"    All parameters: {state_params['all']}")
    
    # 4. Explain reaction parameters
    print("\n4. Reaction Parameter Explanations:")
    for i in range(len(model.reactions)):
        explanation = explain_reaction_parameters(model, i)
        print(f"\n  Reaction {i}:")
        print(f"  {explanation}")
    
    return param_map


def demonstrate_parameter_randomization(model):
    """Demonstrate parameter randomization utilities."""
    print("\n" + "=" * 60)
    print("PARAMETER RANDOMIZATION UTILITIES")
    print("=" * 60)
    
    # Create parameter randomizer
    randomizer = ParameterRandomizer(model)
    
    # Show current parameter values
    print("\n1. Current Parameter Values:")
    current_params = model.get_parameters()
    for param_name, value in current_params.items():
        print(f"  {param_name}: {value}")
    
    # Get parameter statistics
    print("\n2. Parameter Statistics:")
    stats = randomizer.get_parameter_statistics()
    for param_type, stat_info in stats.items():
        print(f"  {param_type}:")
        print(f"    Count: {stat_info['count']}")
        print(f"    Min: {stat_info['min']:.2f}")
        print(f"    Max: {stat_info['max']:.2f}")
        print(f"    Mean: {stat_info['mean']:.2f}")
    
    # Set controlled ranges for Vmax and Km
    print("\n3. Setting Controlled Ranges:")
    randomizer.set_range_for_param_type('Vmax', 5.0, 15.0)  # Control Vmax range
    randomizer.set_range_for_param_type('Km', 25.0, 75.0)   # Control Km range
    print("  Set Vmax range: [5.0, 15.0]")
    print("  Set Km range: [25.0, 75.0]")
    
    # Validate current parameters against new ranges
    print("\n4. Parameter Range Validation:")
    validation = randomizer.validate_parameter_ranges()
    for param_name, is_valid in validation.items():
        status = "✓" if is_valid else "✗"
        print(f"  {status} {param_name}")
    
    # Randomize all parameters
    print("\n5. Randomizing All Parameters:")
    randomized_model = randomizer.randomize_all_parameters(seed=42)
    randomized_params = randomized_model.get_parameters()
    for param_name, value in randomized_params.items():
        param_type = randomizer.get_param_type_from_name(param_name)
        min_val, max_val = randomizer.parameter_ranges.get(param_type, (0.0, 1.0))
        print(f"  {param_name}: {value:.2f} (range: [{min_val:.1f}, {max_val:.1f}])")
    
    # Targeted randomization for specific state
    print("\n6. Targeted Randomization for State 'O':")
    targeted_model = randomizer.randomize_parameters_for_state('O', seed=42)
    targeted_params = targeted_model.get_parameters()
    
    # Get original parameters for comparison
    original_params = model.get_parameters()
    print("  Parameter changes:")
    for param_name in targeted_params:
        orig_val = original_params[param_name]
        targ_val = targeted_params[param_name]
        changed = orig_val != targ_val
        change_indicator = "✓" if changed else " "
        param_type = randomizer.get_param_type_from_name(param_name)
        min_val, max_val = randomizer.parameter_ranges.get(param_type, (0.0, 1.0))
        print(f"  {change_indicator} {param_name}: {orig_val:.2f} -> {targ_val:.2f} (range: [{min_val:.1f}, {max_val:.1f}])")
    
    return randomized_model


def demonstrate_initial_condition_randomization(model):
    """Demonstrate initial condition randomization utilities."""
    print("\n" + "=" * 60)
    print("INITIAL CONDITION RANDOMIZATION UTILITIES")
    print("=" * 60)
    
    # Create initial condition randomizer
    ic_randomizer = InitialConditionRandomizer(model)
    
    # Show current initial conditions
    print("\n1. Current Initial Conditions:")
    current_states = model.get_state_variables()
    for state_name, value in current_states.items():
        print(f"  {state_name}: {value}")
    
    # Get state categories
    print("\n2. State Categories:")
    categories = ic_randomizer.get_state_categories()
    for category, states in categories.items():
        print(f"  {category}: {states}")
    
    # Set different ranges for different state types
    print("\n3. Setting Controlled Ranges:")
    ic_randomizer.set_range_for_state('R1', 50.0, 150.0)   # Control specific state
    ic_randomizer.set_range_for_pattern('*a', 0.0, 20.0)  # Control activated forms
    ic_randomizer.set_range_for_pattern('O*', 100.0, 300.0)  # Control outcomes
    
    print("  Set R1 range: [50.0, 150.0]")
    print("  Set *a (activated forms) range: [0.0, 20.0]")
    print("  Set O* (outcomes) range: [100.0, 300.0]")
    
    # Validate current values against ranges
    print("\n4. Initial Condition Range Validation:")
    validation = ic_randomizer.validate_initial_condition_ranges()
    for state_name, is_valid in validation.items():
        status = "✓" if is_valid else "✗"
        min_val, max_val = ic_randomizer.get_range_for_state(state_name)
        current_val = current_states[state_name]
        print(f"  {status} {state_name}: {current_val:.2f} (range: [{min_val:.1f}, {max_val:.1f}])")
    
    # Randomize all initial conditions
    print("\n5. Randomizing All Initial Conditions:")
    randomized_model = ic_randomizer.randomize_initial_conditions(seed=42)
    randomized_states = randomized_model.get_state_variables()
    
    for state_name, value in randomized_states.items():
        min_val, max_val = ic_randomizer.get_range_for_state(state_name)
        original_val = current_states[state_name]
        changed = original_val != value
        change_indicator = "✓" if changed else " "
        print(f"  {change_indicator} {state_name}: {original_val:.2f} -> {value:.2f} (range: [{min_val:.1f}, {max_val:.1f}])")
    
    # Targeted randomization for specific category
    print("\n6. Targeted Randomization for Activated Forms:")
    targeted_model = ic_randomizer.randomize_subset_initial_conditions('*a', seed=42)
    targeted_states = targeted_model.get_state_variables()
    
    print("  State changes (*a pattern only):")
    for state_name, value in targeted_states.items():
        original_val = current_states[state_name]
        if '*a' in state_name or state_name.endswith('a'):
            min_val, max_val = ic_randomizer.get_range_for_state(state_name)
            changed = original_val != value
            change_indicator = "✓" if changed else " "
            print(f"  {change_indicator} {state_name}: {original_val:.2f} -> {value:.2f} (range: [{min_val:.1f}, {max_val:.1f}])")
    
    return randomized_model


def demonstrate_combined_workflow():
    """Demonstrate complete workflow combining all utilities."""
    print("\n" + "=" * 60)
    print("COMBINED WORKFLOW EXAMPLE")
    print("=" * 60)
    
    # Create a model
    model = create_test_model()
    
    print("\n1. Create model with original parameters and initial conditions:")
    params = model.get_parameters()
    states = model.get_state_variables()
    print("  Parameters:", {k: f"{v:.2f}" for k, v in params.items()})
    print("  States:", {k: f"{v:.2f}" for k, v in states.items()})
    
    # Step 1: Map and understand parameters
    print("\n2. Map and understand parameters:")
    param_map = get_parameter_reaction_map(model)
    print(f"  Found {len(param_map)} parameters")
    
    # Step 2: Randomize parameters with controlled ranges
    print("\n3. Randomize kinetic parameters:")
    param_randomizer = ParameterRandomizer(model)
    param_randomizer.set_range_for_param_type('Vmax', 5.0, 25.0)
    param_randomizer.set_range_for_param_type('Km', 30.0, 120.0)
    
    param_randomized_model = param_randomizer.randomize_all_parameters(seed=123)
    new_params = param_randomized_model.get_parameters()
    print("  New parameters:", {k: f"{v:.2f}" for k, v in new_params.items()})
    
    # Step 3: Randomize initial conditions with controlled ranges
    print("\n4. Randomize initial conditions:")
    ic_randomizer = InitialConditionRandomizer(param_randomized_model)
    ic_randomizer.set_range_for_state('R1', 80.0, 120.0)
    ic_randomizer.set_range_for_state('O', 150.0, 250.0)
    ic_randomizer.set_range_for_pattern('*a', 0.0, 10.0)
    
    final_model = ic_randomizer.randomize_initial_conditions(seed=123)
    final_params = final_model.get_parameters()
    final_states = final_model.get_state_variables()
    
    print("  Final parameters:", {k: f"{v:.2f}" for k, v in final_params.items()})
    print("  Final states:", {k: f"{v:.2f}" for k, v in final_states.items()})
    
    # Step 4: Verify everything is within specified ranges
    print("\n5. Verify all constraints are satisfied:")
    
    # Verify parameter ranges
    param_randomizer = ParameterRandomizer(final_model)
    param_randomizer.parameter_ranges = {
        'vmax': (5.0, 25.0),
        'km': (30.0, 120.0)
    }
    param_validation = param_randomizer.validate_parameter_ranges()
    all_params_valid = all(param_validation.values())
    
    # Verify state ranges
    ic_randomizer = InitialConditionRandomizer(final_model)
    ic_randomizer.set_range_for_state('R1', 80.0, 120.0)
    ic_randomizer.set_range_for_state('O', 150.0, 250.0)
    ic_randomizer.set_range_for_pattern('*a', 0.0, 10.0)
    state_validation = ic_randomizer.validate_initial_condition_ranges()
    all_states_valid = all(state_validation.values())
    
    print(f"  All parameters within ranges: {'✓' if all_params_valid else '✗'}")
    print(f"  All states within ranges: {'✓' if all_states_valid else '✗'}")
    
    if all_params_valid and all_states_valid:
        print("\n  ✅ All constraints satisfied!")
    else:
        print("\n  ❌ Some constraints violated")
    
    return final_model


def main():
    """Main demonstration function."""
    print("UTILITY FUNCTIONS DEMONSTRATION")
    print("=" * 60)
    
    # Create a test model
    model = create_test_model()
    
    # Demonstrate each utility
    demonstrate_parameter_mapping(model)
    demonstrate_parameter_randomization(model)
    demonstrate_initial_condition_randomization(model)
    
    # Demonstrate combined workflow
    demonstrate_combined_workflow()
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("\nSummary of implemented utilities:")
    print("1. Parameter Mapping: Map parameters to reactions and states")
    print("2. Parameter Randomization: Control kinetic parameter ranges")
    print("3. Initial Condition Randomization: Control state variable initialization")


if __name__ == "__main__":
    main()
