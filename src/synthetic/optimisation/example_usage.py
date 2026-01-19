"""
Example usage of ParameterOptimizer for multi-degree drug interaction networks.

Demonstrates how to use the optimization-based approach to achieve target
active fractions for both pre-drug and post-drug conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
from models.Specs.DegreeInteractionSpec import DegreeInteractionSpec
from models.Specs.Drug import Drug
from models.optimisation import ParameterOptimizer, optimize_parameters
from models.Solver.RoadrunnerSolver import RoadrunnerSolver


def create_multi_degree_model():
    """
    Create a multi-degree drug interaction model for demonstration.
    
    Returns:
        ModelBuilder object with multi-degree network
    """
    # Create network with 2 degrees for simplicity
    degree_spec = DegreeInteractionSpec(degree_cascades=[1, 2])
    
    # Generate specifications with moderate feedback
    degree_spec.generate_specifications(
        random_seed=42,
        feedback_density=0.5
    )
    
    # Add drug targeting degree 1 R species
    drug = Drug(
        name="D",
        start_time=500.0,      # Drug applied at time 500
        default_value=10.0,    # Drug concentration
        regulation=["R1_1"],   # Targets R1_1
        regulation_type=["down"]  # Down-regulation
    )
    degree_spec.add_drug(drug)
    
    # Generate the model
    model = degree_spec.generate_network(
        network_name="DemoMultiDegree",
        mean_range_species=(50, 150),
        rangeScale_params=(0.8, 1.2),
        rangeMultiplier_params=(0.9, 1.1),
        random_seed=42,
        receptor_basal_activation=True
    )
    
    return model


def demonstrate_optimization():
    """
    Demonstrate parameter optimization for achieving target active fractions.
    """
    print("=" * 70)
    print("PARAMETER OPTIMIZER DEMONSTRATION")
    print("=" * 70)
    
    # Create model
    model = create_multi_degree_model()
    print(f"Model created: {model.name}")
    print(f"  Species: {len(model.states)}, Parameters: {len(model.parameters)}")
    print(f"  Active states: {[s for s in model.states.keys() if s.endswith('a')]}")
    
    # Create optimizer
    optimizer = ParameterOptimizer(model, random_seed=42)
    print("\nOptimizer initialized")
    
    # Generate targets
    pre_targets, post_targets = optimizer.generate_targets(
        pre_drug_range=(0.5, 0.7),   # Pre-drug: 50-70% active
        post_drug_range=(0.2, 0.5)   # Post-drug: 20-50% active
    )
    
    print(f"\nGenerated targets:")
    print(f"  Pre-drug targets: {len(pre_targets)} species")
    print(f"  Post-drug targets: {len(post_targets)} species (degree 1 only)")
    
    # Show some targets
    print("\nExample targets:")
    for i, (species, target) in enumerate(list(pre_targets.items())[:3]):
        inactive_state = species[:-1]
        total = model.states[inactive_state] + model.states[species]
        fraction = target / total
        print(f"  {species}: target = {target:.1f} ({fraction:.1%} of total)")
    
    # Run optimization with more iterations for better convergence
    print("\n" + "-" * 70)
    print("Running local optimization (L-BFGS-B)...")
    result_local = optimizer.optimize(
        pre_drug_range=(0.5, 0.7),
        post_drug_range=(0.2, 0.5),
        max_iterations=30,   # More iterations for better convergence
        tolerance=1e-3,      # Reasonable tolerance
        method='L-BFGS-B'
    )
    
    print("\nLocal optimization results:")
    print(f"  Success: {result_local['success']}")
    print(f"  Final error: {result_local['final_error']:.6f}")
    print(f"  Iterations: {result_local['n_iterations']}")
    print(f"  Method: {result_local.get('method', 'L-BFGS-B')}")
    
    # Try global optimization if local fails
    if not result_local['success'] or result_local['final_error'] > 0.1:
        print("\n" + "-" * 70)
        print("Local optimization unsatisfactory, trying global optimization...")
        print("Note: Differential evolution explores parameter space more thoroughly")
        
        # For demo, use small population and iterations
        result_global = optimizer.optimize(
            pre_drug_range=(0.5, 0.7),
            post_drug_range=(0.2, 0.5),
            max_iterations=10,   # Small for demo (would be larger in production)
            tolerance=1e-2,      # Looser tolerance for speed
            method='differential_evolution',
            workers=1            # Single worker for demo
        )
        
        print("\nGlobal optimization results:")
        print(f"  Success: {result_global['success']}")
        print(f"  Final error: {result_global['final_error']:.6f}")
        print(f"  Iterations: {result_global['n_iterations']}")
        print(f"  Method: {result_global.get('method', 'differential_evolution')}")
        
        # Use whichever is better
        if result_global['final_error'] < result_local['final_error']:
            result = result_global
            print("  Using global optimization results")
        else:
            result = result_local
            print("  Using local optimization results")
    else:
        result = result_local
    
    print("\nSelected optimization results:")
    print(f"  Success: {result['success']}")
    print(f"  Final error: {result['final_error']:.6f}")
    print(f"  Iterations: {result['n_iterations']}")
    print(f"  Message: {result['message']}")
    
    # Show parameter changes
    print("\nParameter optimization summary:")
    initial_params = result['initial_parameters']
    optimized_params = result['optimized_parameters']
    
    # Calculate relative changes for the top 5 parameters
    changes = []
    for param_name in optimized_params:
        initial = initial_params[param_name]
        optimized = optimized_params[param_name]
        rel_change = (optimized - initial) / initial if initial != 0 else 0
        changes.append((param_name, initial, optimized, rel_change))
    
    # Sort by absolute change
    changes.sort(key=lambda x: abs(x[3]), reverse=True)
    
    print("\nTop 5 parameter changes:")
    for param_name, initial, optimized, rel_change in changes[:5]:
        print(f"  {param_name:15s}: {initial:8.3f} → {optimized:8.3f} ({rel_change:+.1%})")
    
    return model, result


def simulate_and_validate(model, result_dict):
    """
    Simulate with optimized parameters and validate against targets.
    Always simulates regardless of optimization success flag.
    
    Args:
        model: Original model
        result_dict: Optimization result dictionary from ParameterOptimizer
    
    Returns:
        Simulation results dataframe
    """
    print("\n" + "=" * 70)
    print("SIMULATION VALIDATION")
    print("=" * 70)
    
    optimized_params = result_dict['optimized_parameters']
    pre_targets = result_dict['pre_drug_targets']
    post_targets = result_dict['post_drug_targets']
    achieved_fractions = result_dict.get('achieved_fractions', {})
    
    print(f"Optimization success flag: {result_dict['success']}")
    print(f"Final error: {result_dict['final_error']:.6f}")
    
    # Apply optimized parameters
    model_copy = model.copy()
    for param_name, param_value in optimized_params.items():
        model_copy.set_parameter(param_name, param_value)
    
    # Simulate
    solver = RoadrunnerSolver()
    try:
        solver.compile(model_copy.get_sbml_model())
        result = solver.simulate(start=0, stop=1500, step=151)
        
        print("Simulation successful!")
        
        # Extract pre-drug concentrations (just before drug)
        drug_time = 500.0
        pre_time_idx = np.abs(result['time'] - drug_time * 0.9).argmin()
        post_time_idx = np.abs(result['time'] - (drug_time + 500)).argmin()
        
        # Calculate active fractions
        print("\nTarget vs Achieved Active Fractions:")
        print("-" * 70)
        print(f"{'Species':10s} {'Target Pre':>12s} {'Achieved Pre':>12s} {'Target Post':>12s} {'Achieved Post':>12s}")
        print("-" * 70)
        
        # Degree 1 active species
        degree1_active = ['R1_1a', 'I1_1a', 'Oa']
        for species in degree1_active:
            if species in result.columns and species in pre_targets:
                inactive_state = species[:-1]
                total = model.states[inactive_state] + model.states[species]
                
                pre_conc = result[species].iloc[pre_time_idx]
                post_conc = result[species].iloc[post_time_idx]
                
                target_pre_fraction = pre_targets[species] / total
                achieved_pre_fraction = pre_conc / total
                
                target_post_fraction = post_targets.get(species, 0) / total if species in post_targets else 0
                achieved_post_fraction = post_conc / total
                
                # Show from achieved_fractions if available
                if achieved_fractions and 'pre_drug' in achieved_fractions and species in achieved_fractions['pre_drug']:
                    achieved_pre_fraction = achieved_fractions['pre_drug'][species]
                if achieved_fractions and 'post_drug' in achieved_fractions and species in achieved_fractions['post_drug']:
                    achieved_post_fraction = achieved_fractions['post_drug'][species]
                
                pre_diff_pct = abs(achieved_pre_fraction - target_pre_fraction) / target_pre_fraction * 100
                post_diff_pct = abs(achieved_post_fraction - target_post_fraction) / target_post_fraction * 100 if target_post_fraction > 0 else 0
                
                print(f"{species:10s} {target_pre_fraction:>11.1%} {achieved_pre_fraction:>11.1%} {target_post_fraction:>11.1%} {achieved_post_fraction:>11.1%}")
        
        # Show error summary
        print("\nError Summary:")
        print(f"  Final optimization error: {result_dict['final_error']:.6f}")
        print(f"  Success flag: {result_dict['success']}")
        print(f"  Iterations: {result_dict['n_iterations']}")
        print(f"  Message: {result_dict['message']}")
        
        return result
        
    except Exception as e:
        print(f"Simulation error: {e}")
        return None


def compare_approaches():
    """
    Compare optimization approach with original KineticParameterTuner.
    """
    print("\n" + "=" * 70)
    print("COMPARISON: Optimization vs Original Tuner")
    print("=" * 70)
    
    from models.utils.kinetic_tuner import generate_parameters as tuner_generate
    
    # Create model
    model = create_multi_degree_model()
    
    # Approach 1: Original KineticParameterTuner
    print("\n1. Original KineticParameterTuner approach:")
    tuner_params = tuner_generate(
        model=model,
        active_percentage_range=(0.5, 0.7),
        random_seed=42
    )
    print(f"   Generated {len(tuner_params)} parameters")
    print(f"   Parameter range: {min(tuner_params.values()):.3f} to {max(tuner_params.values()):.3f}")
    
    # Approach 2: Optimization-based approach
    print("\n2. Optimization-based approach:")
    
    # Try different methods
    methods_to_try = ['L-BFGS-B', 'differential_evolution']
    best_result = None
    best_error = float('inf')
    
    for method in methods_to_try:
        print(f"   Trying {method}...")
        opt_result = optimize_parameters(
            model=model,
            pre_drug_range=(0.5, 0.7),
            post_drug_range=(0.2, 0.5),
            random_seed=42,
            method=method,
            workers=1
        )
        
        if opt_result['success'] and opt_result['final_error'] < best_error:
            best_result = opt_result
            best_error = opt_result['final_error']
            print(f"     Error: {opt_result['final_error']:.6f}, Success: {opt_result['success']}")
    
    opt_result = best_result if best_result is not None else opt_result
    opt_params = opt_result['optimized_parameters']
    print(f"   Optimized {len(opt_params)} parameters")
    print(f"   Final error: {opt_result['final_error']:.6f}")
    print(f"   Success: {opt_result['success']}")
    
    # Compare parameter ranges
    tuner_values = list(tuner_params.values())
    opt_values = list(opt_params.values())
    
    print("\nParameter value comparison:")
    print(f"   Tuner range: [{min(tuner_values):.3f}, {max(tuner_values):.3f}]")
    print(f"   Optimizer range: [{min(opt_values):.3f}, {max(opt_values):.3f}]")
    print(f"   Tuner mean: {np.mean(tuner_values):.3f}")
    print(f"   Optimizer mean: {np.mean(opt_values):.3f}")


def main():
    """
    Main demonstration function.
    """
    print("Multi-Degree Drug Interaction Parameter Optimization Demo")
    print("=" * 70)
    
    # Demonstrate optimization with more iterations for better results
    model, result = demonstrate_optimization()
    
    # Validate with simulation
    simulation_result = simulate_and_validate(model, result)
    
    # Compare approaches
    compare_approaches()
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nKey takeaways:")
    print("1. ParameterOptimizer finds parameters achieving both pre- and post-drug targets")
    print("2. Uses direct optimization to minimize error between actual and target fractions")
    print("3. Automatically handles drug application timing")
    print("4. Provides reproducibility through random seed control")
    print("5. Shows target vs achieved fractions for validation")
    
    if result['success']:
        print("\nRecommendations for production use:")
        print("- Increase max_iterations to 100-200 for better convergence")
        print("- Use smaller tolerance (1e-4 to 1e-5) for higher precision")
        print("- Use differential_evolution for global optimization with workers > 1 for parallelization")
        print("- For large models, use method='differential_evolution' with workers=cpu_count()")
        print("- Consider two-stage approach: global search followed by local refinement")
    else:
        print("\nOptimization did not converge successfully. Try:")
        print("- Increasing max_iterations")
        print("- Using different random seed")
        print("- Adjusting parameter bounds")


if __name__ == "__main__":
    main()
