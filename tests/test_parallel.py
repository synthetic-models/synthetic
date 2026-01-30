"""
Test script for parallel implementation of make_target_data_with_params_robust.
"""
import time
from synthetic import Builder
from synthetic.utils.make_target_data import make_target_data_with_params_robust
from synthetic.Solver.RoadrunnerSolver import RoadrunnerSolver

def test_parallel_speedup():
    """Test parallel vs sequential execution."""
    print("Creating virtual cell model...")
    vc = Builder.specify([3, 10, 20])
    
    print("Compiling Roadrunner solver...")
    solver = RoadrunnerSolver()
    sbml_str = vc.model.get_sbml_model()
    solver.compile(sbml_str)
    
    # Get initial values (excluding drugs)
    initial_values = vc.get_initial_values(exclude_drugs=True)
    # Exclude any species with 'a' at the end of their name (outcome species and activated species)
    # Since they should not be perturbed initially
    initial_values = {k: v for k, v in initial_values.items() if not k.endswith('a')}
    initial_values.pop('O', None)
    
    print(f"Number of species: {len(initial_values)}")
    
    # Create feature dataframe with perturbations
    from synthetic.utils.make_feature_data import make_feature_data
    feature_df = make_feature_data(
        initial_values=initial_values,
        perturbation_type='lognormal',
        perturbation_params={'shape': 0.1},
        n_samples=500,
        seed=42
    )
    
    simulation_params = {'start': 0, 'end': 10000, 'points': 101}
    
    print("\nTesting sequential execution (n_cores=1)...")
    start_time = time.time()
    targets_seq, timecourse_seq, success_mask_seq = make_target_data_with_params_robust(
        solver=solver,
        feature_df=feature_df,
        simulation_params=simulation_params,
        outcome_var='Oa',
        capture_all_species=False,
        verbose=True,
        n_cores=1,
        target_method='fold_change_drug',
        drug_start_time=5000,
        require_all_successful=False
    )
    seq_time = time.time() - start_time
    print(f"Sequential time: {seq_time:.2f} seconds")
    print(f"Successful samples: {success_mask_seq.sum()}/{len(success_mask_seq)}")
    
    print("\nTesting parallel execution (n_cores=-1)...")
    start_time = time.time()
    targets_par, timecourse_par, success_mask_par = make_target_data_with_params_robust(
        solver=solver,
        feature_df=feature_df,
        simulation_params=simulation_params,
        outcome_var='Oa',
        capture_all_species=False,
        verbose=True,
        n_cores=4,
        target_method='fold_change_drug',
        drug_start_time=5000,
        require_all_successful=False
    )
    par_time = time.time() - start_time
    print(f"Parallel time: {par_time:.2f} seconds")
    print(f"Successful samples: {success_mask_par.sum()}/{len(success_mask_par)}")
    
    print(f"\nSpeedup: {seq_time/par_time:.2f}x")
    
    # Verify results match
    print("\nVerifying results match...")
    if targets_seq.equals(targets_par):
        print("✓ Targets match")
    else:
        print("✗ Targets differ")
        
    if (success_mask_seq == success_mask_par).all():
        print("✓ Success masks match")
    else:
        print("✗ Success masks differ")
        
    return seq_time, par_time

if __name__ == "__main__":
    test_parallel_speedup()
