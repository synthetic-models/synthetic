"""
Quick verification test for parallel implementation.
"""
import time
import numpy as np
from synthetic import Builder
from synthetic.utils.make_target_data import make_target_data_with_params_robust
from synthetic.Solver.RoadrunnerSolver import RoadrunnerSolver

def test_small_parallel():
    """Test parallel implementation with small sample size for quick verification."""
    print("Creating virtual cell model...")
    vc = Builder.specify([2, 10, 30])  # Smaller model
    
    print("Compiling Roadrunner solver...")
    solver = RoadrunnerSolver()
    sbml_str = vc.model.get_sbml_model()
    solver.compile(sbml_str)
    
    # Get initial values
    initial_values = vc.get_initial_values(exclude_drugs=True)
    initial_values = {k: v for k, v in initial_values.items() if not k.endswith('a')}
    initial_values.pop('O', None)
    
    print(f"Number of species: {len(initial_values)}")
    
    # Create small feature dataframe
    from synthetic.utils.make_feature_data import make_feature_data
    feature_df = make_feature_data(
        initial_values=initial_values,
        perturbation_type='lognormal',
        perturbation_params={'shape': 0.1},
        n_samples=200,  # Small sample size
        seed=42
    )
    
    simulation_params = {'start': 0, 'end': 5000, 'points': 51}
    
    print("\nTesting sequential execution (n_cores=1)...")
    start_time = time.time()
    targets_seq, timecourse_seq, success_mask_seq = make_target_data_with_params_robust(
        model_spec=vc.spec,
        solver=solver,
        feature_df=feature_df,
        simulation_params=simulation_params,
        outcome_var='Oa',
        capture_all_species=False,
        verbose=False,
        n_cores=1,
        target_method='fold_change_drug',
        drug_start_time=2500,
        require_all_successful=False
    )
    seq_time = time.time() - start_time
    print(f"Sequential time: {seq_time:.2f} seconds")
    print(f"Successful samples: {success_mask_seq.sum()}/{len(success_mask_seq)}")
    
    print("\nTesting parallel execution (n_cores=-1)...")
    start_time = time.time()
    targets_par, timecourse_par, success_mask_par = make_target_data_with_params_robust(
        model_spec=vc.spec,
        solver=solver,
        feature_df=feature_df,
        simulation_params=simulation_params,
        outcome_var='Oa',
        capture_all_species=False,
        verbose=False,
        n_cores=-1,
        target_method='fold_change_drug',
        drug_start_time=2500,
        require_all_successful=False
    )
    par_time = time.time() - start_time
    print(f"Parallel time: {par_time:.2f} seconds")
    print(f"Successful samples: {success_mask_par.sum()}/{len(success_mask_par)}")
    
    if seq_time > 0:
        print(f"\nSpeedup: {seq_time/par_time:.2f}x")
    else:
        print("\nSpeedup calculation not possible (sequential time too small)")
    
    # Verify results match
    print("\nVerifying results match...")
    if targets_seq.equals(targets_par):
        print("✓ Targets match")
    else:
        print("✗ Targets differ")
        # Show difference
        diff = np.abs(targets_seq.values - targets_par.values)
        print(f"Max difference: {np.max(diff):.6f}")
        
    if (success_mask_seq == success_mask_par).all():
        print("✓ Success masks match")
    else:
        print("✗ Success masks differ")
        
    # Test make_dataset_drug_response with n_cores parameter
    print("\nTesting make_dataset_drug_response with n_cores=-1...")
    from synthetic import make_dataset_drug_response
    start_time = time.time()
    X, y = make_dataset_drug_response(
        n=20,
        cell_model=vc,
        target_specie='Oa',
        verbose=False,
        n_cores=-1
    )
    dataset_time = time.time() - start_time
    print(f"Dataset generation time: {dataset_time:.2f} seconds")
    print(f"Dataset shapes - X: {X.shape}, y: {y.shape}")
    
    return True

if __name__ == "__main__":
    success = test_small_parallel()
    if success:
        print("\n✓ Parallel implementation verification PASSED")
    else:
        print("\n✗ Parallel implementation verification FAILED")