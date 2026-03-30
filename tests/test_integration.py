"""
Integration test for the Synthetic library workflow as defined in CLAUDE.md.

This test validates the complete network generation and simulation workflow:
1. Create specification with hierarchical degrees
2. Add drugs
3. Generate model
4. Compile and use
5. Get Antimony/SBML
6. Simulate with both ScipySolver and RoadrunnerSolver
"""

import sys
import os
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

# Add src directory to path for imports
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Also add models alias for backward compatibility with code imports
# (the code imports from 'models.Solver' but package is 'synthetic')
sys.path.insert(0, str(project_root))



def test_network_generation_workflow():
    """
    Test the complete network generation workflow as described in CLAUDE.md.
    This follows the example from the documentation.
    """
    # 1. Create specification with hierarchical degrees
    # Using degree_cascades=[1, 2, 5] means:
    # Degree 1 has 1 cascade, Degree 2 has 2, Degree 3 has 5
    from synthetic.Specs.DegreeInteractionSpec import DegreeInteractionSpec

    spec = DegreeInteractionSpec(degree_cascades=[1, 2, 5])
    spec.generate_specifications(random_seed=42, feedback_density=0.5)

    # Verify specification generated correctly
    assert len(spec.degree_cascades) == 3
    assert spec.get_total_species_count() > 0
    assert len(spec.regulations) > 0

    # 2. Generate model
    model = spec.generate_network("multi_degree_network", random_seed=42)

    # 3. Compile and use
    model.precompile()

    # Verify model is pre-compiled
    assert model.pre_compiled is True
    assert len(model.states) > 0
    assert len(model.parameters) > 0

    # 4. Get Antimony/SBML
    antimony_str = model.get_antimony_model()
    assert isinstance(antimony_str, str)
    assert "model multi_degree_network" in antimony_str
    assert "end" in antimony_str

    sbml_str = model.get_sbml_model()
    assert isinstance(sbml_str, str)
    assert "<?xml" in sbml_str or "<sbml" in sbml_str.lower()

    # 5. Test parameter and state manipulation
    # These should work after precompile()
    state_names = list(model.states.keys())
    param_names = list(model.parameters.keys())

    # Test set/get state
    if state_names:
        original_value = model.get_state(state_names[0])
        model.set_state(state_names[0], original_value * 2)
        assert model.get_state(state_names[0]) == original_value * 2

    # Test set/get parameter
    if param_names:
        original_value = model.get_parameter(param_names[0])
        model.set_parameter(param_names[0], original_value * 2)
        assert model.get_parameter(param_names[0]) == original_value * 2


def test_network_with_drugs():
    """
    Test network generation with drugs as described in CLAUDE.md.
    """
    from synthetic.Specs.DegreeInteractionSpec import DegreeInteractionSpec
    from synthetic.Specs.Drug import Drug

    # Create specification
    spec = DegreeInteractionSpec(degree_cascades=[1, 2, 5])
    spec.generate_specifications(random_seed=42, feedback_density=0.5)

    # Add drug (must target degree 1 R species only)
    drug = Drug(
        name="DrugX",
        start_time=500,
        default_value=0,
        regulation=["R1_1"],  # Target degree 1 R species
        regulation_type=["up"]
    )
    spec.add_drug(drug, value=10.0)

    # Generate model with drug
    model = spec.generate_network("network_with_drug", random_seed=42)
    model.precompile()

    # Verify drug is included in model
    antimony_str = model.get_antimony_model()
    assert "DrugX" in antimony_str
    assert "piecewise" in antimony_str  # Drugs use piecewise assignment rules


def test_scipy_solver_simulation():
    """
    Test simulation with ScipySolver as described in CLAUDE.md.
    ScipySolver accepts Antimony format and supports JIT compilation.
    """
    from synthetic.Specs.DegreeInteractionSpec import DegreeInteractionSpec
    from synthetic.Solver.ScipySolver import ScipySolver

    # Create specification and model
    spec = DegreeInteractionSpec(degree_cascades=[1, 2])
    spec.generate_specifications(random_seed=42, feedback_density=0.5)
    model = spec.generate_network("scipy_test_network", random_seed=42)
    model.precompile()

    # Get Antimony string
    antimony_str = model.get_antimony_model()

    # Create and compile solver
    solver = ScipySolver()
    solver.compile(antimony_str, jit=False)  # Disable JIT for faster test compilation

    # Simulate
    results = solver.simulate(start=0, stop=100, step=20)

    # Verify results
    assert isinstance(results, pd.DataFrame)
    assert "time" in results.columns
    assert len(results) > 0
    assert len(results.columns) > 1  # At least time + one species

    # Verify time column
    time_values = results["time"].values
    assert time_values[0] >= 0
    assert time_values[-1] <= 100


def test_scipy_solver_with_jit():
    """
    Test ScipySolver with JIT compilation enabled.
    """
    from synthetic.Specs.DegreeInteractionSpec import DegreeInteractionSpec
    from synthetic.Solver.ScipySolver import ScipySolver

    spec = DegreeInteractionSpec(degree_cascades=[1])
    spec.generate_specifications(random_seed=42)
    model = spec.generate_network("jit_test_network", random_seed=42)
    model.precompile()

    antimony_str = model.get_antimony_model()

    solver = ScipySolver()
    solver.compile(antimony_str, jit=True)  # Enable JIT

    results = solver.simulate(start=0, stop=50, step=10)

    assert isinstance(results, pd.DataFrame)
    assert "time" in results.columns


def test_roadrunner_solver_simulation():
    """
    Test simulation with RoadrunnerSolver as described in CLAUDE.md.
    RoadrunnerSolver requires SBML format, not Antimony.
    """
    pytest.importorskip("roadrunner")
    from synthetic.Specs.DegreeInteractionSpec import DegreeInteractionSpec
    from synthetic.Solver.RoadrunnerSolver import RoadrunnerSolver

    # Create specification and model
    spec = DegreeInteractionSpec(degree_cascades=[1, 2])
    spec.generate_specifications(random_seed=42, feedback_density=0.5)
    model = spec.generate_network("roadrunner_test_network", random_seed=42)
    model.precompile()

    # Get SBML string (RoadrunnerSolver requires SBML)
    sbml_str = model.get_sbml_model()

    # Create and compile solver
    solver = RoadrunnerSolver()
    solver.compile(sbml_str)

    # Simulate
    results = solver.simulate(start=0, stop=100, step=20)

    # Verify results
    assert isinstance(results, pd.DataFrame)
    assert "time" in results.columns
    assert len(results) > 0


def test_solver_state_parameter_manipulation():
    """
    Test that solvers can modify state and parameter values after compilation.
    """
    from synthetic.Specs.DegreeInteractionSpec import DegreeInteractionSpec
    from synthetic.Solver.ScipySolver import ScipySolver

    spec = DegreeInteractionSpec(degree_cascades=[1])
    spec.generate_specifications(random_seed=42)
    model = spec.generate_network("manipulation_test", random_seed=42)
    model.precompile()

    antimony_str = model.get_antimony_model()

    solver = ScipySolver()
    solver.compile(antimony_str, jit=False)

    # Get original simulation results
    results1 = solver.simulate(start=0, stop=10, step=5)

    # Modify a parameter
    species = solver.species
    if len(species) > 0:
        # Set new initial state
        new_state_values = {species[0]: solver.y0[0] * 2}
        success = solver.set_state_values(new_state_values)
        assert success is True

        # Run simulation with new state
        results2 = solver.simulate(start=0, stop=10, step=5)

        # Results should be different due to changed initial condition
        assert results1[species[0]].iloc[0] != results2[species[0]].iloc[0]


def test_reproducibility_with_random_seed():
    """
    Test that using the same random seed produces identical results.
    """
    from synthetic.Specs.DegreeInteractionSpec import DegreeInteractionSpec

    # Generate first network
    spec1 = DegreeInteractionSpec(degree_cascades=[1, 2])
    spec1.generate_specifications(random_seed=42, feedback_density=0.5)
    model1 = spec1.generate_network("repro_test_1", random_seed=42)
    model1.precompile()

    # Generate second network with same seed
    spec2 = DegreeInteractionSpec(degree_cascades=[1, 2])
    spec2.generate_specifications(random_seed=42, feedback_density=0.5)
    model2 = spec2.generate_network("repro_test_2", random_seed=42)
    model2.precompile()

    # Both models should have same structure
    assert len(model1.states) == len(model2.states)
    assert len(model1.parameters) == len(model2.parameters)
    assert set(model1.states.keys()) == set(model2.states.keys())


def test_degree_interaction_spec_structure():
    """
    Test that DegreeInteractionSpec creates the correct species structure.
    """
    from synthetic.Specs.DegreeInteractionSpec import DegreeInteractionSpec

    spec = DegreeInteractionSpec(degree_cascades=[1, 2, 5])
    spec.generate_specifications(random_seed=42)

    # Check species names follow convention
    all_species = spec.get_species_by_degree(1, 'all')
    assert len(all_species) == 2  # R1_1, I1_1

    all_species_deg2 = spec.get_species_by_degree(2, 'all')
    assert len(all_species_deg2) == 4  # R2_1, I2_1, R2_2, I2_2

    all_species_deg3 = spec.get_species_by_degree(3, 'all')
    assert len(all_species_deg3) == 10  # 5 cascades * 2 species

    # Check regulations count
    regs = spec.get_regulations_by_degree()
    assert len(regs) > 0


def test_simple_model_generation():
    """
    Test minimal model generation without drugs.
    """
    from synthetic.Specs.DegreeInteractionSpec import DegreeInteractionSpec
    from synthetic.Solver.ScipySolver import ScipySolver

    # Simple single-degree network
    spec = DegreeInteractionSpec(degree_cascades=[1])
    spec.generate_specifications(random_seed=42)
    model = spec.generate_network("simple_model", random_seed=42)
    model.precompile()

    # Verify model structure
    assert "O" in model.states  # Outcome species
    assert len(model.states) >= 2  # At least R1_1, I1_1, O

    # Get Antimony and verify it's valid
    antimony = model.get_antimony_model()
    assert "model simple_model" in antimony

    # Compile and simulate
    solver = ScipySolver()
    solver.compile(antimony, jit=False)
    # Note: 'step' in simulate() is the number of points, not interval
    results = solver.simulate(start=0, stop=10, step=6)

    assert len(results) == 6  # 6 time points from 0 to 10
    assert "O" in results.columns


def test_get_species_list_scipy():
    """
    Test that ScipySolver.get_species_list() returns species names correctly.
    """
    from synthetic.Specs.DegreeInteractionSpec import DegreeInteractionSpec
    from synthetic.Solver.ScipySolver import ScipySolver

    spec = DegreeInteractionSpec(degree_cascades=[1, 2])
    spec.generate_specifications(random_seed=42)
    model = spec.generate_network("species_list_test", random_seed=42)
    model.precompile()

    solver = ScipySolver()
    solver.compile(model.get_antimony_model(), jit=False)

    species_list = solver.get_species_list()

    # Verify species list is returned and contains expected species
    assert isinstance(species_list, list)
    assert len(species_list) > 0
    assert "O" in species_list  # Outcome species should be present


def test_get_species_list_roadrunner():
    """
    Test that RoadrunnerSolver.get_species_list() returns species names correctly.
    """
    pytest.importorskip("roadrunner")
    from synthetic.Specs.DegreeInteractionSpec import DegreeInteractionSpec
    from synthetic.Solver.RoadrunnerSolver import RoadrunnerSolver

    spec = DegreeInteractionSpec(degree_cascades=[1, 2])
    spec.generate_specifications(random_seed=42)
    model = spec.generate_network("species_list_test_rr", random_seed=42)
    model.precompile()

    solver = RoadrunnerSolver()
    solver.compile(model.get_sbml_model())

    species_list = solver.get_species_list()

    # Verify species list is returned and contains expected species
    assert isinstance(species_list, list)
    assert len(species_list) > 0
    assert "O" in species_list  # Outcome species should be present


def test_get_species_list_matches_simulation_columns():
    """
    Test that get_species_list() returns the same species as simulation output columns.
    """
    from synthetic.Specs.DegreeInteractionSpec import DegreeInteractionSpec
    from synthetic.Solver.ScipySolver import ScipySolver

    spec = DegreeInteractionSpec(degree_cascades=[1])
    spec.generate_specifications(random_seed=42)
    model = spec.generate_network("species_match_test", random_seed=42)
    model.precompile()

    # Test ScipySolver
    scipy_solver = ScipySolver()
    scipy_solver.compile(model.get_antimony_model(), jit=False)
    scipy_species = set(scipy_solver.get_species_list())
    scipy_results = scipy_solver.simulate(start=0, stop=10, step=5)
    scipy_columns = set(scipy_results.columns) - {"time"}

    assert scipy_species == scipy_columns, f"ScipySolver mismatch: {scipy_species} vs {scipy_columns}"

    # Test RoadrunnerSolver (skip if roadrunner not available)
    pytest.importorskip("roadrunner")
    from synthetic.Solver.RoadrunnerSolver import RoadrunnerSolver

    rr_solver = RoadrunnerSolver()
    rr_solver.compile(model.get_sbml_model())
    rr_species = set(rr_solver.get_species_list())
    rr_results = rr_solver.simulate(start=0, stop=10, step=5)
    rr_columns = set(rr_results.columns) - {"time"}

    assert rr_species == rr_columns, f"RoadrunnerSolver mismatch: {rr_species} vs {rr_columns}"


def test_get_species_list_raises_before_compile():
    """
    Test that get_species_list() raises RuntimeError when called before compile().
    """
    from synthetic.Solver.ScipySolver import ScipySolver

    # Test ScipySolver
    scipy_solver = ScipySolver()
    with pytest.raises(RuntimeError, match="compile"):
        scipy_solver.get_species_list()

    # Test RoadrunnerSolver (skip if roadrunner not available)
    pytest.importorskip("roadrunner")
    from synthetic.Solver.RoadrunnerSolver import RoadrunnerSolver

    rr_solver = RoadrunnerSolver()
    with pytest.raises(RuntimeError, match="compile|created"):
        rr_solver.get_species_list()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
