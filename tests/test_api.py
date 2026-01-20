"""
Integration tests for the high-level API.

Tests for VirtualCell, Builder, and make_dataset_drug_response functionality.
"""

import pytest
import numpy as np
from synthetic import Builder, VirtualCell, make_dataset_drug_response


class TestVirtualCell:
    """Tests for VirtualCell class."""

    def test_builder_creates_valid_virtual_cell(self):
        """Test that Builder.specify() creates valid VirtualCell."""
        vc = Builder.specify(degree_cascades=[1, 2, 5])
        assert isinstance(vc, VirtualCell)
        assert not vc._compiled

    def test_add_drug_method_chaining(self):
        """Test that add_drug() returns self for chaining."""
        vc = Builder.specify(degree_cascades=[1, 2, 5])
        result = vc.add_drug(
            name="DrugX",
            start_time=500.0,
            default_value=0.0,
            regulation=["R1_1"],
            regulation_type=["up"],
        )
        assert result is vc
        assert len(vc._drugs) == 1

    def test_compile_with_kinetic_tuner(self):
        """Test VirtualCell.compile() with kinetic tuner enabled."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42)
        vc.compile(use_kinetic_tuner=True)
        assert vc._compiled
        assert vc._tuner is not None
        assert vc._spec is not None
        assert vc._model is not None

    def test_compile_without_kinetic_tuner(self):
        """Test VirtualCell.compile() with kinetic tuner disabled."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42)
        vc.compile(use_kinetic_tuner=False)
        assert vc._compiled
        assert vc._tuner is None
        assert vc._spec is not None
        assert vc._model is not None

    def test_get_species_names_excludes_drugs(self):
        """Test that get_species_names() excludes drugs by default."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42)
        vc.add_drug(name="DrugX", regulation=["R1_1"], regulation_type=["up"])
        vc.compile()

        species_names = vc.get_species_names()
        # Check that DrugX is not in species
        assert "DrugX" not in species_names

    def test_get_initial_values_excludes_drugs(self):
        """Test that get_initial_values() excludes drugs by default."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42)
        vc.add_drug(name="DrugX", regulation=["R1_1"], regulation_type=["up"])
        vc.compile()

        initial_values = vc.get_initial_values()
        # Check that DrugX is not in initial values
        assert "DrugX" not in initial_values
        # Check that other species are present
        assert len(initial_values) > 0

    def test_get_target_concentrations_with_tuner(self):
        """Test that get_target_concentrations() returns values with tuner."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42)
        vc.compile(use_kinetic_tuner=True)
        target_concs = vc.get_target_concentrations()
        assert len(target_concs) > 0
        # All keys should be species ending with 'a' (active forms)
        for key in target_concs.keys():
            assert key.endswith('a')

    def test_get_target_concentrations_without_tuner(self):
        """Test that get_target_concentrations() returns empty dict without tuner."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42)
        vc.compile(use_kinetic_tuner=False)
        target_concs = vc.get_target_concentrations()
        assert len(target_concs) == 0

    def test_spec_property_raises_if_not_compiled(self):
        """Test that spec property raises ValueError if not compiled."""
        vc = Builder.specify(degree_cascades=[1, 2, 5])
        with pytest.raises(ValueError, match="must be compiled"):
            _ = vc.spec

    def test_model_property_raises_if_not_compiled(self):
        """Test that model property raises ValueError if not compiled."""
        vc = Builder.specify(degree_cascades=[1, 2, 5])
        with pytest.raises(ValueError, match="must be compiled"):
            _ = vc.model


class TestMakeDatasetDrugResponse:
    """Tests for make_dataset_drug_response function."""

    def test_returns_correct_shapes(self):
        """Test that make_dataset_drug_response() returns correct shapes."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42)
        vc.compile()

        n_samples = 100
        X, y = make_dataset_drug_response(
            n=n_samples,
            cell_model=vc,
            perturbation_type='gaussian',
            perturbation_params={'rsd': 0.2},
            seed=42,
        )

        assert X.shape[0] == n_samples
        assert y.shape[0] == n_samples
        # X should have one column per species
        assert X.shape[1] == len(vc.get_species_names())

    def test_drugs_not_in_x_features(self):
        """Test that drugs are NOT included in X features."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42)
        vc.add_drug(name="DrugX", regulation=["R1_1"], regulation_type=["up"])
        vc.compile()

        X, y = make_dataset_drug_response(
            n=100,
            cell_model=vc,
            perturbation_type='gaussian',
            perturbation_params={'rsd': 0.2},
            seed=42,
        )
        species_names = vc.get_species_names()

        # Check number of columns matches species (excludes drugs)
        assert X.shape[1] == len(species_names)

    def test_raises_if_not_compiled(self):
        """Test that make_dataset_drug_response() raises if model not compiled."""
        vc = Builder.specify(degree_cascades=[1, 2, 5])

        with pytest.raises(ValueError, match="must be compiled"):
            make_dataset_drug_response(n=100, cell_model=vc)

    def test_with_drug_regulation(self):
        """Test dataset generation with drug regulation."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42)
        vc.add_drug(
            name="DrugD",
            start_time=500.0,
            regulation=["R1_1"],
            regulation_type=["down"],
        )
        vc.compile()

        X, y = make_dataset_drug_response(
            n=100,
            cell_model=vc,
            perturbation_type='gaussian',
            perturbation_params={'rsd': 0.2},
            seed=42,
        )
        assert X.shape[0] == 100
        assert y.shape[0] == 100
        # y should have valid values
        assert np.all(np.isfinite(y))

    def test_custom_simulation_params(self):
        """Test with custom simulation parameters."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42)
        vc.compile()

        sim_params = {'start': 0, 'end': 500, 'points': 50}
        X, y = make_dataset_drug_response(
            n=50,
            cell_model=vc,
            perturbation_type='gaussian',
            perturbation_params={'rsd': 0.2},
            simulation_params=sim_params,
            seed=42,
        )

        assert X.shape[0] == 50
        assert y.shape[0] == 50

    def test_custom_perturbation_type(self):
        """Test with custom perturbation type."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42)
        vc.compile()

        # Test each perturbation type with appropriate parameters
        X, y = make_dataset_drug_response(
            n=50,
            cell_model=vc,
            perturbation_type='uniform',
            perturbation_params={'min': 0.8, 'max': 1.2},
            seed=42,
        )
        assert X.shape == (50, len(vc.get_species_names()))
        assert y.shape == (50,)

        X, y = make_dataset_drug_response(
            n=50,
            cell_model=vc,
            perturbation_type='gaussian',
            perturbation_params={'rsd': 0.2},
            seed=42,
        )
        assert X.shape == (50, len(vc.get_species_names()))
        assert y.shape == (50,)

    def test_scipy_and_roadrunner_solvers(self):
        """Test both scipy and roadrunner solvers."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42)
        vc.compile()

        # Scipy solver
        X_scipy, y_scipy = make_dataset_drug_response(
            n=50,
            cell_model=vc,
            solver_type='scipy',
            perturbation_type='gaussian',
            perturbation_params={'rsd': 0.2},
            seed=42,
        )

        # Roadrunner solver
        X_rr, y_rr = make_dataset_drug_response(
            n=50,
            cell_model=vc,
            solver_type='roadrunner',
            perturbation_type='gaussian',
            perturbation_params={'rsd': 0.2},
            seed=42,
        )

        assert X_scipy.shape == X_rr.shape
        assert y_scipy.shape == y_rr.shape

    def test_invalid_solver_type_raises(self):
        """Test that invalid solver type raises ValueError."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42)
        vc.compile()

        with pytest.raises(ValueError, match="Unsupported solver_type"):
            make_dataset_drug_response(
                n=50,
                cell_model=vc,
                solver_type='invalid',
                perturbation_type='gaussian',
                perturbation_params={'rsd': 0.2},
                seed=42,
            )


class TestKineticTuning:
    """Tests for kinetic parameter tuning integration."""

    def test_kinetic_tuned_parameters_produce_signal(self):
        """Test that kinetic tuned parameters produce reasonable signal propagation."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42)
        vc.compile(
            use_kinetic_tuner=True,
            active_percentage_range=(0.3, 0.7),
        )

        X, y = make_dataset_drug_response(
            n=100,
            cell_model=vc,
            target_specie='Oa',
            perturbation_type='gaussian',
            perturbation_params={'rsd': 0.2},
            seed=42,
        )

        # Check that output values vary (signal propagated)
        assert np.std(y) > 0

    def test_target_concentrations_match_active_range(self):
        """Test that target concentrations are within specified active percentage range."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42)
        vc.compile(
            use_kinetic_tuner=True,
            active_percentage_range=(0.3, 0.7),
        )

        target_concs = vc.get_target_concentrations()
        initial_values = vc.get_initial_values()

        # For each active species, check it's within expected range
        for active_species, target_conc in target_concs.items():
            inactive_species = active_species[:-1]
            total = initial_values.get(inactive_species, 100) + target_conc
            # Active percentage should be roughly in [0.3, 0.7]
            # (allowing some tolerance due to rounding)
            active_pct = target_conc / total
            assert 0.2 < active_pct < 0.8  # Slightly wider range for tolerance


class TestBuilder:
    """Tests for Builder factory class."""

    def test_builder_returns_virtual_cell(self):
        """Test that Builder.specify() returns VirtualCell instance."""
        vc = Builder.specify(degree_cascades=[1, 2, 3])
        assert isinstance(vc, VirtualCell)

    def test_builder_with_all_parameters(self):
        """Test Builder.specify() with all parameters."""
        vc = Builder.specify(
            degree_cascades=[1, 2, 5, 10],
            name="TestCell",
            random_seed=123,
            feedback_density=0.7,
        )
        assert vc._degree_cascades == [1, 2, 5, 10]
        assert vc._name == "TestCell"
        assert vc._random_seed == 123
        assert vc._feedback_density == 0.7
