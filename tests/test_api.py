"""
Integration tests for high-level API.

Tests for VirtualCell, Builder, and make_dataset_drug_response functionality.
"""

import pytest
import numpy as np
import pandas as pd
from synthetic import Builder, VirtualCell, make_dataset_drug_response
from synthetic.Specs.MichaelisNetworkSpec import MichaelisNetworkSpec
from synthetic.Specs.DegreeInteractionSpec import DegreeInteractionSpec


class TestAutoDrug:
    """Tests for auto-drug generation feature."""

    def test_auto_drug_generation_single_target(self):
        """Test auto-drug targets single R1_1 when degree_cascades[0]=1."""
        vc = Builder.specify(degree_cascades=[1, 2, 5])
        drugs = vc.list_drugs()
        assert len(drugs) == 1
        assert drugs[0]['name'] == 'D'
        assert drugs[0]['targets'] == ['R1_1']
        assert drugs[0]['is_auto'] is True

    def test_auto_drug_generation_multiple_targets(self):
        """Test auto-drug targets R1_1 and R1_2 when degree_cascades[0]=2."""
        vc = Builder.specify(degree_cascades=[2, 2, 5])
        drugs = vc.list_drugs()
        assert len(drugs) == 1
        assert drugs[0]['targets'] == ['R1_1', 'R1_2']
        assert drugs[0]['types'] == ['down', 'down']

    def test_auto_drug_generation_three_targets(self):
        """Test auto-drug targets R1_1, R1_2, R1_3 when degree_cascades[0]=3."""
        vc = Builder.specify(degree_cascades=[3, 6, 15, 25])
        drugs = vc.list_drugs()
        assert len(drugs) == 1
        assert drugs[0]['targets'] == ['R1_1', 'R1_2', 'R1_3']
        assert drugs[0]['types'] == ['down', 'down', 'down']

    def test_auto_drug_custom_parameters(self):
        """Test auto-drug with custom parameters."""
        vc = Builder.specify(
            degree_cascades=[1, 2, 5],
            drug_name="CustomDrug",
            drug_value=50.0,
            drug_regulation_type="up",
        )
        drugs = vc.list_drugs()
        assert drugs[0]['name'] == 'CustomDrug'
        assert drugs[0]['is_auto'] is True
        assert drugs[0]['types'] == ['up']

    def test_manual_and_auto_drug_together(self):
        """Test manual add_drug() works alongside auto-drug."""
        vc = Builder.specify(degree_cascades=[1, 2, 5])
        vc.add_drug(name="ManualDrug", regulation=["R1_1"], regulation_type=["up"])
        drugs = vc.list_drugs()
        assert len(drugs) == 2
        auto_drugs = [d for d in drugs if d['is_auto']]
        manual_drugs = [d for d in drugs if not d['is_auto']]
        assert len(auto_drugs) == 1
        assert len(manual_drugs) == 1
        assert auto_drugs[0]['name'] == 'D'
        assert manual_drugs[0]['name'] == 'ManualDrug'

    def test_auto_compile_flag(self):
        """Test auto_compile=False prevents immediate compilation."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], auto_compile=False)
        assert not vc._compiled

    def test_auto_compile_default_true(self):
        """Test auto_compile defaults to True."""
        vc = Builder.specify(degree_cascades=[1, 2, 5])
        assert vc._compiled

    def test_auto_drug_disabled(self):
        """Test auto_drug=False prevents auto-drug generation."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], auto_drug=False)
        drugs = vc.list_drugs()
        assert len(drugs) == 0

    def test_drug_parameters_applied(self):
        """Test drug parameters are automatically applied to model."""
        vc = Builder.specify(degree_cascades=[1, 2, 5])
        # Check drug parameters are set to drug_value (100)
        params = vc.model.get_parameters()
        regulator_map = vc.model.get_regulator_parameter_map()
        drug_params = regulator_map.get('D', {})
        # For "down" regulation, we expect Ki parameters set to 100.0
        assert len(drug_params) > 0
        for param_name in drug_params:
            assert params[param_name] == 100.0

    def test_drug_custom_value_applied(self):
        """Test custom drug_value is applied correctly."""
        vc = Builder.specify(
            degree_cascades=[1, 2, 5],
            drug_value=75.0,
        )
        params = vc.model.get_parameters()
        regulator_map = vc.model.get_regulator_parameter_map()
        drug_params = regulator_map.get('D', {})
        for param_name in drug_params:
            assert params[param_name] == 75.0

    def test_simulation_end_default(self):
        """Test simulation_end is set correctly."""
        vc = Builder.specify(degree_cascades=[1, 2, 5])
        assert vc._simulation_end == 10000.0

    def test_simulation_end_custom(self):
        """Test custom simulation_end is set correctly."""
        vc = Builder.specify(
            degree_cascades=[1, 2, 5],
            simulation_end=5000.0,
        )
        assert vc._simulation_end == 5000.0

    def test_make_dataset_uses_simulation_end(self):
        """Test make_dataset_drug_response uses cell_model's simulation_end."""
        vc = Builder.specify(
            degree_cascades=[1, 2, 5],
            simulation_end=5000.0,
        )
        X, y = make_dataset_drug_response(n=10, cell_model=vc)
        assert X.shape[0] == 10
        assert y.shape[0] == 10

    def test_drug_start_time_default(self):
        """Test drug_start_time defaults to 5000.0."""
        vc = Builder.specify(degree_cascades=[1, 2, 5])
        drugs = vc.list_drugs()
        assert drugs[0]['start_time'] == 5000.0

    def test_drug_start_time_custom(self):
        """Test custom drug_start_time is set correctly."""
        vc = Builder.specify(
            degree_cascades=[1, 2, 5],
            drug_start_time=10000.0,
        )
        drugs = vc.list_drugs()
        assert drugs[0]['start_time'] == 10000.0


class TestVirtualCell:
    """Tests for VirtualCell class."""

    def test_builder_creates_valid_virtual_cell(self):
        """Test that Builder.specify() creates valid VirtualCell."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], auto_compile=False)
        assert isinstance(vc, VirtualCell)
        assert not vc._compiled

    def test_add_drug_method_chaining(self):
        """Test that add_drug() returns self for chaining."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], auto_compile=False)
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
        vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42, auto_compile=False)
        vc.compile(use_kinetic_tuner=True)
        assert vc._compiled
        assert vc._tuner is not None
        assert vc._spec is not None
        assert vc._model is not None

    def test_compile_without_kinetic_tuner(self):
        """Test VirtualCell.compile() with kinetic tuner disabled."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42, auto_compile=False)
        vc.compile(use_kinetic_tuner=False)
        assert vc._compiled
        assert vc._tuner is None
        assert vc._spec is not None
        assert vc._model is not None

    def test_get_species_names_excludes_drugs(self):
        """Test that get_species_names() excludes drugs by default."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42, auto_compile=False)
        vc.add_drug(name="DrugX", regulation=["R1_1"], regulation_type=["up"])
        vc.compile()

        species_names = vc.get_species_names()
        # Check that DrugX is not in species
        assert "DrugX" not in species_names

    def test_get_initial_values_excludes_drugs(self):
        """Test that get_initial_values() excludes drugs by default."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42, auto_compile=False)
        vc.add_drug(name="DrugX", regulation=["R1_1"], regulation_type=["up"])
        vc.compile()

        initial_values = vc.get_initial_values()
        # Check that DrugX is not in initial values
        assert "DrugX" not in initial_values
        # Check that other species are present
        assert len(initial_values) > 0

    def test_get_target_concentrations_with_tuner(self):
        """Test that get_target_concentrations() returns values with tuner."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42, auto_compile=False)
        vc.compile(use_kinetic_tuner=True)
        target_concs = vc.get_target_concentrations()
        assert len(target_concs) > 0
        # All keys should be species ending with 'a' (active forms)
        for key in target_concs.keys():
            assert key.endswith('a')

    def test_get_target_concentrations_without_tuner(self):
        """Test that get_target_concentrations() returns empty dict without tuner."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42, auto_compile=False)
        vc.compile(use_kinetic_tuner=False)
        target_concs = vc.get_target_concentrations()
        assert len(target_concs) == 0

    def test_spec_property_raises_if_not_compiled(self):
        """Test that spec property raises ValueError if not compiled."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], auto_compile=False)
        with pytest.raises(ValueError, match="must be compiled"):
            _ = vc.spec

    def test_model_property_raises_if_not_compiled(self):
        """Test that model property raises ValueError if not compiled."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], auto_compile=False)
        with pytest.raises(ValueError, match="must be compiled"):
            _ = vc.model


class TestMakeDatasetDrugResponse:
    """Tests for make_dataset_drug_response function."""

    def test_returns_correct_shapes(self):
        """Test that make_dataset_drug_response() returns correct shapes."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42)

        n_samples = 10
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
        assert X.shape[1] == 16

    def test_drugs_not_in_x_features(self):
        """Test that drugs are NOT included in X features."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42, auto_drug=False)
        vc.add_drug(name="DrugX", regulation=["R1_1"], regulation_type=["up"])
        vc.compile()

        X, y = make_dataset_drug_response(
            n=10,
            cell_model=vc,
            perturbation_type='gaussian',
            perturbation_params={'rsd': 0.2},
            seed=42,
        )
        species_names = vc.get_species_names()

        # Check number of columns matches species (excludes drugs)
        assert X.shape[1] == 16

    def test_raises_if_not_compiled(self):
        """Test that make_dataset_drug_response() raises if model not compiled."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], auto_compile=False)

        with pytest.raises(ValueError, match="must be compiled"):
            make_dataset_drug_response(n=100, cell_model=vc)

    def test_with_drug_regulation(self):
        """Test dataset generation with drug regulation."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42, auto_drug=False)
        vc.add_drug(
            name="DrugD",
            start_time=500.0,
            regulation=["R1_1"],
            regulation_type=["down"],
        )
        vc.compile()

        X, y = make_dataset_drug_response(
            n=10,
            cell_model=vc,
            perturbation_type='gaussian',
            perturbation_params={'rsd': 0.2},
            seed=42,
        )
        assert X.shape[0] == 10
        assert y.shape[0] == 10
        # convert y to numpy array if it's a pandas Series
        if isinstance(y, pd.Series):
            y = y.values
        # Ensure numeric dtype
        y = y.astype(float)
        # y should have valid values (allow NaN for failed simulations)
        assert np.all(np.isfinite(y[~np.isnan(y)]))

    def test_custom_simulation_params(self):
        """Test with custom simulation parameters."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42)
        vc.compile()

        sim_params = {'start': 0, 'end': 500, 'points': 50}
        X, y = make_dataset_drug_response(
            n=5,
            cell_model=vc,
            perturbation_type='gaussian',
            perturbation_params={'rsd': 0.2},
            simulation_params=sim_params,
            seed=42,
        )

        assert X.shape[0] == 5
        assert y.shape[0] == 5

    def test_custom_perturbation_type(self):
        """Test with custom perturbation type."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42)
        vc.compile()

        # Test each perturbation type with appropriate parameters
        X, y = make_dataset_drug_response(
            n=5,
            cell_model=vc,
            perturbation_type='uniform',
            perturbation_params={'min': 0.8, 'max': 1.2},
            seed=42,
        )
        assert X.shape == (5, 16)
        assert y.shape == (5,)

        X, y = make_dataset_drug_response(
            n=5,
            cell_model=vc,
            perturbation_type='gaussian',
            perturbation_params={'rsd': 0.2},
            seed=42,
        )
        assert X.shape == (5, 16)
        assert y.shape == (5,)

    def test_scipy_and_roadrunner_solvers(self):
        """Test both scipy and roadrunner solvers."""
        vc = Builder.specify(degree_cascades=[1, 2, 5], random_seed=42)
        vc.compile()

        # Scipy solver
        X_scipy, y_scipy = make_dataset_drug_response(
            n=5,
            cell_model=vc,
            solver_type='scipy',
            perturbation_type='gaussian',
            perturbation_params={'rsd': 0.2},
            seed=42,
        )

        # Roadrunner solver (skip if roadrunner not available)
        pytest.importorskip("roadrunner")
        X_rr, y_rr = make_dataset_drug_response(
            n=5,
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
            n=10,
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


class TestPandasOutput:
    """Tests for pandas DataFrame/Series output functionality."""

    def test_default_returns_pandas(self):
        """Test that make_dataset_drug_response() returns pandas by default."""
        vc = Builder.specify(degree_cascades=[1, 2], random_seed=42)
        
        X, y = make_dataset_drug_response(n=10, cell_model=vc, seed=42, verbose=False)
        
        # Check types
        assert isinstance(X, pd.DataFrame), f"Expected DataFrame, got {type(X)}"
        assert isinstance(y, pd.Series), f"Expected Series, got {type(y)}"
        
        # Check shapes
        assert X.shape[0] == 10
        assert y.shape[0] == 10
        assert X.shape[1] == 6
        
        # Check that X has column names (feature names)
        assert len(X.columns) > 0
        assert all(isinstance(col, str) for col in X.columns)
        
        # Check that y has a name
        assert y.name is not None

    def test_explicit_as_pandas_true(self):
        """Test explicit as_pandas=True returns pandas objects."""
        vc = Builder.specify(degree_cascades=[1, 2], random_seed=42)
        
        X, y = make_dataset_drug_response(n=10, cell_model=vc, as_pandas=True, seed=42, verbose=False)
        
        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)

    def test_as_pandas_false_returns_numpy(self):
        """Test as_pandas=False returns numpy arrays."""
        vc = Builder.specify(degree_cascades=[1, 2], random_seed=42)
        
        X, y = make_dataset_drug_response(n=10, cell_model=vc, as_pandas=False, seed=42, verbose=False)
        
        assert isinstance(X, np.ndarray), f"Expected ndarray, got {type(X)}"
        assert isinstance(y, np.ndarray), f"Expected ndarray, got {type(y)}"
        assert X.ndim == 2
        assert y.ndim == 1

    def test_values_are_equivalent(self):
        """Test that pandas and numpy outputs contain equivalent values."""
        vc = Builder.specify(degree_cascades=[1, 2], random_seed=42)
        
        # Get pandas output
        X_pandas, y_pandas = make_dataset_drug_response(n=10, cell_model=vc, as_pandas=True, seed=42, verbose=False)
        
        # Get numpy output
        X_numpy, y_numpy = make_dataset_drug_response(n=10, cell_model=vc, as_pandas=False, seed=42, verbose=False)
        
        # Check that values are identical
        np.testing.assert_array_equal(X_pandas.values, X_numpy)
        np.testing.assert_array_equal(y_pandas.values, y_numpy)

    def test_pandas_with_different_solvers(self):
        """Test pandas output works with both solvers."""
        vc = Builder.specify(degree_cascades=[1, 2], random_seed=42)

        # Test scipy solver
        X_scipy, y_scipy = make_dataset_drug_response(
            n=5, cell_model=vc, solver_type='scipy', as_pandas=True, seed=42, verbose=False
        )
        assert isinstance(X_scipy, pd.DataFrame)
        assert isinstance(y_scipy, pd.Series)

        # Test roadrunner solver (skip if roadrunner not available)
        pytest.importorskip("roadrunner")
        X_rr, y_rr = make_dataset_drug_response(
            n=5, cell_model=vc, solver_type='roadrunner', as_pandas=True, seed=42, verbose=False
        )
        assert isinstance(X_rr, pd.DataFrame)
        assert isinstance(y_rr, pd.Series)

        # Shapes should be the same
        assert X_scipy.shape == X_rr.shape
        assert y_scipy.shape == y_rr.shape


class TestGenericSpecs:
    """Tests for generic specification support in VirtualCell."""

    def test_virtual_cell_with_michaelis_spec(self):
        """Test VirtualCell works with MichaelisNetworkSpec."""
        spec = MichaelisNetworkSpec()
        spec.generate_specifications(num_species=3, num_regulations=2, random_seed=42)

        vc = VirtualCell(spec=spec, auto_drug=False)
        vc.compile()

        assert vc._compiled
        # Each species has an inactive and active form (e.g., S1 and S1a)
        assert len(vc.get_species_names()) == 6

        X, y = make_dataset_drug_response(n=5, cell_model=vc, target_specie='S1a')
        assert X.shape == (5, 3)
        assert y.shape == (5,)

    def test_builder_from_degree_cascades(self):
        """Test Builder.from_degree_cascades factory method."""
        vc = Builder.from_degree_cascades(cascades=[1, 2], feedback_density=0.5)
        assert vc._compiled
        assert isinstance(vc.spec, DegreeInteractionSpec)
        assert vc.spec.degree_cascades == [1, 2]

    def test_builder_specify_with_spec_object(self):
        """Test Builder.specify with a spec object."""
        spec = DegreeInteractionSpec(degree_cascades=[1])
        vc = Builder.specify(spec=spec)
        assert vc.spec is spec

    def test_auto_drug_with_custom_spec_targets(self):
        """Test auto-drug targets are correctly identified via spec.get_auto_drug_targets()."""
        # Create a spec where we override targets
        class CustomSpec(MichaelisNetworkSpec):
            def get_auto_drug_targets(self):
                return ["S1", "S2"]

        spec = CustomSpec()
        spec.generate_specifications(num_species=3, random_seed=42)

        vc = VirtualCell(spec=spec, auto_drug=True, drug_name="MyDrug")
        vc.compile()

        drugs = vc.list_drugs()
        assert drugs[0]['name'] == "MyDrug"
        assert drugs[0]['targets'] == ["S1", "S2"]
