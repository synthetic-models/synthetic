# API Reference

Auto-generated API documentation from source code docstrings.

## High-Level API

The `Builder` and `VirtualCell` classes provide the simplest interface for creating and simulating virtual cell models. See [Quick Start](quick_start.md) for usage examples.

::: synthetic.api.Builder
    options:
      show_root_heading: true

::: synthetic.api.VirtualCell
    options:
      show_root_heading: true
      members:
        - compile
        - add_drug
        - list_drugs
        - get_species_names
        - get_initial_values
        - get_target_concentrations

::: synthetic.api.make_dataset_drug_response
    options:
      show_root_heading: true

## Model Building

Core classes for building biochemical reaction networks from scratch. See [Model Building](model_building.md) for the full guide.

::: synthetic.ModelBuilder.ModelBuilder
    options:
      show_root_heading: true
      members:
        - add_reaction
        - delete_reaction
        - precompile
        - get_parameters
        - get_state_variables
        - set_parameter
        - get_parameter
        - set_state
        - get_state
        - get_regulator_parameter_map
        - get_parameter_regulator_map
        - get_antimony_model
        - get_sbml_model
        - add_simple_piecewise
        - combine
        - copy
        - head

::: synthetic.Reaction.Reaction
    options:
      show_root_heading: true

::: synthetic.ReactionArchtype.ReactionArchtype
    options:
      show_root_heading: true

## Specifications

Classes for defining network topology, regulations, and drug interactions. See [Network & Drug Design](network_and_drug_design.md) for details.

::: synthetic.Specs.BaseSpec.BaseSpec
    options:
      show_root_heading: true

::: synthetic.Specs.DegreeInteractionSpec.DegreeInteractionSpec
    options:
      show_root_heading: true
      members:
        - generate_specifications
        - add_drug
        - generate_network
        - get_species_by_degree
        - get_regulations_by_degree

::: synthetic.Specs.MichaelisNetworkSpec.MichaelisNetworkSpec
    options:
      show_root_heading: true

::: synthetic.Specs.Drug.Drug
    options:
      show_root_heading: true

::: synthetic.Specs.Regulation.Regulation
    options:
      show_root_heading: true

## Solvers

ODE simulation backends. Both concrete solvers inherit from the abstract `Solver` base class, which defines the `compile()` and `simulate()` interface. See [Solvers & Simulation](solvers_and_simulation.md) for usage.

::: synthetic.Solver.ScipySolver.ScipySolver
    options:
      show_root_heading: true
      members:
        - compile
        - simulate
        - set_state_values
        - set_parameter_values

::: synthetic.Solver.RoadrunnerSolver.RoadrunnerSolver
    options:
      show_root_heading: true
      members:
        - compile
        - simulate
        - set_state_values
        - set_parameter_values

## Utilities

Helpers for parameter tuning, feature generation, and dataset creation. See [Data Generation](data_generation.md), [Advanced Features](advanced_features.md), and [Benchmarking](benchmarking.md) for context.

::: synthetic.utils.kinetic_tuner.KineticParameterTuner
    options:
      show_root_heading: true
      members:
        - generate_parameters
        - get_target_concentrations

::: synthetic.utils.make_feature_data.make_feature_data
    options:
      show_root_heading: true

::: synthetic.utils.make_target_data.make_target_data_with_params
    options:
      show_root_heading: true

::: synthetic.utils.make_target_data.calculate_targets_from_timecourse
    options:
      show_root_heading: true
