# Future Refactor Plan: Decoupling Top-Level API from Specific Specs (COMPLETED)

## Current Status
This refactor has been successfully completed as of 2026-04-16.

## Completed Objectives
1.  **Pure Interface:** `Builder.specify` and `VirtualCell` are now spec-agnostic.
2.  **Explicit Dependency:** Users can pass a configured `BaseSpec` or `ModelBuilder` object.
3.  **Hinge on Solver:** `make_dataset_drug_response` now supports `VirtualCell`, `ModelBuilder`, and `Solver` instances directly.

## Implemented Steps

### 1. Refactored `VirtualCell` Constructor
- Positional/keyword arguments now favor a generic `spec` parameter.
- `degree_cascades` and `feedback_density` are handled via `**kwargs` for backward compatibility.

### 2. Generalized `Builder.specify`
- Signature updated to: `Builder.specify(spec: Union[BaseSpec, ModelBuilder], **kwargs)`.
- Added `Builder.from_degree_cascades()` as a factory method for the hierarchical topology workflow.

### 3. Moved Spec-Specific Logic into Specs
- Auto-drug target identification logic moved to `BaseSpec.get_auto_drug_targets()`.
- Added `BaseSpec.get_outcome_species()` and `BaseSpec.is_activated_form()` to allow specifications to define their own naming conventions and analysis targets.

### 4. Transitioned toward a "Model-as-Solver" Pattern
- `make_dataset_drug_response` now extracts defaults (states, parameters) from compiled `Solver` instances.
- VirtualCell now creates and manages its own internal `Solver` instance.

## Benefits Achieved
- **Extensibility:** New network types can be added by inheriting from `BaseSpec` without touching the core API.
- **Clarity:** Separated the topological "Spec" from the physical "Model" and the executable "Solver".
- **Interoperability:** `make_dataset_drug_response` is now more flexible, allowing users to provide different model representations depending on their stage in the workflow.
