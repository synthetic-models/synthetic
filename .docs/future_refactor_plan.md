# Future Refactor Plan: Decoupling Top-Level API from Specific Specs

## Current Issue
The `Builder.specify` and `VirtualCell` constructors still contain parameters that are specific to the `DegreeInteractionSpec` (e.g., `degree_cascades`, `feedback_density`). While the internal implementation now handles generic `BaseSpec` instances, the public interface still signals a tight coupling to a single network topology type.

## Objectives
1.  **Pure Interface:** Make `Builder.specify` and `VirtualCell` truly spec-agnostic.
2.  **Explicit Dependency:** Require users to pass a fully configured `Spec` instance or use a cleaner configuration pattern.
3.  **Hinge on Solver:** Solidify the `Solver` as the primary interface for "executable models" during data generation.

## Proposed Steps

### 1. Refactor `VirtualCell` Constructor
- Move `degree_cascades` and `feedback_density` out of the positional/keyword arguments.
- Only accept a `Spec` object or a `ModelBuilder` object.
- Any spec-specific configuration should happen *before* passing the spec to `VirtualCell`.

### 2. Generalize `Builder.specify`
- Change signature to: `Builder.specify(spec: Union[BaseSpec, ModelBuilder], **kwargs)`.
- Use `kwargs` only for metadata that applies to *all* models (e.g., `name`, `random_seed`).
- Provide a convenience helper for the legacy degree-cascade workflow if necessary, but move it to a separate factory method like `Builder.from_degree_cascades(...)`.

### 3. Move Spec-Specific Logic into Specs
- The `_generate_auto_drug` logic in `VirtualCell` is currently tied to the naming convention of `DegreeInteractionSpec`. This logic should be moved into the `DegreeInteractionSpec` class itself or into a specialized utility.

### 4. Transition toward a "Model-as-Solver" Pattern
- Encourage the use of `make_dataset_drug_response(cell_model=solver, ...)` as the primary entry point.
- Ensure that all required metadata (drugs, targets, ranges) can be carried by the `Solver` or an associated `ModelMetadata` object.

## Benefits
- **Extensibility:** Easily add new network types (e.g., `BooleanNetworkSpec`, `ReactionFluxSpec`) without changing the `api.py` core.
- **Clarity:** Users who aren't using hierarchical cascades won't be confused by `degree_cascades` parameters.
- **Modular Testing:** Specs can be tested in isolation from the `VirtualCell` facade.
