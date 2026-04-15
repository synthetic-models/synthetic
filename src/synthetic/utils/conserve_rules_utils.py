"""
Utilities for conserve_rules perturbation type.

This module provides functions to determine species hierarchy and assign
variation ranges based on biological conservation principles.
"""

import logging
from typing import Dict, Any, List, Union
import re

logger = logging.getLogger(__name__)


def resolve_species_range(
    model_spec=None,
    initial_values: Dict[str, float] = None,
    base_shape: float = 0.01,
    max_shape: float = 0.5,
    num_cascades: Union[int, List[int]] = None
) -> Dict[str, float]:
    """
    Determine species hierarchy based on naming semantics and assign variation ranges.
    
    This function analyzes species names to identify their position in the pathway
    and assigns shape parameters for lognormal distribution based on conservation rules.
    Species closer to the central pathway/output are more conserved (lower shape),
    while outer species have higher variance (higher shape).
    
    Args:
        model_spec: BaseSpec object (preferred method)
        initial_values: Dictionary of initial values as fallback
        base_shape: Minimum shape parameter (most conserved species, default: 0.01)
        max_shape: Maximum shape parameter (least conserved species, default: 0.5)
        num_cascades: Number of cascades/degrees
            - Can be an integer (for ModelSpec with single degree)
            - Can be a list (for DegreeInteractionSpec with multiple degrees)
            - If None, will be inferred from species names
    
    Returns:
        Dictionary mapping species names to their shape parameters
        e.g., {'R1_1': 0.01, 'R2_1': 0.1275, 'I1_1': 0.01, ...}
    
    Raises:
        ValueError: If neither model_spec nor initial_values is provided
        ValueError: If species cannot be parsed
    
    Examples:
        >>> # For DegreeInteractionSpec with degree_cascades=[1,1,1,1,1]
        >>> species_range = resolve_species_range(
        ...     model_spec=degree_spec,
        ...     base_shape=0.01,
        ...     max_shape=0.5,
        ...     num_cascades=5
        ... )
        >>> # Result: {'R1_1': 0.01, 'I1_1': 0.01, 'R2_1': 0.1275, 'I2_1': 0.1275, ...}
        
        >>> # For ModelSpec4 with 5 cascades
        >>> species_range = resolve_species_range(
        ...     model_spec=model_spec,
        ...     base_shape=0.01,
        ...     max_shape=0.5,
        ...     num_cascades=5
        ... )
    """
    # Get list of species
    if model_spec is not None:
        species = _get_species_from_model_spec(model_spec)
    elif initial_values is not None:
        species = list(initial_values.keys())
    else:
        raise ValueError("Either model_spec or initial_values must be provided")
    
    logger.info(f"Resolving species range for {len(species)} species")
    
    # Determine model type and calculate hierarchy
    model_type = _detect_model_type(species)
    logger.info(f"Detected model type: {model_type}")
    
    if model_type == 'DegreeInteractionSpec':
        return _resolve_degree_interaction_species(species, base_shape, max_shape, num_cascades)
    else:
        logger.warning("Unknown model type, using default hierarchy")
        return _resolve_default_hierarchy(species, base_shape, max_shape)


def _get_species_from_model_spec(model_spec) -> List[str]:
    """
    Extract list of species from model specification.

    Args:
        model_spec: BaseSpec object

    Returns:
        List of species names
    """
    # Try different attributes based on model spec version
    species = []
    
    # Check for degree_species (DegreeInteractionSpec)
    if hasattr(model_spec, 'degree_species'):
        for degree_dict in model_spec.degree_species.values():
            species.extend(degree_dict['R'])
            species.extend(degree_dict['I'])
    
    # Check for receptors and intermediate_layers (ModelSpec4)
    if hasattr(model_spec, 'receptors'):
        species.extend(model_spec.receptors)
    if hasattr(model_spec, 'intermediate_layers'):
        for layer in model_spec.intermediate_layers:
            species.extend(layer)
    
    # Check for species_list (BaseSpec)
    if hasattr(model_spec, 'species_list'):
        species = model_spec.species_list
    
    # Remove duplicates while preserving order
    seen = set()
    unique_species = []
    for s in species:
        if s not in seen:
            seen.add(s)
            unique_species.append(s)
    
    return unique_species


def _detect_model_type(species: List[str]) -> str:
    """
    Detect the type of model specification based on species naming patterns.
    
    Args:
        species: List of species names
    
    Returns:
        Model type string: 'DegreeInteractionSpec', 'ModelSpec4', or 'unknown'
    """
    if not species:
        return 'unknown'
    
    # Check for DegreeInteractionSpec pattern: R{degree}_{index}, I{degree}_{index}
    degree_pattern = re.compile(r'^[RI]\d+_\d+$')
    degree_match_count = sum(1 for s in species if degree_pattern.match(s))
    
    if degree_match_count / len(species) > 0.5:
        return 'DegreeInteractionSpec'
    
    # Check for ModelSpec4 pattern: R{index}, I{layer}_{index}
    model4_r_pattern = re.compile(r'^R\d+$')
    model4_i_pattern = re.compile(r'^I\d+_\d+$')
    model4_r_count = sum(1 for s in species if model4_r_pattern.match(s))
    model4_i_count = sum(1 for s in species if model4_i_pattern.match(s))
    
    if (model4_r_count + model4_i_count) / len(species) > 0.5:
        return 'ModelSpec4'
    
    return 'unknown'


def _parse_species_name(species_name: str) -> Dict[str, Any]:
    """
    Parse a species name to extract its components.
    
    Args:
        species_name: Species name (e.g., 'R1_1', 'I2_3', 'R5')
    
    Returns:
        Dictionary with parsed components:
        - 'type': 'R' or 'I' or 'O'
        - 'degree': degree number (for DegreeInteractionSpec)
        - 'index': index within degree (for DegreeInteractionSpec)
        - 'layer': layer number (for ModelSpec4)
        - 'cascade': cascade number (for ModelSpec4)
    
    Raises:
        ValueError: If species name cannot be parsed
    """
    # Try DegreeInteractionSpec pattern: R{degree}_{index} or I{degree}_{index}
    di_pattern = re.match(r'^([RI])(\d+)_(\d+)$', species_name)
    if di_pattern:
        return {
            'type': di_pattern.group(1),
            'degree': int(di_pattern.group(2)),
            'index': int(di_pattern.group(3)),
            'is_degree_interaction': True
        }
    
    # Try ModelSpec4 pattern: R{cascade} or I{layer}_{cascade}
    ms4_r_pattern = re.match(r'^R(\d+)$', species_name)
    if ms4_r_pattern:
        return {
            'type': 'R',
            'cascade': int(ms4_r_pattern.group(1)),
            'is_modelspec4': True
        }
    
    ms4_i_pattern = re.match(r'^I(\d+)_(\d+)$', species_name)
    if ms4_i_pattern:
        return {
            'type': 'I',
            'layer': int(ms4_i_pattern.group(1)),
            'cascade': int(ms4_i_pattern.group(2)),
            'is_modelspec4': True
        }
    
    # Outcome species
    if species_name == 'O' or species_name.startswith('O'):
        return {
            'type': 'O',
            'is_outcome': True
        }
    
    raise ValueError(f"Cannot parse species name: {species_name}")


def _resolve_degree_interaction_species(
    species: List[str],
    base_shape: float,
    max_shape: float,
    num_cascades: Union[int, List[int]] = None
) -> Dict[str, float]:
    """
    Resolve species ranges for DegreeInteractionSpec.

    Hierarchy: Degree 1 (most conserved) -> Degree 2 -> ... -> Degree N (least conserved)

    Notes:
        - 'D' is treated as a special-case input (e.g., dose) and is not parsed here.
        - Outcome species ('O', 'O*') are skipped.

    Args:
        species: List of species names
        base_shape: Minimum shape parameter
        max_shape: Maximum shape parameter
        num_cascades: List of cascade counts per degree or None to auto-detect

    Returns:
        Dictionary mapping species to shape parameters
    """
    species_range: Dict[str, float] = {}

    # Auto-detect degrees if not provided (ignore special-case species)
    if num_cascades is None:
        degrees = set()
        for s in species:
            if s == "D" or s == "O" or s.startswith("O"):
                continue
            try:
                parsed = _parse_species_name(s)
                if "degree" in parsed:
                    degrees.add(parsed["degree"])
            except ValueError:
                # Ignore non-standard names during degree detection
                continue

        num_degrees = max(degrees) if degrees else 1
        logger.info(f"Auto-detected {num_degrees} degrees")
    elif isinstance(num_cascades, list):
        num_degrees = len(num_cascades)
        logger.info(f"Using provided {num_degrees} degrees")
    else:
        num_degrees = num_cascades
        logger.info(f"Using provided {num_degrees} degrees")

    # Calculate shape for each degree
    for s in species:
        # Special cases that should not be parsed/perturbed here
        if s == "D" or s == "O" or s.startswith("O"):
            continue

        try:
            parsed = _parse_species_name(s)

            if "degree" in parsed:
                degree = parsed["degree"]
                # Linear interpolation from base_shape to max_shape
                # Degree 1: base_shape, Degree N: max_shape
                if num_degrees > 1:
                    shape = base_shape + (max_shape - base_shape) * (degree - 1) / (num_degrees - 1)
                else:
                    shape = base_shape
                species_range[s] = shape

        except ValueError as e:
            # Non-standard species name: assign a safe default without noisy warnings
            logger.debug(f"Could not parse species {s}: {e}")
            species_range[s] = (base_shape + max_shape) / 2

    logger.info(f"Generated species_range with {len(species_range)} entries")
    return species_range


def _resolve_default_hierarchy(
    species: List[str],
    base_shape: float,
    max_shape: float
) -> Dict[str, float]:
    """
    Default hierarchy resolution for unknown model types.
    
    Assigns increasing shape parameters alphabetically.
    
    Args:
        species: List of species names
        base_shape: Minimum shape parameter
        max_shape: Maximum shape parameter
    
    Returns:
        Dictionary mapping species to shape parameters
    """
    species_range = {}
    
    # Sort species alphabetically and assign increasing shape
    sorted_species = sorted(s for s in species if s != 'O')
    
    for i, s in enumerate(sorted_species):
        if len(sorted_species) > 1:
            shape = base_shape + (max_shape - base_shape) * i / (len(sorted_species) - 1)
        else:
            shape = (base_shape + max_shape) / 2
        species_range[s] = shape
    
    logger.warning(f"Using default hierarchy for {len(species_range)} species")
    return species_range