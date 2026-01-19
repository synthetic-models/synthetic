"""
Parameter mapping utilities for ModelBuilder objects.
Provides functions to analyze parameter-reaction relationships.
"""

from typing import Dict, List, Optional, Tuple, Any
import re
from ..ModelBuilder import ModelBuilder


def get_parameter_reaction_map(model_builder: ModelBuilder) -> Dict[str, Dict]:
    """
    Map each parameter to its associated reaction details.
    
    Args:
        model_builder: The ModelBuilder object to analyze
        
    Returns:
        Dictionary mapping parameter names to reaction details:
        {
            'Km_J0': {
                'reaction_index': 0,
                'parameter_type': 'Km',
                'reactants': ['O'],
                'products': ['Oa'],
                'reaction_description': 'O -> Oa'
            },
            ...
        }
    """
    if not model_builder.pre_compiled:
        raise ValueError("Model must be pre-compiled before parameter mapping")
    
    param_map = {}
    all_params = model_builder.get_parameters()
    
    for param_name in all_params.keys():
        # Parse parameter name: {type}_{index}
        match = re.match(r'^([a-zA-Z][a-zA-Z0-9]*)_(J\d+)$', param_name)
        if not match:
            # Try alternative pattern for linked parameters or custom names
            match = re.match(r'^([a-zA-Z][a-zA-Z0-9]*)_(.+)$', param_name)
            if not match:
                continue
        
        param_type = match.group(1)
        reaction_id = match.group(2)
        
        # Extract reaction index from J0, J1, etc.
        reaction_idx = None
        if reaction_id.startswith('J'):
            try:
                reaction_idx = int(reaction_id[1:])
            except ValueError:
                reaction_idx = None
        
        reaction_info = {
            'parameter_name': param_name,
            'parameter_type': param_type,
            'reaction_id': reaction_id,
            'reaction_index': reaction_idx,
            'reactants': [],
            'products': [],
            'reaction_description': 'Unknown'
        }
        
        # Try to get reaction details if we have an index
        if reaction_idx is not None and reaction_idx < len(model_builder.reactions):
            reaction = model_builder.reactions[reaction_idx]
            reaction_info['reactants'] = list(reaction.reactants_names)
            reaction_info['products'] = list(reaction.products_names)
            reaction_info['reaction_description'] = (
                f"{' + '.join(reaction.reactants_names)} -> "
                f"{' + '.join(reaction.products_names)}"
            )
            
            # Try to get archtype information if available
            try:
                reaction_info['parameter_roles'] = list(reaction.archtype.parameters)
            except AttributeError:
                reaction_info['parameter_roles'] = []
        
        param_map[param_name] = reaction_info
    
    return param_map


def find_parameter_by_role(
    model_builder: ModelBuilder, 
    role: str, 
    state_var: Optional[str] = None
) -> List[str]:
    """
    Find parameters that match a specific role.
    
    Args:
        model_builder: The ModelBuilder object to analyze
        role: Parameter role to search for (e.g., 'Vmax', 'Km', 'Kc')
               Use None to match all parameters (filtered by state_var only)
        state_var: Optional state variable to filter by (e.g., 'O', 'Oa')
                    Checks if state_var is involved in the reaction
        
    Returns:
        List of parameter names matching the criteria
    """
    if not model_builder.pre_compiled:
        raise ValueError("Model must be pre-compiled before parameter search")
    
    param_map = get_parameter_reaction_map(model_builder)
    matching_params = []
    
    for param_name, param_info in param_map.items():
        # Filter by role if specified
        if role is not None:
            param_type = param_info.get('parameter_type', '')
            # Case-insensitive partial matching
            if role.lower() not in param_type.lower():
                continue
        
        # Filter by state variable if specified
        if state_var is not None:
            reactants = param_info.get('reactants', [])
            products = param_info.get('products', [])
            
            # Check exact match
            is_involved = (state_var in reactants) or (state_var in products)
            
            # Check for activated/deactivated forms
            if not is_involved:
                # Handle X vs Xa patterns
                if state_var.endswith('a'):
                    base_var = state_var[:-1]  # Remove 'a'
                    is_involved = (base_var in reactants) or (base_var in products)
                else:
                    activated_var = state_var + 'a'
                    is_involved = (activated_var in reactants) or (activated_var in products)
            
            if not is_involved:
                continue
        
        matching_params.append(param_name)
    
    return matching_params


def explain_reaction_parameters(
    model_builder: ModelBuilder, 
    reaction_index: int
) -> str:
    """
    Generate human-readable explanation of all parameters in a reaction.
    
    Args:
        model_builder: The ModelBuilder object
        reaction_index: Index of the reaction to explain (0-based)
        
    Returns:
        Human-readable explanation string
    """
    if not model_builder.pre_compiled:
        raise ValueError("Model must be pre-compiled")
    
    if reaction_index >= len(model_builder.reactions):
        raise IndexError(f"Reaction index {reaction_index} out of bounds")
    
    reaction = model_builder.reactions[reaction_index]
    param_map = get_parameter_reaction_map(model_builder)
    
    # Find parameters for this reaction
    reaction_params = []
    for param_name, param_info in param_map.items():
        if param_info.get('reaction_index') == reaction_index:
            reaction_params.append((param_name, param_info))
    
    if not reaction_params:
        return f"Reaction {reaction_index} has no directly associated parameters"
    
    # Build explanation
    reactants = reaction.reactants_names
    products = reaction.products_names
    explanation = [f"Reaction {reaction_index}: {', '.join(reactants)} → {', '.join(products)}"]
    explanation.append("Parameters:")
    
    for param_name, param_info in reaction_params:
        param_type = param_info.get('parameter_type', 'Unknown')
        role_desc = _describe_parameter_role(param_type, reactants, products)
        explanation.append(f"  - {param_name}: {role_desc}")
    
    return "\n".join(explanation)


def _describe_parameter_role(
    param_type: str, 
    reactants: Tuple[str, ...], 
    products: Tuple[str, ...]
) -> str:
    """
    Generate human-readable description of a parameter's role.
    
    Args:
        param_type: Parameter type (e.g., 'Vmax', 'Km', 'Kc')
        reactants: Reactant names
        products: Product names
        
    Returns:
        Description string
    """
    param_type_lower = param_type.lower()
    
    # Common parameter roles based on naming patterns
    if 'vmax' in param_type_lower:
        if reactants and products:
            return f"Maximum rate for conversion of {reactants[0]} to {products[0]}"
        return "Maximum reaction rate"
    
    elif 'km' in param_type_lower:
        if reactants:
            return f"Michaelis constant for substrate {reactants[0]}"
        return "Michaelis constant (substrate affinity)"
    
    elif 'kc' in param_type_lower:
        return "Constitutive/basal rate constant"
    
    elif param_type_lower.startswith('ka'):
        return "Allosteric activation constant"
    
    elif param_type_lower.startswith('ki'):
        if 'c' in param_type_lower:
            return "Competitive inhibition constant"
        return "Allosteric inhibition constant"
    
    elif param_type_lower.startswith('k'):
        return "Rate constant"
    
    else:
        return f"Parameter of type {param_type}"


def get_parameters_for_state(
    model_builder: ModelBuilder, 
    state_var: str
) -> Dict[str, List[str]]:
    """
    Get all parameters affecting a specific state variable.
    
    Args:
        model_builder: The ModelBuilder object
        state_var: State variable name (e.g., 'O', 'R1a')
        
    Returns:
        Dictionary with lists of parameters:
        {
            'as_reactant': ['Km_J0', 'Vmax_J0'],  # Parameters in reactions where state is reactant
            'as_product': ['Kc_J1'],              # Parameters in reactions where state is product
            'all': ['Km_J0', 'Vmax_J0', 'Kc_J1']  # All parameters affecting state
        }
        
    Raises:
        ValueError: If state_var is not found in the model
    """
    # Check if state exists in the model
    all_states = model_builder.get_state_variables()
    if state_var not in all_states:
        # Check for activated/deactivated forms
        if state_var.endswith('a'):
            base_var = state_var[:-1]
            if base_var not in all_states:
                raise ValueError(f"State variable '{state_var}' not found in the model")
        else:
            activated_var = state_var + 'a'
            if activated_var not in all_states:
                raise ValueError(f"State variable '{state_var}' not found in the model")
    
    param_map = get_parameter_reaction_map(model_builder)
    
    as_reactant = []
    as_product = []
    
    for param_name, param_info in param_map.items():
        reactants = param_info.get('reactants', [])
        products = param_info.get('products', [])
        
        if state_var in reactants:
            as_reactant.append(param_name)
        elif state_var in products:
            as_product.append(param_name)
        else:
            # Check for activated/deactivated forms
            if state_var.endswith('a'):
                base_var = state_var[:-1]
                if base_var in reactants or base_var in products:
                    as_reactant.append(param_name)
            else:
                activated_var = state_var + 'a'
                if activated_var in reactants or activated_var in products:
                    as_product.append(param_name)
    
    all_params = list(set(as_reactant + as_product))
    
    return {
        'as_reactant': as_reactant,
        'as_product': as_product,
        'all': all_params
    }
