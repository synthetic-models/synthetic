import re
from typing import Dict, Union, Tuple
from copy import copy
from .ReactionArchtype import ReactionArchtype
from .LinkedParameters import LinkedParameters

class Reaction: 
    '''
    This class represents a reaction in the system.
        ENSURE: use (a,) instead of (a) for tuples typing
        reaction_archtype: ReactionArchtype, 
        reactants: Tuple[str], 
        products: Tuple[str], 
        extra_states: Tuple[str] = (),
        parameters_values: Union[dict, tuple] = (),
        reactant_values: Union[dict, tuple] = (),
        product_values: Union[dict, tuple] = ()
                
    NOTE: This class performs two types of matching:
        1. Index-based matching.
        2. Name-based matching < - far more reliable but difficult to implement
        
    '''
    def __init__(self, 
        reaction_archtype: ReactionArchtype, 
        reactants: Tuple[str], 
        products: Tuple[str], 
        reaction_name: str = '',
        extra_states: Tuple[str] = (),
        parameters_values: Union[dict, tuple, int, float] = (),
        reactant_values: Union[dict, tuple, int, float] = (),
        product_values: Union[dict, tuple, int, float] = (),
        linked_parameters: Tuple[LinkedParameters] = (),
        use_parameter_from_reaction: str = '',
        zero_init: bool = True):

        self.archtype = reaction_archtype
        # reactants, products and extra states must be provided in the length of the archtype
        
        assert len(reactants) == reaction_archtype.reactants_count, f'length of reactants must be equal to the number of reactants in the reaction archtype, {len(reactants)} != {reaction_archtype.reactants_count}'
        assert len(products) == reaction_archtype.products_count, f'length of products must be equal to the number of products in the reaction archtype, {len(products)} != {reaction_archtype.products_count}'
        assert len(extra_states) == reaction_archtype.extra_states_count, f'length of extra_states must be equal to the number of extra_states in the reaction archtype, {len(extra_states)} != {reaction_archtype.extra_states_count}'

        self.name = reaction_name
        self.parameter_r_index = use_parameter_from_reaction

        # must specify reactant, product and extra state names if given in rate law
        self.reactants_names = reactants    
        self.products_names = products
        self.extra_states = extra_states

        # need to map reactant, product and extra state names to archtype names in 
        # the forms of dict 
        self.reactant_names_to_archtype_names = self._direct_tuples_to_dict(reactants, reaction_archtype.reactants)
        self.product_names_to_archtype_names = self._direct_tuples_to_dict(products, reaction_archtype.products)
        self.extra_states_names_to_archtype_names = self._direct_tuples_to_dict(extra_states, reaction_archtype.extra_states)

        # Compute regulator-parameter mapping
        self.regulator_parameters, self.parameter_regulators = self._compute_regulator_parameter_mapping()

        if zero_init: 
            reactant_values = {i: 0 for i in self.reactants_names}
            product_values = {i: 0 for i in self.products_names}

        # override values if provided
        self.parameters_values = self._unify_value_types_to_dict(parameters_values, self.archtype.parameters)
        self.reactant_values = self._unify_value_types_to_dict(reactant_values, reactants)
        self.product_values = self._unify_value_types_to_dict(product_values, products)

        if isinstance(parameters_values, dict):
            assert self._exist_in_archtype(parameters_values, self.archtype.parameters), 'parameters_values supplied in dict format must match the parameter names in the archtype'
        
        if isinstance(reactant_values, dict):
            assert self._dict_vals_exist_in_tuple(reactant_values, reactants), 'reactant_values supplied in dict format must match the reactant names in the reaction'

        if isinstance(product_values, dict):
            assert self._dict_vals_exist_in_tuple(product_values, products), 'product_values supplied in dict format must match the product names in the reaction'

        # checking assumed values existence for validation logic on reaction values assignment 

        assert len(self.parameters_values) == self.archtype.parameters_count or len(self.archtype.assume_parameters_values) > 0, f'Since, archtype do not have assumed parameters, length of parameters_values must be equal to the number of parameters in the reaction archtype, {len(parameters_values)} != {len(self.archtype.parameters)}'
        assert len(self.reactant_values) == self.archtype.reactants_count or len(self.archtype.assume_reactant_values) > 0, f'Since, archtype do not have assumed reactant values, length of reactant_values must be equal to the number of reactants in the reaction, {len(reactant_values)} != {len(reactants)}'
        assert len(self.product_values) == self.archtype.products_count or len(self.archtype.assume_product_values) > 0, f'Since, archtype do not have assumed product values, length of product_values must be equal to the number of products in the reaction, {len(product_values)} != {len(products)}'

        # reversibility unchangable in Reaction, but only in Archtype
        self.reversible = self.archtype.reversible

        # linked parameters

        self.linked_parameters = linked_parameters
        if self.exists_linked_parameters():
            assert len(self.linked_parameters) == self.archtype.parameters_count, f'length of linked_parameters must be equal to the number of parameters in the reaction archtype, {len(linked_parameters)} != {self.archtype.parameters_count}'

    def exists_linked_parameters(self) -> bool:
        return len(self.linked_parameters) > 0

    def _unify_value_types_to_dict(self, values: Union[dict, tuple, int, float], names: Tuple[str]) -> Dict[str, float]:
        '''
        unifies the value types to dict if not already in dict format
        '''
        if isinstance(values, dict):
            return values
        elif isinstance(values, tuple):
            return {names[i]: values[i] for i in range(len(values))}
        else:
            return {names[0]: values}

    def _dict_vals_exist_in_tuple(self, dict_vals: Dict[str, str], tuple_vals: Tuple[str]) -> bool:
        '''
        checks if all the values in the dict exist in the tuple
        '''
        for val in dict_vals.keys():
            if val not in tuple_vals:
                return False
        return True

    def _exist_in_archtype(self, reaction_dict: Dict[str, str], archtype_dict: Dict[str, str]) -> bool:
        '''
        checks if all the keys in the reaction dict exist in the archtype dict
        '''
        for key in reaction_dict:
            if key not in archtype_dict:
                return False
        return True

    def _direct_tuples_to_dict(self, reaction_tuple, archtype_tuple) -> Dict:
        '''
        maps a tuple of reaction-based names to the archtype-based names
        '''
        return {reaction_tuple[i]: archtype_tuple[i] for i in range(len(archtype_tuple))}

    def _compute_regulator_parameter_mapping(self):
        '''
        Compute mapping between regulators (extra_states) and parameters.
        Returns two dictionaries:
            regulator_parameters: mapping actual extra_state name -> list of parameter base names
            parameter_regulators: mapping parameter base name -> extra_state name
        '''
        regulator_parameters = {}
        parameter_regulators = {}
        # Prefix mapping based on ArchtypeCollections naming conventions
        # Map parameter prefix to possible extra_state prefix letters (without leading &/?)
        # Some parameter prefixes may map to multiple possibilities; we'll match the first that exists.
        prefix_map = {
            'Ka': ['A'],   # stimulators
            'Kc': ['W'],   # stimulator_weak
            'Ki': ['L', 'I'],   # allosteric_inhibitors (L) or mass action allo_inhibitors (I)
            'Kic': ['I'],  # competitive_inhibitors
            'Ks': ['A'],   # synthesis stimulators
            'Kw': ['W'],   # mass action additive stimulators
            'Kir': ['I'],  # reverse allosteric inhibitors
            'Kcr': ['C'],  # reverse competitive inhibitors
            'Kwr': ['W'],  # reverse additive stimulators
            'Ksr': ['A'],  # reverse stimulators
        }
        import re
        for param in self.archtype.parameters:
            # Extract prefix and optional numeric suffix
            match = re.match(r'([A-Za-z]+)(\d*)', param)
            if not match:
                continue
            prefix = match.group(1)
            suffix = match.group(2) if match.group(2) else ''
            if prefix in prefix_map:
                expected_prefix_list = prefix_map[prefix]
                # Find extra_state with matching suffix and prefix
                for arch_extra in self.archtype.extra_states:
                    es_match = re.match(r'([&?])([A-Za-z]+)(\d*)', arch_extra)
                    if es_match:
                        es_prefix = es_match.group(2)
                        es_suffix = es_match.group(3)
                        if es_suffix == suffix and es_prefix in expected_prefix_list:
                            # Map to actual extra_state name using extra_states_names_to_archtype_names inverse
                            actual_extra = None
                            for actual, arch in self.extra_states_names_to_archtype_names.items():
                                if arch == arch_extra:
                                    actual_extra = actual
                                    break
                            if actual_extra is not None:
                                regulator_parameters.setdefault(actual_extra, []).append(param)
                                parameter_regulators[param] = actual_extra
                            break
        return regulator_parameters, parameter_regulators

    def get_parameters_for_regulator(self, regulator_name: str) -> list:
        '''
        Returns list of parameter base names associated with the given regulator.
        '''
        return self.regulator_parameters.get(regulator_name, [])

    def get_regulator_for_parameter(self, param_name: str) -> str:
        '''
        Returns the regulator name associated with the given parameter base name.
        Returns empty string if no association.
        '''
        return self.parameter_regulators.get(param_name, '')

    def get_reaction_parameters(self, r_index) -> Dict[str, float]:
        '''
        returns a dictionary of the parameters in the reaction
        '''


        parameters = {}
        # linked parameters bypass the archtype parameters and are directly assigned
        # they will void r_index assignment completely
        if self.exists_linked_parameters():
            for i in range(len(self.linked_parameters)):
                parameters[str(self.linked_parameters[i])] = self.linked_parameters[i].get_value()
            return parameters

        if self.name != '':
            r_index = self.name
        
        # if parameter_r_index is not None, then the parameter values are assigned to the reaction
        # priority is above self.name assignment
        if self.parameter_r_index != '':
            r_index = self.parameter_r_index

        if len(self.archtype.assume_parameters_values) > 0: 
            # if the reaction archtype has parameters_values specified, use those
            # to create a dictionary 
            parameters = {f"{p}_{r_index}": 0 for p in self.archtype.parameters}
            for key, val in self.archtype.assume_parameters_values.items():
                parameters[f"{key}_{r_index}"] = val
            
        # override parameters if provided
        if len(self.parameters_values) > 0: 
            # if the specific reaction has parameters_values specified, 
            # it will override the default values
            if isinstance(self.parameters_values, dict):
                for key, val in self.parameters_values.items():
                    parameters[f"{key}_{r_index}"] = val
            
            else:
                raise ValueError('parameters_values must be a dictionary or tuple')

        return parameters

    def get_reaction_states(self) -> Dict[str, float]:

        '''
        Extracts state variables and their values from all reactions
        in the class and returns a dict, where state variables are 
        the reactants and products

        non-unique state variables will only be repeated once, their 
        default value will only follow the first repeated state variable
        '''

        states = {}
        for reactant in self.reactants_names:
            states[reactant] = self.archtype.assume_reactant_values[self.reactant_names_to_archtype_names[reactant]]
        for product in self.products_names:
            states[product] = self.archtype.assume_product_values[self.product_names_to_archtype_names[product]]

        # override values if provided
        if len(self.reactant_values) > 0:
            if isinstance(self.reactant_values, dict):
                for key, val in self.reactant_values.items():
                    states[key] = val
            else:
                raise ValueError('reactant_values must be a dictionary or tuple')

        if len(self.product_values) > 0:
            if isinstance(self.product_values, dict):
                for key, val in self.product_values.items():
                    states[key] = val 
            else:
                raise ValueError('product_values must be a dictionary or tuple')

        return states
        


    def get_antimony_reaction_str(self, r_index: str) -> str:
        '''
        generates an antimony string for the reaction, given the index of the reaction 
            r_index: str, represents reaction name in the system, usually an simple index 
        
        '''
        if self.name != '':
            r_index = self.name

        reactant_str = ' + '.join(self.reactants_names)
        product_str = ' + '.join(self.products_names)
        rate_law_str = self.archtype.rate_law 
        # rate law substitution needs to occur for reactants, products, extra states and parameters
        i = 0 
        while i < len(self.reactants_names):
            archtype_name = self.archtype.reactants[i]
            replacement_name = self.reactants_names[i]
            # Use regular expression to replace whole words only
            pattern = r'(?<!\w)' + re.escape(archtype_name) + r'(?!\w)'
            rate_law_str = re.sub(pattern, replacement_name, rate_law_str)
            i += 1 
            
        i = 0
        while i < len(self.products_names):
            archtype_name = self.archtype.products[i]
            replacement_name = self.products_names[i]
            # Use regular expression to replace whole words only
            pattern = r'(?<!\w)' + re.escape(archtype_name) + r'(?!\w)'
            rate_law_str = re.sub(pattern, replacement_name, rate_law_str)
            i += 1

        i = 0
        while i < len(self.archtype.extra_states):
            archtype_name = self.archtype.extra_states[i]
            replacement_name = self.extra_states[i]
            # Use regular expression to replace whole words only
            pattern = r'(?<!\w)' + re.escape(archtype_name) + r'(?!\w)'
            rate_law_str = re.sub(pattern, replacement_name, rate_law_str)
            i += 1


        i = 0
        while i < len(self.archtype.extra_states):
            archtype_name = self.archtype.extra_states[i]
            replacement_name = self.extra_states[i]
            # Use regular expression to replace whole words only
            pattern = r'(?<!\w)' + re.escape(archtype_name) + r'(?!\w)'
            rate_law_str = re.sub(pattern, replacement_name, rate_law_str)
            i += 1

        r_index_p = r_index 
        if self.parameter_r_index != '':
            r_index_p = self.parameter_r_index

        i = 0
        while i < len(self.archtype.parameters):
            archtype_name = self.archtype.parameters[i]
            if self.exists_linked_parameters():
                replacement_name = str(self.linked_parameters[i])
            else: 
                replacement_name = archtype_name + '_' + r_index_p
            # Use regular expression to replace whole words only
            pattern = r'(?<!\w)' + re.escape(archtype_name) + r'(?!\w)'
            rate_law_str = re.sub(pattern, replacement_name, rate_law_str)
            i += 1

        return f'{r_index}: {reactant_str} -> {product_str}; {rate_law_str}'

    def get_antimony_reactions_reverse_str(self, r_index: str) -> str:
        '''
        generates an antimony string for the reverse reaction, given the index of the reaction 
            r_index: str, represents reaction name in the system, usually an simple index 
        
        '''
        if self.name != '':
            r_index = self.name

        reactant_str = ' + '.join(self.reactants_names)
        product_str = ' + '.join(self.products_names)
        rate_law_str = self.archtype.reverse_rate_law 
        # rate law substitution needs to occur for reactants, products, extra states and parameters
        i = 0 
        while i < len(self.reactants_names):
            archtype_name = self.archtype.reactants[i]
            replacement_name = self.reactants_names[i]
            # Use regular expression to replace whole words only
            pattern = r'(?<!\w)' + re.escape(archtype_name) + r'(?!\w)'
            rate_law_str = re.sub(pattern, replacement_name, rate_law_str)
            i += 1 
            
        i = 0
        while i < len(self.products_names):
            archtype_name = self.archtype.products[i]
            replacement_name = self.products_names[i]
            # Use regular expression to replace whole words only
            pattern = r'(?<!\w)' + re.escape(archtype_name) + r'(?!\w)'
            rate_law_str = re.sub(pattern, replacement_name, rate_law_str)
            i += 1


        i = 0
        while i < len(self.archtype.extra_states):
            archtype_name = self.archtype.extra_states[i]
            replacement_name = self.extra_states[i]
            # Use regular expression to replace whole words only
            pattern = r'(?<!\w)' + re.escape(archtype_name) + r'(?!\w)'
            rate_law_str = re.sub(pattern, replacement_name, rate_law_str)
            i += 1

        r_index_p = r_index 
        if self.parameter_r_index != '':
            r_index_p = self.parameter_r_index

        i = 0
        while i < len(self.archtype.parameters):
            archtype_name = self.archtype.parameters[i]
            if self.exists_linked_parameters():
                replacement_name = str(self.linked_parameters[i])
            else:
                replacement_name = archtype_name + "_" + r_index_p
            # Use regular expression to replace whole words only
            pattern = r'(?<!\w)' + re.escape(archtype_name) + r'(?!\w)'
            rate_law_str = re.sub(pattern, replacement_name, rate_law_str)
            i += 1

        return f'{r_index}r: {product_str} -> {reactant_str}; {rate_law_str}'

    def copy(self) -> 'Reaction':
        '''
        returns a copy of the reaction
        '''
        return copy(self)


    def __str__(self) -> str:
        
        if self.archtype.reversible:
            return self.get_antimony_reaction_str(r_index='for') + '\n' + self.get_antimony_reactions_reverse_str(r_index='rev')

        return self.get_antimony_reaction_str(r_index='react' if self.name == '' else self.name)
