from typing import Union, Tuple, List 

class ReactionArchtype:

    '''
    ReactionArchtype is a factory class that generates Reaction objects
        name: str, 
        reactants: Tuple[str], 
        products: Tuple[str], 
        parameters: Tuple[str], 
        rate_law: str, 
        extra_states: Tuple[str] = (),
        assume_parameters_values: Union[dict, tuple] = None,
        assume_reactant_values: Union[dict, tuple] = None,
        assume_product_values: Union[dict, tuple] = None
    '''

    def __init__(self, 
        name: str, 
        reactants: Tuple[str], 
        products: Tuple[str], 
        parameters: Tuple[str], 
        rate_law: str, 
        extra_states: Tuple[str] = (),
        assume_parameters_values: Union[dict, tuple, None] = None,
        assume_reactant_values: Union[dict, tuple, None] = None,
        assume_product_values: Union[dict, tuple, None] = None,
        reversible: bool = False,
        reverse_rate_law: str = None):

        assert len(reactants) == len(set(reactants)), 'reactants must be unique'
        assert len(products) == 0 or len(products) == len(set(products)), 'products must be unique'
        assert len(parameters) == len(set(parameters)), 'parameters must be unique'
        assert len(extra_states) == len(set(extra_states)), 'extra_states must be unique'
        assert reversible is False or reverse_rate_law is not None, 'reverse_rate_law must be provided if reversible is True'
        
        self.name = name
        self.reactants = reactants
        self.products = products
        self.parameters = parameters
        self.extra_states = extra_states
        self.rate_law = rate_law

        # reverse reaction between reactants and products
        self.reversible = reversible
        self.reverse_rate_law = reverse_rate_law

        self.reactants_count = len(reactants)
        self.products_count = len(products)
        self.state_variables_count = len(reactants) + len(products)
        self.extra_states_count = len(extra_states)
        self.parameters_count = len(parameters)

        if reversible:
            self.validate_rate_laws(rate_law, reverse_rate_law)
        else:
            self.validate_rate_law(rate_law)
        self.validate_parameters(assume_parameters_values, parameters)
        self.validate_reactant_values(assume_reactant_values, reactants)
        self.validate_product_values(assume_product_values, products)

        # assumed values are used to fill in the missing values in the rate law
        # always update None or tuple into dict objects, so that Reaction class
        # only uses dict to handle assumed values logic 
        self.assume_parameters_values = self._convert_tuple_to_dict(assume_parameters_values, parameters)
        self.assume_reactant_values = self._convert_tuple_to_dict(assume_reactant_values, reactants)
        self.assume_product_values = self._convert_tuple_to_dict(assume_product_values, products)



    def _convert_tuple_to_dict(self, tuple: Union[dict, tuple, None], reference_tuple: Tuple[str]) -> dict:
        '''
        this function converts a tuple to a dict by 
        referring i-th element in the tuple to the i-th element in the reference_tuple
        '''
        # do not do anything if tuple is dict 
        if isinstance(tuple, dict):
            return tuple

        if tuple is None:
            return {}
        elif len(tuple) == 0: 
            return {}

        new_dict = {}
        for idx, value in enumerate(tuple):
            key = reference_tuple[idx]
            new_dict[key] = value
        
        return new_dict


    def validate_rate_law(self, rate_law) -> bool:
        '''
        this function validates the rate_law against the archtype
            It expects the rate_law to contain the parameters
            It expects the rate_law to contain the extra states
            ^ copilot generated
        '''

        for parameter in self.parameters:
            if rate_law.find(parameter) == -1:
                raise ValueError(f'{parameter} is not in the rate law')

        for extra_state in self.extra_states:
            if rate_law.find(extra_state) == -1:
                raise ValueError(f'{extra_state} is not in the rate law')

        return True

    def validate_rate_laws(self, rate_law, reverse_rate_law) -> bool:
        '''
        this function validates the rate_law against the archtype
            It expects the rate_law to contain the parameters
            It expects the rate_law to contain the extra states
            ^ copilot generated
        '''

        for parameter in self.parameters:
            if rate_law.find(parameter) == -1 and reverse_rate_law.find(parameter) == -1:
                raise ValueError(f'{parameter} is not in the rate law or reverse rate law')

        for extra_state in self.extra_states:
            if rate_law.find(extra_state) == -1 and reverse_rate_law.find(extra_state) == -1:
                raise ValueError(f'{extra_state} is not in the rate law or reverse rate law')

        return True

    def _abstract_validate_values(self, values: Union[tuple, dict, None], contained_list: Tuple) -> bool:
        
        '''
        this function validates the values against the archtype
        if values is tuple: 
            then it needs to be smaller than the number of [reactants, products, parameters] in the archtype
            assume first element in tuple matches with first element in self.parameters

        if values is dict:
            then it needs to be smaller than the number of [reactants, products, parameters] in the archtype
            cannot have dict keys that are not in self.parameters 
        
        '''

        if values is None:
            return True

        if len(values) != len(contained_list):
                raise ValueError(f'length of {values} is not the same as the archtype {contained_list}')

        if isinstance(values, dict):
            for key in values:
                if key not in contained_list:
                    raise ValueError(f'{key} is not in the {contained_list}')
        
        return True

    def validate_parameters(self, parameter_values, parameters_list) -> bool:

        '''
        Delegate to abstract implementation at _abstract_validate_values 
        '''

        return self._abstract_validate_values(parameter_values, parameters_list)


    def validate_reactant_values(self, reactant_values, reactant_list) -> bool:

        '''
        Delegate to abstract implementation at _abstract_validate_values 
        '''

        return self._abstract_validate_values(reactant_values, reactant_list)


    def validate_product_values(self, product_values, products_list) -> bool:

        '''
        Delegate to abstract implementation at _abstract_validate_values 
        '''

        return self._abstract_validate_values(product_values, products_list)
            

    def __str__(self) -> str:
        return f'{self.name} {self.reactants} {self.products} {self.parameters} {self.rate_law}'