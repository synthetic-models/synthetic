# The following function generates antimony strings from high level syntax

from typing import List, Union, Tuple
import antimony
import pickle
import warnings

from .Reaction import Reaction
from .ReactionArchtype import ReactionArchtype

class ModelBuilder:

    '''
    Docstring
    '''

    def __init__(self, name):
        self.name = name

        self.reactions: List[Reaction] = []
        # these two fields are coupled to self.reaction
        self.pre_compiled = False
        self.states = {}
        self.parameters = {}

        self.variables = {}
        self.enforce_state_values = {}
        self.custom_strings = {}
        
    def set_parameter(self, parameter_name: str, value: float):
        '''
        Set a parameter value in the model
        '''
        if parameter_name not in self.parameters:
            raise Exception(f'Parameter {parameter_name} not found in the model')
        if not self.pre_compiled: 
            raise Exception('Model must be pre-compiled before setting a parameter value, run self.precompile()')
        self.parameters[parameter_name] = value
        
    def get_parameter(self, parameter_name: str) -> float:
        '''
        Get a parameter value in the model
        '''
        if parameter_name not in self.parameters:
            raise Exception(f'Parameter {parameter_name} not found in the model')
        return self.parameters[parameter_name]
    
    def set_state(self, state_name: str, value: float):
        '''
        Set a state value in the model
        '''
        if state_name not in self.states:
            raise Exception(f'State {state_name} not found in the model')
        if not self.pre_compiled: 
            raise Exception('Model must be pre-compiled before setting a state value, run self.precompile()')
        self.states[state_name] = value
        
    def get_state(self, state_name: str) -> float:
        '''
        Get a state value in the model
        '''
        if state_name not in self.states:
            raise Exception(f'State {state_name} not found in the model')
        return self.states[state_name]

    def get_parameters(self) -> dict:
        '''
        Extracts parameters and their values from all reactions 
        in the class and returns a dict while subjecting to a naming rule.
        
        If model is pre-compiled, returns the cached parameters.
        '''
        if self.pre_compiled:
            return self.parameters.copy()
        
        parameters = {}
        i = 0
        while i < len(self.reactions):
            r = self.reactions[i]
            # first, get the parameters names from the archtype
            # and perform naming rule

            
            #Here, a simple naming rule is implemented. It simply appends the reaction index 
            #to the parameter name
            #TODO: implement more complex naming rules in the future 
            
            r_index = f'J{i}'

            parameters.update(r.get_reaction_parameters(r_index))

            i += 1

        return parameters

    def get_state_variables(self) -> dict:
        '''
        Extracts state variables and their values from all reactions 
        in the class and returns a dict

        non-unique state variables will only be repeated once, their 
        default value will only follow the first repeated state variable
        
        If model is pre-compiled, returns the cached states.
        '''
        if self.pre_compiled:
            return self.states.copy()

        states = {}
        for r in self.reactions:
            states.update(r.get_reaction_states())

        # enforce state values if defined in the class
        for k, v in self.enforce_state_values.items():
            states[k] = v

        return states

    def get_other_variables(self):
        '''
        Doc
        '''
        return self.variables

    def get_all_variables_keys(self, with_time=False):
        '''
        Doc
        '''
        if with_time:
            return list(self.states.keys()) + list(self.variables.keys()) + ['time']

        return list(self.get_state_variables().keys()) + list(self.variables.keys())

    def get_custom_variable_keys(self):
        '''
        Doc
        '''
        return list(self.variables.keys())

    def get_regulator_parameter_map(self) -> dict:
        '''
        Returns mapping from regulator (extra_state) names to parameter names (with r_index).
        The mapping is aggregated across all reactions in the model.
        '''
        if not self.pre_compiled:
            self.precompile()
        
        regulator_map = {}
        for i, r in enumerate(self.reactions):
            r_index = f'J{i}'
            for regulator in r.extra_states:
                param_bases = r.get_parameters_for_regulator(regulator)
                for param_base in param_bases:
                    param_name = f"{param_base}_{r_index}"
                    regulator_map.setdefault(regulator, []).append(param_name)
        return regulator_map

    def get_parameter_regulator_map(self) -> dict:
        '''
        Returns mapping from parameter names (with r_index) to regulator (extra_state) names.
        The mapping is aggregated across all reactions in the model.
        '''
        if not self.pre_compiled:
            self.precompile()
        
        param_map = {}
        for i, r in enumerate(self.reactions):
            r_index = f'J{i}'
            for param_base in r.archtype.parameters:
                regulator = r.get_regulator_for_parameter(param_base)
                if regulator:
                    param_name = f"{param_base}_{r_index}"
                    param_map[param_name] = regulator
        return param_map
    

    def add_reaction(self, reaction: Reaction):
        '''
        Add a reaction to the model, the reaction must be an instance of the Reaction class
        '''

        # validate that the reaction has a unique name within the model if a name is defined
        if reaction.name != '':
            for r in self.reactions:
                if r.name == reaction.name:
                    raise Exception(f'Reaction name is not unique within the model {reaction.name}')

        self.reactions.append(reaction)
        # reset the pre_compiled flag, since the model has changed 
        self.pre_compiled = False

    def inject_antimony_string_at(self, ant_string: str, position: str = 'reaction'):

        '''
        position can only be in str: top, reaction, state, parameters, end 
        '''
        warnings.warn('Adding variables using this function will result in a variable mismatch with antimony variables, since the variables will not be registered within the class itself', SyntaxWarning)
        all_positions = ['top', 'reaction', 'state', 'parameters', 'end']
        for p in all_positions:
            if p == position:
                if p in self.custom_strings:
                    self.custom_strings[p] += ant_string + '\n'
                else: 
                    self.custom_strings[p] = ant_string + '\n'
                break

    def add_enforce_state_value(self, state_name: str, value: float):
        '''
        Doc
        '''
        self.enforce_state_values[state_name] = value

    def add_custom_variables(self, variable_name, reaction_rule):
        '''
        Doc
        '''
        self.variables[variable_name] = reaction_rule


    def add_simple_piecewise(self, before_value: float, activation_time: float, after_value: float, state_name: str):
        '''
        Adds a simple piecewise function to the state variable state_name
        '''
        self.variables[state_name] = f'{state_name} := piecewise({before_value}, time < {activation_time}, {after_value})'
        # self.inject_antimony_string_at(f"{state_name} := piecewise({after_value}, time > {activation_time}, {before_value})", 'parameters')

    def delete_reaction(self, reaction_name: str):
        '''
        Deletes a reaction from the model
        '''
        for r in self.reactions:
            if r.name == reaction_name:
                self.reactions.remove(r)
                break
            
        self.pre_compiled = False

    def copy(self, overwrite_name='') -> 'ModelBuilder':
        '''
        Copy the model
        '''

        name = self.name if overwrite_name == '' else overwrite_name

        new_model = ModelBuilder(name)
        for r in self.reactions:
            new_model.add_reaction(r.copy())
        new_model.variables = self.variables
        new_model.enforce_state_values = self.enforce_state_values
        new_model.custom_strings = self.custom_strings
        
        # If the original model is pre-compiled, copy the pre-compiled state
        if self.pre_compiled:
            new_model.parameters = self.parameters.copy()
            new_model.states = self.states.copy()
            new_model.pre_compiled = True

        return new_model

    def combine(self, model: 'ModelBuilder', reactions_only=False) -> 'ModelBuilder':
        '''
        Combine two models
        '''
        new_model = self.copy()
        for r in model.reactions:
            new_model.add_reaction(r.copy())
        
        if not reactions_only:
            new_model.variables.update(model.variables)
            new_model.enforce_state_values.update(model.enforce_state_values)
            new_model.custom_strings.update(model.custom_strings)
        return new_model

    def __str__(self) -> str:
        
        return self.head()

    def head(self):
        '''
        Returns the general characteristics of the model
        '''
        return_str = ''
        return_str += f'Model Name {self.name}\n'
        return_str += f'Number of Reactions {len(self.reactions)}\n'
        return_str += f'Number of State Variables {len(self.states)}\n'
        return_str += f'Number of Parameters {len(self.parameters)}\n'
        return_str += f'Number of Custom Variables {len(self.variables)}\n'
        return_str += f'Number of Enforced State Values {len(self.enforce_state_values)}\n'
        return_str += f'Number of Custom Strings {len(self.custom_strings)}\n'
        return return_str

    def precompile(self):
        '''
        Populates the state and parameter variables list in the class which are used to generate the antimony text 
        The pre-compiled model will have functionalities for getting and setting specific state and parameter values, 
        which gets passed directly into the antimony string 
        '''
        self.parameters = self.get_parameters()
        self.states = self.get_state_variables()
        self.pre_compiled = True
    
    def get_antimony_model(self):
        '''
        Doc
        '''
        
        if not self.pre_compiled:
            raise Exception('Model must be pre-compiled before generating antimony string, run self.precompile(), the pre-compiled model will have functionalities for getting and setting specific state and parameter values, which gets passed directly into the antimony string')
        
        antimony_string = ''

        antimony_string += f'model {self.name}\n\n'

        # add top custom str 
        if 'top' in self.custom_strings:
            antimony_string += self.custom_strings['top']

        # add reactions
        
        # first, add reaction custom str 
        if 'reaction' in self.custom_strings:
            antimony_string += self.custom_strings['reaction']

        i = 0
        while i < len(self.reactions):
            r = self.reactions[i]
            r_index = f'J{i}'
            antimony_string += r.get_antimony_reaction_str(r_index)
            antimony_string += '\n'
            if r.reversible: 
                antimony_string += r.get_antimony_reactions_reverse_str(r_index)
                antimony_string += '\n'
            i += 1

        # add state vars

        # first, add state custom str
        antimony_string += '\n'
        antimony_string += '# State variables in the system\n'
        if 'state' in self.custom_strings:
            antimony_string += self.custom_strings['state']

        all_states = self.states
        for key, val in all_states.items():
            antimony_string += f'{key}={val}\n'
        antimony_string += '\n'

        # add parameters

        # first, add parameter custom str
        antimony_string += '# Parameters in the system\n'
        if 'parameters' in self.custom_strings:
            antimony_string += self.custom_strings['parameters']

        all_params = self.parameters
        for key, val in all_params.items():
            antimony_string += f'{key}={val}\n'

        # add other variables
        antimony_string += '\n'
        antimony_string += '# Other variables in the system\n'
        for key, val in self.variables.items():
            antimony_string += f'{val}\n'
        antimony_string += '\n' 

        # add end custom str
        if 'end' in self.custom_strings:
            antimony_string += self.custom_strings['end']

        antimony_string += '\nend'

        return antimony_string
    
    def get_sbml_model(self) -> str:

        '''
        Doc
        '''
        
        ant_model = self.get_antimony_model()
        antimony.clearPreviousLoads()
        antimony.freeAll()
        code = antimony.loadAntimonyString(ant_model)
        if code >= 0:
            mid = antimony.getMainModuleName()
            sbml_model = antimony.getSBMLString(mid)
            return sbml_model

        raise Exception('Error in loading antimony model', code)

    def get_sbml_model_from(self, ant_model) -> str:
        """
        Doc
        """

        antimony.clearPreviousLoads()
        antimony.freeAll()
        code = antimony.loadAntimonyString(ant_model)
        if code >= 0:
            mid = antimony.getMainModuleName()
            sbml_model = antimony.getSBMLString(mid)
            return sbml_model

        raise Exception("Error in loading antimony model", code)

    def save_antimony_model_as(self, file_name: str):
        '''
        Doc
        '''
        ant_model = self.get_antimony_model()
        with open(file_name, 'w') as f:
            f.write(ant_model)

    def save_sbml_model_as(self, file_name: str):
        '''
        Doc
        '''
        sbml_model = self.get_sbml_model()
        with open(file_name, 'w') as f:
            f.write(sbml_model)

    def save_model_as_pickle(self, file_name: str):
        '''
        Doc
        '''
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

    def plot(self):
        '''
        Plots the results of the simulation
        '''
        self.r_model.plot()
