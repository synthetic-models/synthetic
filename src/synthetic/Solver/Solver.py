# abstract class solver 
# This class is used to solve the problem

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple
import pandas as pd

class Solver(ABC):
    """
    Abstract class for an ODE solver that can run simulations 
    """
    
    def __init__(self):
        super().__init__()
        self.last_sim_result = None
        
    @abstractmethod
    def compile(self, compile_str: str, **kwargs):
        """
        Return a model instance which can be used to run simulations. 
        The compile_str is a string either in antimony or sbml format, alternatively it can be a file path to a file in either of these formats.
        The kwargs are additional arguments that can be passed for the creation of the model instance.
        """
        pass 

    @abstractmethod
    def simulate(self, start: float, stop: float, step: float) -> pd.DataFrame:
        """
        Simulate the problem from start to stop with a given step size.
        Returns a pandas dataframe with the results with the following columns:
        - time: time points of the simulation
        - [species]: species names
        with each row corresponds to a time point and each column corresponds to a species.
        """
        pass
    
    def set_state_values(self, state_values: Dict[str, float]) -> bool:
        """
        Hot swapping of state variables in the running instance of the model.
        Set the values of state variables in the model instance, this should only possible after compiling the model. 
        Not every solver will support this, so it is possible that this function to return an not implemented error.
        returns True if the state variable was set successfully, False otherwise.
        """
        pass
     
    def set_parameter_values(self, parameter_values: Dict[str, float]) -> bool:
        """
        Hot swapping of parameters in the running instance of the model.
        Set the values of parameter variables in the model instance, this should only possible after compiling the model. 
        Not every solver will support this, so it is possible that this function to return an not implemented error.
        returns True if the state variable was set successfully, False otherwise.
        """
        pass
