from models.Solver.Solver import Solver

from typing import Dict, Any, Tuple
import pandas as pd
from roadrunner import RoadRunner

class RoadrunnerSolver(Solver):
    """
    RoadRunner-based ODE solver for biochemical models.
    
    This solver uses the libRoadRunner library to compile and simulate
    biochemical models defined in Antimony or SBML formats. It provides
    an interface for setting initial states and parameters, and returns
    simulation results as pandas DataFrames.
    
    Inherits from:
        Solver: Abstract base class for ODE solvers
    """
    
    def __init__(self):
        super().__init__()
        self.roadrunner_instance = None
        self.last_sim_result = None
        
    def compile(self, compile_str: str, **kwargs) -> RoadRunner:
        """
        Compile a model string using libRoadRunner.
        
        Args:
            compile_str: Model definition in Antimony or SBML format
            **kwargs: Additional options passed to RoadRunner constructor
        
        Returns:
            RoadRunner: The compiled RoadRunner instance
        """
        self.roadrunner_instance = RoadRunner(compile_str, **kwargs)
    
    def simulate(self, start: float, stop: float, step: float) -> pd.DataFrame:
        """
        Simulate the problem from start to stop with a given step size.
        Returns a pandas dataframe with the results with the following columns:
        - time: time points of the simulation
        - [species]: species names
        with each row corresponds to a time point and each column corresponds to a species.
        """
        
        # Check if the roadrunner instance is created
        if self.roadrunner_instance is None:
            raise ValueError("RoadRunner instance is not created. Please call compile() first.")
        
        runner = self.roadrunner_instance
        res = runner.simulate(start, stop, step)
        # Convert the result to a pandas dataframe, by default, this will not work 
        
        
        ## First step is to obtain all the state variables in the model
        state_vars = runner.model.getFloatingSpeciesIds()
        
        new_data = []
        new_data.append(res['time'])
        for state in state_vars:
            new_data.append(res[f'[{state}]'])
        
        # Convert the result to a pandas dataframe
        df = pd.DataFrame(new_data).T
        df.columns = ['time'] + list(state_vars)
        
        # reset the model to the initial state
        runner.resetToOrigin()
        return df
    
    def set_state_values(self, state_values: Dict[str, float]) -> bool:
        """
        Hot swapping of state variables in the running instance of the model, note this is setting the initial values of the state variables.
        Set the values of state variables in the model instance, this should only possible after compiling the model. 
        Not every solver will support this, so it is possible that this function to return an not implemented error.
        returns True if the state variable was set successfully, False otherwise.
        """
        # Check if the roadrunner instance is created
        if self.roadrunner_instance is None:
            raise ValueError("RoadRunner instance is not created. Please call compile() first.")
        
        runner = self.roadrunner_instance
        
        # Set the state values in the model instance
        for state, value in state_values.items():
            try:
                runner[f'init({state})'] = value
            except Exception as e:
                print(f"Error setting state variable {state}: {e}")
                return False
        
        return True
        

    def set_parameter_values(self, parameter_values: Dict[str, float]) -> bool:
        """
        Hot swapping of parameters in the running instance of the model.
        Set the values of parameter variables in the model instance, this should only possible after compiling the model. 
        Not every solver will support this, so it is possible that this function to return an not implemented error.
        returns True if the state variable was set successfully, False otherwise.
        """
        # Check if the roadrunner instance is created
        if self.roadrunner_instance is None:
            raise ValueError("RoadRunner instance is not created. Please call compile() first.")
        
        runner = self.roadrunner_instance
        
        # Set the parameter values in the model instance
        for param, value in parameter_values.items():
            try:
                runner[param] = value
            except Exception as e:
                print(f"Error setting parameter {param}: {e}")
                return False
        
        return True
