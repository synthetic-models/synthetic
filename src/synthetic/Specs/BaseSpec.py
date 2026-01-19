from abc import ABC, abstractmethod
from typing import List
from ..ModelBuilder import ModelBuilder
from .Regulation import Regulation


class BaseSpec(ABC):
    """
    Minimal abstract model specification. Contains only the essential contract for generating models.
    At the core, generate specifications takes in specific inputs from the user, which is then 
    used to infer the regulation list. 
    
    Attributes:
        regulations (List[Regulation]): List of regulations applicable to the model.
    """

    # Core regulation storage 
    regulations: List[Regulation]
    species_list: List[str]

    def __init__(self):
        self.regulations = [] 
        self.species_list = [] # this will be the final list of species in the model

    @abstractmethod
    def generate_specifications(self, **kwargs):
        """
        Populate the specification's internal state.
        This includes species lists and regulations based on the specific model type.
        """
        pass

    @abstractmethod
    def generate_network(self, network_name: str, **kwargs) -> ModelBuilder:
        """
        Generate a complete ModelBuilder instance from the specification.
        """
        pass
