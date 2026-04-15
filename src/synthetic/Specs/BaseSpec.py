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

    @abstractmethod
    def get_auto_drug_targets(self) -> List[str]:
        """
        Get target species for an auto-generated drug.
        """
        pass

    def get_outcome_species(self) -> List[str]:
        """
        Get species names that represent outcomes/targets for analysis.
        These are typically excluded from features.
        """
        return []

    def is_activated_form(self, species_name: str) -> bool:
        """
        Check if a species name represents an activated form of another species.
        These are typically excluded from initial value perturbations.
        """
        return species_name.endswith('a')
