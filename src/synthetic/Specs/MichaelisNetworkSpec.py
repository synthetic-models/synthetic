from typing import List, Dict, Optional, Tuple
import numpy as np
import logging
from ..ArchtypeCollections import create_archtype_michaelis_menten_v2, michaelis_menten, create_archtype_basal_michaelis
from ..ReactionArchtype import ReactionArchtype
from ..ModelBuilder import ModelBuilder
from ..Reaction import Reaction
from .BaseSpec import BaseSpec
from .Regulation import Regulation
from .Drug import Drug


class MichaelisNetworkSpec(BaseSpec):
    """
    Generic Michaelis-Menten network specification with ModelSpec4-style drug mechanism.
    
    Features:
    1. Generic species management (any network topology)
    2. ModelSpec4's reaction generation scheme
    3. ModelSpec4's drug mechanism (drugs as species with regulations)
    4. Flexible specification generation (random or manual)
    5. Michaelis-Menten kinetics with up/down regulation
    
    Example usage:
    ```python
    spec = MichaelisNetworkSpec()
    spec.generate_specifications(num_species=5, num_regulations=8, random_seed=42)
    model = spec.generate_network("my_network")
    print(model.get_antimony_model())
    ```
    """
    
    def __init__(self, use_basal_activation: bool = False):
        super().__init__()
        self.use_basal_activation = use_basal_activation
        self.drugs: List[Drug] = []
        self.drug_values: Dict[str, float] = {}
        self.species_groups: Dict[str, List[str]] = {}
        self.randomise_parameters = True
        self.ordinary_regulations: List[Regulation] = []
        self.feedback_regulations: List[Regulation] = []
        self._logger = logging.getLogger(__name__)
    
    # === Species Management ===
    def add_species(self, species: str, group: str = "default"):
        """
        Add species to the model.
        
        Args:
            species: Name of the species
            group: Optional group name for categorization
        """
        if species not in self.species_list:
            self.species_list.append(species)
        if group not in self.species_groups:
            self.species_groups[group] = []
        if species not in self.species_groups[group]:
            self.species_groups[group].append(species)
    
    def get_species_by_group(self, group: str) -> List[str]:
        """Get all species in a specific group."""
        return self.species_groups.get(group, [])
    
    # === Drug Mechanism (from ModelSpec4) ===
    def add_drug(self, drug: Drug, value: Optional[float] = None):
        """
        Add a drug to the model using ModelSpec4's mechanism.
        
        Args:
            drug: Drug object containing name, start_time, default_value, and regulations
            value: Optional override for drug value
        """
        self.drugs.append(drug)
        if value is not None:
            self.drug_values[drug.name] = value
        else:
            self.drug_values[drug.name] = drug.default_value
        
        # Add drug as a species
        self.add_species(drug.name, "drugs")
        
        # Create regulations from drug to target species
        for i in range(len(drug.regulation)):
            specie = drug.regulation[i]
            reg_type = drug.regulation_type[i]
            
            # Validate target species exists
            if specie not in self.species_list:
                raise ValueError(f"Drug model not compatible: Specie {specie} not found in the model")
            if reg_type not in ['up', 'down']:
                raise ValueError("Drug model not compatible: Regulation type must be either 'up' or 'down'")
            
            # Create Regulation object
            reg = Regulation(from_specie=drug.name, to_specie=specie, reg_type=reg_type)
            self.regulations.append(reg)
    
    def clear_drugs(self):
        """Remove all drugs and their associated regulations."""
        # Remove drug species
        for drug in self.drugs:
            if drug.name in self.species_list:
                self.species_list.remove(drug.name)
            if "drugs" in self.species_groups and drug.name in self.species_groups["drugs"]:
                self.species_groups["drugs"].remove(drug.name)
        
        # Remove drug regulations
        drug_names = [d.name for d in self.drugs]
        self.regulations = [r for r in self.regulations if r.from_specie not in drug_names]
        
        # Clear drug lists
        self.drugs = []
        self.drug_values = {}

    # === Regulation Management ===
    def add_regulation(self, from_specie: str, to_specie: str, reg_type: str, 
                       is_feedback: bool = False):
        """
        Add regulation between species.
        
        Args:
            from_specie: Regulating species
            to_specie: Regulated species
            reg_type: 'up' or 'down'
            is_feedback: Whether this is a feedback regulation
        """
        if reg_type not in ['up', 'down']:
            raise ValueError("Regulation type must be either 'up' or 'down'")
        
        if from_specie not in self.species_list:
            self.add_species(from_specie)
        if to_specie not in self.species_list:
            self.add_species(to_specie)
        
        reg = Regulation(from_specie, to_specie, reg_type)
        self.regulations.append(reg)
        
        # Track as ordinary or feedback
        if is_feedback:
            self.feedback_regulations.append(reg)
        else:
            self.ordinary_regulations.append(reg)
    
    def get_regulators_for_species(self, specie: str) -> List[Tuple[str, str]]:
        """
        Get all regulators for a species.
        
        Returns:
            List of (regulator_name, reg_type) tuples
        """
        regulators = []
        for reg in self.regulations:
            if reg.to_specie == specie:
                regulators.append((reg.from_specie, reg.reg_type))
        return regulators
    
    # === Core Abstract Method Implementations ===
    def generate_specifications(self, 
                               num_species: Optional[int] = None,
                               num_regulations: Optional[int] = None,
                               species_names: Optional[List[str]] = None,
                               regulation_list: Optional[List[Tuple[str, str, str]]] = None,
                               random_seed: Optional[int] = None,
                               **kwargs):
        """
        Generate network specifications.
        
        Can generate random network or use provided species/regulations.
        
        Args:
            num_species: Number of species for random generation
            num_regulations: Number of regulations for random generation
            species_names: List of species names for manual specification
            regulation_list: List of (from_specie, to_specie, reg_type) tuples
            random_seed: Random seed for reproducibility
        """
        # Clear existing state
        self.regulations = []
        self.species_list = []
        self.species_groups = {}
        self.ordinary_regulations = []
        self.feedback_regulations = []
        
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
        
        # Handle manual specification
        if species_names is not None:
            for species in species_names:
                self.add_species(species)
        
        # Handle random generation
        elif num_species is not None:
            # Generate species names S1, S2, ..., SN
            species_names = [f"S{i+1}" for i in range(num_species)]
            for species in species_names:
                self.add_species(species)
        
        # Add regulations
        if regulation_list is not None:
            for from_specie, to_specie, reg_type in regulation_list:
                self.add_regulation(from_specie, to_specie, reg_type)
        
        elif num_regulations is not None and len(self.species_list) > 0:
            # Generate random regulations
            species = self.species_list.copy()
            for _ in range(num_regulations):
                from_specie = rng.choice(species)
                to_specie = rng.choice([s for s in species if s != from_specie])
                reg_type = rng.choice(['up', 'down'])
                self.add_regulation(from_specie, to_specie, reg_type, is_feedback=True)
        
        self._logger.debug(f"Generated specifications: {len(self.species_list)} species, {len(self.regulations)} regulations")
    
    def generate_network(self,
                        network_name: str,
                        mean_range_species: Tuple[int, int] = (1, 100),
                        rangeScale_params: Tuple[float, float] = (0.5, 2.0),
                        rangeMultiplier_params: Tuple[float, float] = (0.8, 1.2),
                        random_seed: Optional[int] = None,
                        receptor_basal_activation: bool = False,
                        **kwargs) -> ModelBuilder:
        """
        Generate ModelBuilder using Michaelis-Menten reaction scheme.
        
        Args:
            network_name: Name of the network
            mean_range_species: Range for initial species values
            rangeScale_params: Range for parameter scaling
            rangeMultiplier_params: Range for parameter multiplier
            random_seed: Random seed for reproducibility
            receptor_basal_activation: Whether to use basal activation for all species
            
        Returns:
            Pre-compiled ModelBuilder object
        """
        model = ModelBuilder(network_name)
        
        # Set up random number generator
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
        
        def add_reactions(specie: str, model: ModelBuilder, basal: bool = False):
            """Helper to add forward and reverse reactions for a species."""
            forward_reaction = self.get_forward_reaction(
                specie, mean_range_species, rangeScale_params, 
                rangeMultiplier_params, rng, basal=basal
            )
            reverse_reaction = self.get_reverse_reaction(
                specie, rangeScale_params, rangeMultiplier_params, rng
            )
            # Add reverse first, then forward (ModelSpec4 pattern)
            model.add_reaction(reverse_reaction)
            model.add_reaction(forward_reaction)
        
        # Generate reactions for all species (excluding drugs)
        non_drug_species = [s for s in self.species_list if s not in [d.name for d in self.drugs]]
        
        for specie in non_drug_species:
            add_reactions(specie, model, basal=receptor_basal_activation)
        
        # Handle drugs (add piecewise functions)
        for drug in self.drugs:
            model.add_simple_piecewise(
                0, drug.start_time, self.drug_values[drug.name], drug.name
            )
        
        model.precompile()
        
        self._logger.debug(f"Generated model {network_name} with {len(model.reactions)} reactions")
        self._logger.debug(f"Species: {len(model.states)}, Parameters: {len(model.parameters)}")
        
        return model
    
    # === Reaction Generation Helpers (from ModelSpec4) ===
    def generate_forward_archtype_and_regulators(self, specie: str) -> Tuple[ReactionArchtype, List[str]]:
        """
        Generate forward reaction archtype and regulators for a species.
        Direct copy from ModelSpec4.
        """
        all_regulations = [(reg.from_specie, reg.to_specie) for reg in self.regulations]
        all_regulation_types = [reg.reg_type for reg in self.regulations]
        
        regulators_for_specie = []
        for i, reg in enumerate(all_regulations):
            if reg[1] == specie:
                reg_type = all_regulation_types[i]
                regulators_for_specie.append((reg[0], reg_type))
        
        if len(regulators_for_specie) == 0:
            return michaelis_menten, []
        
        total_up_regulations = len([r for r in regulators_for_specie if r[1] == 'up'])
        total_down_regulations = len([r for r in regulators_for_specie if r[1] == 'down'])
        
        rate_law = create_archtype_michaelis_menten_v2(
            stimulators=0,
            stimulator_weak=total_up_regulations,
            allosteric_inhibitors=total_down_regulations,
            competitive_inhibitors=0
        )
        
        # Sort regulators by type, up first and down second
        regulators_for_specie = sorted(regulators_for_specie, key=lambda x: x[1], reverse=True)
        regulators_sorted = [r[0] for r in regulators_for_specie]
        
        return rate_law, regulators_sorted
    
    def generate_forward_archtype_and_regulators_basal(self, specie: str) -> Tuple[ReactionArchtype, List[str]]:
        """
        Generate forward reaction archtype with basal activation.
        Direct copy from ModelSpec4.
        """
        all_regulations = [(reg.from_specie, reg.to_specie) for reg in self.regulations]
        all_regulation_types = [reg.reg_type for reg in self.regulations]
        
        regulators_for_specie = []
        for i, reg in enumerate(all_regulations):
            if reg[1] == specie:
                reg_type = all_regulation_types[i]
                regulators_for_specie.append((reg[0], reg_type))
        
        if len(regulators_for_specie) == 0:
            return michaelis_menten, []
        
        total_up_regulations = len([r for r in regulators_for_specie if r[1] == 'up'])
        total_down_regulations = len([r for r in regulators_for_specie if r[1] == 'down'])
        
        rate_law = create_archtype_basal_michaelis(
            stimulators=0,
            stimulator_weak=total_up_regulations,
            allosteric_inhibitors=total_down_regulations,
            competitive_inhibitors=0
        )
        
        regulators_for_specie = sorted(regulators_for_specie, key=lambda x: x[1], reverse=True)
        regulators_sorted = [r[0] for r in regulators_for_specie]
        
        return rate_law, regulators_sorted
    
    def generate_reverse_archtype_and_regulators(self, specie: str) -> Tuple[ReactionArchtype, List[str]]:
        """
        Generate reverse reaction archtype and regulators.
        Direct copy from ModelSpec4.
        """
        all_regulations = [(reg.from_specie, reg.to_specie) for reg in self.regulations]
        all_regulation_types = [reg.reg_type for reg in self.regulations]
        
        regulators_for_specie = []
        for i, reg in enumerate(all_regulations):
            if reg[1] == specie:
                reg_type = all_regulation_types[i]
                if reg_type == 'down':
                    regulators_for_specie.append((reg[0], reg_type))
        
        if len(regulators_for_specie) == 0:
            return michaelis_menten, []
        
        return michaelis_menten, []
    
    def generate_random_parameters(self, 
                                  reaction_archtype: ReactionArchtype,
                                  scale_range: Tuple[float, float],
                                  multiplier_range: Tuple[float, float],
                                  random_seed: Optional[int] = None) -> Tuple:
        """
        Generate random parameters informed by scale.
        Direct copy from ModelSpec4.
        """
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
        
        assumed_values = reaction_archtype.assume_parameters_values
        r_params = []
        for _, value in assumed_values.items():
            rand = rng.uniform(value * scale_range[0], value * scale_range[1])
            rand *= rng.uniform(multiplier_range[0], multiplier_range[1])
            r_params.append(rand)
        
        return tuple(r_params)
    
    # === Reaction Creation Methods ===
    def get_forward_reaction(self, specie: str, mean_range_species: Tuple[int, int],
                            rangeScale_params: Tuple[float, float],
                            rangeMultiplier_params: Tuple[float, float],
                            rng: np.random.Generator,
                            basal: bool = False) -> Reaction:
        """
        Create forward reaction for a species.
        Adapted from ModelSpec4.
        """
        if basal or self.use_basal_activation:
            forward_rate_law, regulators = self.generate_forward_archtype_and_regulators_basal(specie)
        else:
            forward_rate_law, regulators = self.generate_forward_archtype_and_regulators(specie)
        
        activated_regulators = []
        for r in regulators:
            if 'D' in r:  # Drug does not get activated (ModelSpec4 pattern)
                activated_regulators.append(r)
            else:
                activated_regulators.append(r + 'a')
        
        forward_params = self.generate_random_parameters(
            forward_rate_law, rangeScale_params, rangeMultiplier_params, rng
        )
        forward_state_val = rng.integers(mean_range_species[0], mean_range_species[1])
        
        forward_reaction = Reaction(
            forward_rate_law,
            (specie,),
            (specie + 'a',),
            reactant_values=forward_state_val,
            extra_states=tuple(activated_regulators),
            parameters_values=tuple(forward_params),
            zero_init=False
        )
        return forward_reaction
    
    def get_reverse_reaction(self, specie: str,
                            rangeScale_params: Tuple[float, float],
                            rangeMultiplier_params: Tuple[float, float],
                            rng: np.random.Generator) -> Reaction:
        """
        Create reverse reaction for a species.
        Adapted from ModelSpec4.
        """
        reverse_rate_law, regulators = self.generate_reverse_archtype_and_regulators(specie)
        
        activated_regulators = []
        for r in regulators:
            if 'D' in r:
                activated_regulators.append(r)
            else:
                activated_regulators.append(r + 'a')
        
        reverse_params = self.generate_random_parameters(
            reverse_rate_law, rangeScale_params, rangeMultiplier_params, rng
        )
        
        reverse_reaction = Reaction(
            reverse_rate_law,
            (specie + 'a',),
            (specie,),
            extra_states=tuple(activated_regulators),
            parameters_values=tuple(reverse_params),
            zero_init=False
        )
        return reverse_reaction
    
    # === Utility Methods ===
    def get_all_species(self, 
                       include_drugs: bool = True,
                       include_default: bool = True) -> List[str]:
        """
        Get all species in the model.
        
        Args:
            include_drugs: Whether to include drug species
            include_default: Whether to include non-drug species
            
        Returns:
            List of species names
        """
        species = []
        if include_default:
            drug_names = [d.name for d in self.drugs]
            species.extend([s for s in self.species_list if s not in drug_names])
        if include_drugs:
            species.extend([d.name for d in self.drugs])
        return species
    
    def __str__(self) -> str:
        """String representation of the specification."""
        return (f"MichaelisNetworkSpec(species={len(self.species_list)}, "
                f"regulations={len(self.regulations)}, drugs={len(self.drugs)})")
