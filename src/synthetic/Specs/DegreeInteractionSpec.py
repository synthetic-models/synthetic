from typing import List, Dict, Optional, Tuple
import numpy as np
import logging
from .BaseSpec import BaseSpec
from .Regulation import Regulation
from .Drug import Drug
from .MichaelisNetworkSpec import MichaelisNetworkSpec
from ..ModelBuilder import ModelBuilder


class DegreeInteractionSpec(MichaelisNetworkSpec):
    """
    Multi-degree drug interaction network specification.
    
    Generates networks with hierarchical degrees of interaction:
    - Degree 1: D -> {R} -> {I} -> O (central pathway)
    - Degrees 2+: R -> I only (independent cascades influencing lower degrees)
    
    Each degree contains multiple cascades defined by degree_cascades list.
    Higher degrees provide feedback regulation to immediate lower degrees.
    
    Example:
        spec = DegreeInteractionSpec(degree_cascades=[1, 2, 5, 10])
        spec.generate_specifications(random_seed=42)
        model = spec.generate_network("multi_degree_network")
    """
    
    def __init__(self, degree_cascades: List[int], critical_pathways: Optional[int] = None):
        """
        Initialize degree interaction specification.
        
        Args:
            degree_cascades: List of cascade counts per degree, e.g., [1, 2, 5, 10]
            critical_pathways: Optional override for degree 1 cascade count
        """
        super().__init__()
        # Validate inputs before accessing degree_cascades[0]
        if len(degree_cascades) < 1:
            raise ValueError("degree_cascades must contain at least one element")
        
        self.degree_cascades = degree_cascades
        self.critical_pathways = critical_pathways or degree_cascades[0]
        self.degree_species: Dict[int, Dict[str, List[str]]] = {}  # {degree: {'R': [...], 'I': [...]}}
        self.degree_regulations: Dict[int, List[Regulation]] = {}  # Regulations per degree
        self._logger = logging.getLogger(__name__)
        
        if self.critical_pathways != degree_cascades[0]:
            self._logger.warning(f"critical_pathways ({self.critical_pathways}) differs from degree_cascades[0] ({degree_cascades[0]})")
    
    def generate_species_names(self) -> None:
        """
        Generate species names for all degrees based on cascade counts.
        Follows naming convention: R{degree}_{index}, I{degree}_{index}
        """
        self.degree_species.clear()
        
        for deg_idx, cascade_count in enumerate(self.degree_cascades):
            degree = deg_idx + 1  # 1-indexed degrees
            r_species = []
            i_species = []
            
            for cascade_idx in range(cascade_count):
                r_name = f"R{degree}_{cascade_idx + 1}"
                i_name = f"I{degree}_{cascade_idx + 1}"
                r_species.append(r_name)
                i_species.append(i_name)
                
                # Add to global species list
                self.add_species(r_name, group=f"degree_{degree}")
                self.add_species(i_name, group=f"degree_{degree}")
            
            self.degree_species[degree] = {'R': r_species, 'I': i_species}
        
        # Add outcome species O (only one)
        self.add_species("O", group="outcome")
    
    def generate_ordinary_regulations(self) -> None:
        """
        Generate ordinary regulations within each cascade:
        - Degree 1: R_i -> I_i, I_i -> O
        - Degrees 2+: R_i -> I_i only
        """
        self.degree_regulations.clear()
        
        for degree, species_dict in self.degree_species.items():
            degree_regs = []
            r_species = species_dict['R']
            i_species = species_dict['I']
            
            # Each cascade: R_i -> I_i
            for r_name, i_name in zip(r_species, i_species):
                reg = Regulation(from_specie=r_name, to_specie=i_name, reg_type='up')
                self.regulations.append(reg)
                degree_regs.append(reg)
            
            # Degree 1: I_i -> O
            if degree == 1:
                for i_name in i_species:
                    reg = Regulation(from_specie=i_name, to_specie='O', reg_type='up')
                    self.regulations.append(reg)
                    degree_regs.append(reg)
            
            self.degree_regulations[degree] = degree_regs
        
        # Track ordinary regulations (non-feedback)
        self.ordinary_regulations = self.regulations.copy()
    
    def generate_feedback_regulations(self, random_seed: Optional[int] = None, 
                                     feedback_density: float = 0.5) -> None:
        """
        Generate feedback regulations between adjacent degrees following structured rules.
        
        Rules:
        1. Downward regulations (n -> n-1): MANDATORY
           - For each cascade i in degree n > 1:
             - Regulator: I{n}_i (I species of that cascade)
             - Target: Random R or I species in degree (n-1)
             - Type: Random 'up' or 'down'
        
        2. Upward regulations (n-1 -> n): DENSITY-CONTROLLED
           - For each cascade i in degree n > 1:
             - Create candidate regulation:
               - Regulator: Random R or I species in degree (n-1)
               - Target: Random R{n}_i or I{n}_i
               - Type: Random 'up' or 'down'
           - Trim from outermost degrees first:
             - Keep only floor(feedback_density * cascade_count) upward regulations per degree
             - Randomly select which cascades keep their upward regulation
        
        Args:
            random_seed: Random seed for reproducibility
            feedback_density: Proportion of upward regulations to keep (0-1)
                - 0: No upward regulations
                - 1: All cascades get upward regulation
                - (0,1): Random subset based on density
        """
        if random_seed is not None:
            rng = np.random.default_rng(random_seed)
        else:
            rng = np.random.default_rng()
        
        # Clear existing feedback regulations
        self.feedback_regulations = []
        
        # Store candidate upward regulations by degree for trimming
        upward_candidates_by_degree: Dict[int, List[Tuple[int, Regulation]]] = {}
        # (degree, (cascade_index, regulation))
        
        # Helper function to check if regulation already exists
        def regulation_exists(regulator: str, target: str) -> bool:
            return any(
                r.from_specie == regulator and r.to_specie == target
                for r in self.regulations + self.feedback_regulations
            )
        
        # Generate regulations for each degree n > 1
        for degree in range(2, len(self.degree_cascades) + 1):
            prev_degree = degree - 1
            
            # Get species from both degrees
            prev_r_species = self.degree_species[prev_degree]['R']
            prev_i_species = self.degree_species[prev_degree]['I']
            prev_all_species = prev_r_species + prev_i_species
            
            curr_r_species = self.degree_species[degree]['R']
            curr_i_species = self.degree_species[degree]['I']
            curr_all_species = curr_r_species + curr_i_species
            
            if not prev_all_species or not curr_all_species:
                continue
            
            cascade_count = len(curr_r_species)  # Should be same as len(curr_i_species)
            upward_candidates_by_degree[degree] = []
            
            # 1. MANDATORY DOWNWARD REGULATIONS (n -> n-1)
            for i in range(cascade_count):
                regulator = curr_i_species[i]  # I{degree}_{i+1}
                
                # Randomly select target from previous degree
                target = rng.choice(prev_all_species)
                reg_type = rng.choice(['up', 'down'])
                
                # Ensure no duplicate regulation
                attempts = 0
                while regulation_exists(regulator, target) and attempts < 10:
                    target = rng.choice(prev_all_species)
                    attempts += 1
                
                reg = Regulation(from_specie=regulator, to_specie=target, reg_type=reg_type)
                self.regulations.append(reg)
                self.feedback_regulations.append(reg)
            
            # 2. CREATE CANDIDATE UPWARD REGULATIONS (n-1 -> n)
            for i in range(cascade_count):
                # Randomly select regulator from previous degree
                regulator = rng.choice(prev_all_species)
                
                # Randomly select target from current cascade (R or I)
                target_options = [curr_r_species[i], curr_i_species[i]]
                target = rng.choice(target_options)
                
                reg_type = rng.choice(['up', 'down'])
                
                # Ensure no duplicate regulation
                attempts = 0
                while regulation_exists(regulator, target) and attempts < 10:
                    regulator = rng.choice(prev_all_species)
                    target = rng.choice(target_options)
                    attempts += 1
                
                reg = Regulation(from_specie=regulator, to_specie=target, reg_type=reg_type)
                upward_candidates_by_degree[degree].append((i, reg))
        
        # 3. TRIM UPWARD REGULATIONS FROM OUTERMOST DEGREES FIRST
        # Process degrees from highest to lowest
        degrees_sorted = sorted(upward_candidates_by_degree.keys(), reverse=True)
        
        for degree in degrees_sorted:
            candidates = upward_candidates_by_degree[degree]
            if not candidates:
                continue
            
            cascade_count = len(self.degree_species[degree]['R'])
            
            # Calculate how many upward regulations to keep
            if feedback_density <= 0:
                # No upward regulations
                continue
            elif feedback_density >= 1:
                # Keep all upward regulations
                num_to_keep = len(candidates)
            else:
                # Keep proportion based on density
                num_to_keep = max(1, int(feedback_density * cascade_count))
                num_to_keep = min(num_to_keep, len(candidates))
            
            # Randomly select which cascades keep their upward regulation
            # We want to select num_to_keep unique cascade indices
            cascade_indices = list(set(idx for idx, _ in candidates))
            if len(cascade_indices) > num_to_keep:
                keep_indices = set(rng.choice(cascade_indices, size=num_to_keep, replace=False))
            else:
                keep_indices = set(cascade_indices)
            
            # Add kept regulations
            for cascade_idx, reg in candidates:
                if cascade_idx in keep_indices:
                    self.regulations.append(reg)
                    self.feedback_regulations.append(reg)
            
            self._logger.debug(
                f"Degree {degree}: Kept {len(keep_indices)}/{cascade_count} upward regulations "
                f"(density={feedback_density})"
            )
        
        self._logger.info(
            f"Generated {len(self.feedback_regulations)} feedback regulations "
            f"(density={feedback_density})"
        )
    
    def generate_specifications(self, random_seed: Optional[int] = None, 
                               feedback_density: float = 0.5, **kwargs):
        """
        Generate complete network specifications.
        
        Args:
            random_seed: Random seed for reproducibility
            feedback_density: Proportion of feedback connections to create (0-1)
            **kwargs: Additional arguments (unused, for compatibility)
        """
        # Clear existing state
        self.regulations = []
        self.species_list = []
        self.species_groups = {}
        self.ordinary_regulations = []
        self.feedback_regulations = []
        self.drugs = []
        self.drug_values = {}
        
        # Generate species and regulations
        self.generate_species_names()
        self.generate_ordinary_regulations()
        self.generate_feedback_regulations(random_seed, feedback_density)
        
        self._logger.info(
            f"Generated {len(self.degree_cascades)}-degree network with "
            f"{len(self.species_list)} species and {len(self.regulations)} regulations"
        )
    
    def add_drug(self, drug: Drug, value: Optional[float] = None):
        """
        Add a drug to the model, validating it only targets degree 1 R species.
        
        Args:
            drug: Drug object containing name, start_time, default_value, and regulations
            value: Optional override for drug value
            
        Raises:
            ValueError: If drug targets non-degree-1 species or non-R species
        """
        # Validate drug targets
        for target_species in drug.regulation:
            # Check if target is a degree 1 R species
            is_degree1_r = any(
                target_species == r_name 
                for r_name in self.degree_species.get(1, {}).get('R', [])
            )
            
            if not is_degree1_r:
                raise ValueError(
                    f"Drug {drug.name} can only target degree 1 R species. "
                    f"Invalid target: {target_species}"
                )
        
        # Call parent method
        super().add_drug(drug, value)
    
    def get_species_by_degree(self, degree: int, species_type: str = 'all') -> List[str]:
        """
        Get species for a specific degree.
        
        Args:
            degree: Degree number (1-indexed)
            species_type: 'R', 'I', or 'all'
            
        Returns:
            List of species names
        """
        if degree not in self.degree_species:
            return []
        
        species_dict = self.degree_species[degree]
        if species_type == 'R':
            return species_dict['R']
        elif species_type == 'I':
            return species_dict['I']
        elif species_type == 'all':
            return species_dict['R'] + species_dict['I']
        else:
            raise ValueError("species_type must be 'R', 'I', or 'all'")
    
    def get_regulations_by_degree(self, degree: Optional[int] = None) -> List[Regulation]:
        """
        Get regulations for a specific degree or all degrees.
        
        Args:
            degree: Degree number (1-indexed) or None for all
            
        Returns:
            List of Regulation objects
        """
        if degree is None:
            return self.regulations.copy()
        elif degree in self.degree_regulations:
            return self.degree_regulations[degree].copy()
        else:
            return []
    
    def get_total_species_count(self) -> int:
        """Get total number of species in the network."""
        return len(self.species_list)
    
    def get_total_cascades(self) -> int:
        """Get total number of cascades across all degrees."""
        return sum(self.degree_cascades)
    
    def __str__(self) -> str:
        """String representation of the specification."""
        return (f"DegreeInteractionSpec(degrees={len(self.degree_cascades)}, "
                f"cascades={self.degree_cascades}, "
                f"species={self.get_total_species_count()}, "
                f"regulations={len(self.regulations)})")
