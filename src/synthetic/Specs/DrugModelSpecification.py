from ..Utils import * 
from .ModelSpecification import ModelSpecification
from .Drug import Drug

class DrugModelSpecification(ModelSpecification):
    
    def __init__(self):
        super().__init__()
        self.drug_list = []
        self.drug_values = {}
        self.D_species = []
        
    def add_drug(self, drug: Drug, value=None):
        ''' 
        Adds a drug to the model. 
        Input: 
            drug: Drug | The drug to add to the model
            value: float | if not None, the value of the drug to set in the model
        '''
        
        self.drug_list.append(drug)
        if value is not None: 
            self.drug_values[drug.name] = value
        else: 
            self.drug_values[drug.name] = drug.default_value
        ## update species and regulations based on drug information
        
        # update drug species
        self.D_species.append(drug.name)
        
        # update regulations based on species
        for i in range(len(drug.regulation)): 
            specie = drug.regulation[i]
            type = drug.regulation_type[i]
            if specie not in self.A_species and specie not in self.B_species and specie not in self.C_species: 
                raise ValueError(f"Drug model not compatible: Specie {specie} not found in the model")
            if type != 'up' and type != 'down': 
                raise ValueError(f"Drug model not compatible: Regulation type must be either 'up' or 'down'")
            
            reg = (drug.name, specie)
            self.regulations.append(reg)
            self.regulation_types.append(type)
            
    def clear_drugs(self):
        ''' 
        Clears all drugs from the model. 
        '''
        # remove regulations based on species
        for drug in self.drug_list: 
            for i, reg in enumerate(self.regulations):
                if reg[0] == drug.name: 
                    self.regulations.remove(reg)
                    self.regulation_types.pop(i)
                    
        self.drug_list = []
        self.drug_values = {}
        self.D_species = []
        
    def __str__(self):
        disp_text = super().__str__()
        disp_text += "Drugs:\n"
        for drug in self.drug_list:
            disp_text += str(drug) + "\n"
        return disp_text
        
    def generate_specifications_old(self, random_seed, NA, NR, verbose=1):
        return super().generate_specifications_old(random_seed, NA, NR, verbose)
    
    def generate_network(self, network_name, mean_range_species, rangeScale_params, rangeMultiplier_params, verbose=1, random_seed=None):
        '''
        Returns a pre-compiled ModelBuilder object with the given specifications, 
        ready to be simulated. Pre-compiled model allows the user to manually set the initial values of the species
        before compiling to Antimony or SBML. 
        Parameters:
            network_name: str, the name of the network
            mean_range_species: tuple, the range of the mean values for the species
            rangeScale_params: tuple, the range of the scale values for the parameters
            rangeMultiplier_params: tuple, the range of the multiplier values for the parameters
            verbose: int, the verbosity level of the function
            random_seed: int, the random seed to use for reproducibility
        '''
        model = super().generate_network(network_name, mean_range_species, rangeScale_params, rangeMultiplier_params, verbose, random_seed)
        for drug in self.drug_list:
            model.add_simple_piecewise(0, drug.start_time, self.drug_values[drug.name], drug.name)
        model.precompile()
        return model 