class Drug: 
    
    def __init__(self, name, start_time, default_value, regulation=None, regulation_type=None):
        ''' 
        A drug class that represents a drug that can be applied to a model. 
        Input: 
            name: str | The name of the drug
            start_time: int | The time at which the drug is applied to the model.
            default_value: float | The default value of the drug in the model
            regulation: List[str] | The list of species that the drug regulates 
            regulation_type: List[str] | The type of regulation that the drug has on the species, either 'up' or 'down'
        '''
        self.name = name
        self.start_time = start_time
        self.default_value = default_value
        self.regulation = [] if regulation is None else regulation
        self.regulation_type = [] if regulation_type is None else regulation_type
        assert len(self.regulation) == len(self.regulation_type), "The regulation and regulation_type lists must be the same length"
        
    def __str__(self):
        return f"Drug({self.name}, {self.start_time}, {self.regulation}, {self.regulation_type})"
    
    def __repr__(self):
        return str(self)
    
    def add_regulation(self, specie, type): 
        ''' 
        Adds a regulation to the drug. 
        Input: 
            specie: str | The name of the specie to regulate
            type: str | The type of regulation, either 'up' or 'down'
        '''
        self.regulation.append(specie)
        self.regulation_type.append(type)
        
    def add_regulations(self, species, types):
        ''' 
        Adds multiple regulations to the drug. 
        Input: 
            species: List[str] | The list of species to regulate
            types: List[str] | The list of types of regulation, either 'up' or 'down'
        '''
        assert len(species) == len(types), "The species and types lists must be the same length"
        for i in range(len(species)): 
            self.add_regulation(species[i], types[i])
        
    def print_regulation(self): 
        ''' 
        Prints the regulation of the drug. 
        '''
        for i in range(len(self.regulation)): 
            print(f"{self.regulation[i]}: {self.regulation_type[i]}")