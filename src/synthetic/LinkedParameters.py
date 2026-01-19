
from dataclasses import dataclass


@dataclass
class LinkedParameters: 

    '''
    represents parameters which are not bound to a single reaction object, 
    but are instead shared between multiple reactions. 
    '''

    def __init__(self, name: str, value: float):
        self.name = name
        self.value = value
