from dataclasses import dataclass

@dataclass
class Regulation:
    """Class to store the regulation information"""
    from_specie: str
    to_specie: str
    reg_type: str
