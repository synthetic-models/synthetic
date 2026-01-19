"""
Data processing utilities for synthetic data generation.

Contains reusable data transformation and DataFrame creation logic extracted 
from SyntheticGen.py to eliminate code duplication.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union


def create_feature_dataframe(data: List[Dict], columns: List[str] = None) -> pd.DataFrame:
    """
    Create a feature DataFrame from a list of dictionaries.
    
    Args:
        data: List of dictionaries containing feature data
        columns: Column names for the DataFrame (optional)
        
    Returns:
        DataFrame with feature data
    """
    df = pd.DataFrame(data)
    
    if columns is not None:
        # Ensure DataFrame has the specified columns
        for col in columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[columns]
    
    return df


def create_target_dataframe(data: List[Any], columns: Union[str, List[str]] = 'Cp') -> pd.DataFrame:
    """
    Create a target DataFrame from simulation results.
    
    Args:
        data: List of target values (scalars or arrays)
        columns: Column name(s) for the DataFrame
        
    Returns:
        DataFrame with target data
    """
    if not data:
        return pd.DataFrame(columns=columns if isinstance(columns, list) else [columns])
    
    # Handle both scalar and list data
    if isinstance(data[0], (int, float, np.number)):
        # Scalar data
        df = pd.DataFrame(data, columns=[columns] if isinstance(columns, str) else columns)
    else:
        # List/array data - assume each element corresponds to a column
        df = pd.DataFrame(data, columns=columns if isinstance(columns, list) else [columns])
    
    return df


def process_time_course_data(results: List[Dict], 
                           capture_species: Union[str, List[str]] = 'all',
                           model_spec=None) -> pd.DataFrame:
    """
    Process time course data from simulation results.
    
    Args:
        results: List of dictionaries containing time course data
        capture_species: Species to capture ('all' or list of species)
        model_spec: ModelSpecification object (required if capture_species='all')
        
    Returns:
        DataFrame with processed time course data
    """
    if not results:
        raise ValueError('No results to process')
    
    if capture_species == 'all' and model_spec is None:
        raise ValueError('model_spec is required when capture_species="all"')
    
    # Convert list of dictionaries to DataFrame
    df = pd.DataFrame(results)
    
    if df.empty:
        raise ValueError('Output dataframe is empty, check the model specifications and feature dataframe')
    
    return df


def extract_simulation_output(res: pd.DataFrame, 
                            outcome_var: str = 'Cp',
                            capture_time_course: bool = False) -> Union[float, np.ndarray, tuple]:
    """
    Extract simulation output from result DataFrame.
    
    Args:
        res: Simulation result DataFrame
        outcome_var: Variable to extract
        capture_time_course: Whether to return time course data
        
    Returns:
        Last value of outcome_var, or tuple with last value and time course
    """
    if outcome_var not in res.columns:
        raise ValueError(f'Outcome variable "{outcome_var}" not found in simulation results')
    
    last_value = res[outcome_var].iloc[-1]
    
    if capture_time_course:
        time_course = res[outcome_var].values
        return last_value, time_course
    else:
        return last_value


def create_species_time_course_dict(res: pd.DataFrame,
                                  species_list: List[str],
                                  include_phosphorylated: bool = True) -> Dict[str, np.ndarray]:
    """
    Create a dictionary of time course data for specified species.
    
    Args:
        res: Simulation result DataFrame
        species_list: List of species names
        include_phosphorylated: Whether to include phosphorylated versions
        
    Returns:
        Dictionary mapping species names to time course arrays
    """
    output = {}
    
    for species in species_list:
        if species in res.columns:
            output[species] = res[species].values
        
        if include_phosphorylated:
            phosphorylated = species + 'p'
            if phosphorylated in res.columns:
                output[phosphorylated] = res[phosphorylated].values
    
    return output


def normalize_dynamic_features(col_data: pd.Series, max_sim_time: int) -> Dict[str, float]:
    """
    Normalize dynamic features based on simulation time.
    
    Args:
        col_data: Time series data
        max_sim_time: Maximum simulation time
        
    Returns:
        Dictionary of normalized features
    """
    auc = np.trapz(col_data)
    n_auc = auc / max_sim_time
    
    max_time = np.argmax(col_data)
    n_max_time = max_time / max_sim_time
    
    min_time = np.argmin(col_data)
    n_min_time = min_time / max_sim_time
    
    return {
        'n_auc': n_auc,
        'n_max_time': n_max_time,
        'n_min_time': n_min_time
    }


def convert_to_series(data: Any) -> pd.Series:
    """
    Convert data to pandas Series for consistent processing.
    
    Args:
        data: Data to convert (array, list, or Series)
        
    Returns:
        pandas Series
    """
    if isinstance(data, pd.Series):
        return data
    else:
        return pd.Series(data)
