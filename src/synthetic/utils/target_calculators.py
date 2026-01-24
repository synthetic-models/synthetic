"""
Target calculation methods for data generation.

This module provides two target calculation methods:
1. last_point: Returns the last time point value (default, backward compatible)
2. fold_change_drug: Returns fold change from drug start to end time

Both methods maintain backward compatibility with existing code.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


def get_drug_start_index(
    simulation_params: Dict[str, Any], drug_start_time: Optional[float] = None
) -> int:
    """
    Find the index of the drug start time in the time array.

    Args:
        simulation_params: Dictionary with 'start', 'end', 'points' keys
        drug_start_time: Time when drug treatment starts (default: midpoint)

    Returns:
        Index of the drug start time point in the time array

    Examples:
        >>> sim_params = {'start': 0, 'end': 10000, 'points': 101}
        >>> idx = get_drug_start_index(sim_params, drug_start_time=5000)
        >>> idx  # Returns index 50 (midpoint for 101 points)
    """
    start = simulation_params["start"]
    end = simulation_params["end"]
    points = simulation_params["points"]

    # Default drug start time to midpoint
    if drug_start_time is None:
        drug_start_time = (start + end) / 2

    # Generate time array
    time_array = np.linspace(start, end, points)

    # Find index closest to drug start time
    drug_start_index = np.argmin(np.abs(time_array - drug_start_time))

    return drug_start_index


def calculate_last_point(values: np.ndarray) -> float:
    """
    Calculate target as the last time point value.

    This is the default behavior and maintains backward compatibility.

    Args:
        values: Array of time series values

    Returns:
        The last value in the array

    Examples:
        >>> values = np.array([10.0, 12.0, 15.0, 18.0, 20.0])
        >>> calculate_last_point(values)
        20.0
    """
    return float(values[-1])


def calculate_fold_change_drug(
    values: np.ndarray, drug_start_index: int, eps: float = 1e-12, log: bool = False
) -> float:
    """
    Calculate target as fold change from drug start to end time.

    - If log is False (default): (end - drug_start) / drug_start
    - If log is True: log2(end / drug_start)

    Args:
        values: Array of time series values
        drug_start_index: Index of the drug start time in the array
        eps: Small value used to avoid division by zero / log(0)
        log: If True, return log2 fold change instead of linear fold change

    Returns:
        Fold change from drug start to end. If the drug start value is (close to) zero,
        eps is used to avoid inf/nan.

    Raises:
        ValueError: If log=True and either baseline or end value is negative.

    Examples:
        >>> values = np.array([10.0, 12.0, 15.0, 18.0, 20.0])
        >>> # Assuming drug starts at index 2 (value=15.0)
        >>> calculate_fold_change_drug(values, drug_start_index=2)
        0.3333333333333333  # (20 - 15) / 15
        >>> calculate_fold_change_drug(values, drug_start_index=2, log=True)
        0.41503749927884376  # log2(20 / 15)
    """
    drug_start_value = float(values[drug_start_index])
    end_value = float(values[-1])

    # Handle exact (or near) zeros consistently
    if np.isclose(drug_start_value, 0.0) and np.isclose(end_value, 0.0):
        return 0.0

    if log:
        # Log fold change requires non-negative values; clamp zeros to eps
        if drug_start_value < 0.0 or end_value < 0.0:
            raise ValueError("Log fold change requires non-negative values.")
        baseline = eps if np.isclose(drug_start_value, 0.0) else drug_start_value
        end = eps if np.isclose(end_value, 0.0) else end_value
        return float(np.log2(end / baseline))

    # Linear fold change: avoid division by zero by using eps for (near-)zero baseline
    if np.isclose(drug_start_value, 0.0):
        drug_start_value = float(eps)

    return (end_value - drug_start_value) / drug_start_value


def calculate_target_from_series(
    series: pd.Series,
    target_method: str = "last_point",
    simulation_params: Optional[Dict[str, Any]] = None,
    drug_start_time: Optional[float] = None,
) -> float:
    """
    Calculate target value from a time series using specified method.

    Args:
        series: Pandas Series containing time series values
        target_method: Method to use ('last_point' or 'fold_change_drug')
        simulation_params: Dictionary with 'start', 'end', 'points' keys (required for fold_change_drug)
        drug_start_time: Time when drug treatment starts (required for fold_change_drug)

    Returns:
        Calculated target value

    Raises:
        ValueError: If target_method is not recognized
        ValueError: If simulation_params is required but not provided

    Examples:
        >>> series = pd.Series([10.0, 12.0, 15.0, 18.0, 20.0])
        >>> sim_params = {'start': 0, 'end': 10000, 'points': 5}
        >>> # Using last_point (default)
        >>> calculate_target_from_series(series, 'last_point')
        20.0
        >>> # Using fold_change_drug (drug starts at index 2)
        >>> calculate_target_from_series(series, 'fold_change_drug', sim_params, drug_start_time=5000)
        0.3333333333333333
    """
    values = series.values

    if target_method == "last_point":
        return calculate_last_point(values)
    elif target_method == "fold_change_drug":
        if simulation_params is None:
            raise ValueError(
                "simulation_params must be provided for fold_change_drug method"
            )
        drug_start_index = get_drug_start_index(simulation_params, drug_start_time)
        return calculate_fold_change_drug(values, drug_start_index)
    else:
        raise ValueError(
            f"Unknown target_method: {target_method}. "
            f"Valid options are: 'last_point', 'fold_change_drug'"
        )


def calculate_targets(
    df: pd.DataFrame,
    target_method: str = "last_point",
    simulation_params: Optional[Dict[str, Any]] = None,
    drug_start_time: Optional[float] = None,
) -> pd.Series:
    """
    Calculate target values for each row (time series) in a DataFrame.

    Args:
        df: DataFrame where each row is a time series
        target_method: Method to use ('last_point' or 'fold_change_drug')
        simulation_params: Dictionary with 'start', 'end', 'points' keys (required for fold_change_drug)
        drug_start_time: Time when drug treatment starts (required for fold_change_drug)

    Returns:
        Series of calculated target values

    Examples:
        >>> df = pd.DataFrame({
        ...     'sample1': [10.0, 12.0, 15.0, 18.0, 20.0],
        ...     'sample2': [8.0, 10.0, 12.0, 14.0, 16.0]
        ... })
        >>> sim_params = {'start': 0, 'end': 10000, 'points': 5}
        >>> # Using last_point (default)
        >>> calculate_targets(df, 'last_point')
        sample1    20.0
        sample2    16.0
        dtype: float64
    """
    return df.apply(
        lambda row: calculate_target_from_series(
            row,
            target_method=target_method,
            simulation_params=simulation_params,
            drug_start_time=drug_start_time,
        )
    )


__all__ = [
    "get_drug_start_index",
    "calculate_last_point",
    "calculate_fold_change_drug",
    "calculate_target_from_series",
    "calculate_targets",
]
