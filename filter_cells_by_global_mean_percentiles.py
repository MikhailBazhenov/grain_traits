# Функция для фильтрации темных и пересвеченных ячеек пустого планшета

import numpy as np
import pandas as pd

def filter_cells_by_global_mean_percentiles(
    df: pd.DataFrame,
    mean_prefix: str = "mean_",
    low_pct: float = 10.0,
    high_pct: float = 90.0,
    max_outside_pct: float = 90.0,
) -> pd.DataFrame:
    """
    Compute global percentiles over all mean_ values (all rows x all mean_ columns),
    then drop rows where > max_outside_pct of that row's mean_ values lie outside
    [low_pct, high_pct] percentiles.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing mean_XXX columns.
    mean_prefix : str
        Prefix used to identify mean columns (default 'mean_').
    low_pct : float
        Lower percentile (e.g., 10.0).
    high_pct : float
        Upper percentile (e.g., 90.0).
    max_outside_pct : float
        Drop a row if more than this percentage of its mean_ values are outside bounds.
        Example: 90.0 means drop if >90% are above high or below low.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe (rows kept).
    """
    if not (0 <= low_pct <= 100 and 0 <= high_pct <= 100 and 0 <= max_outside_pct <= 100):
        raise ValueError("Percent parameters must be between 0 and 100.")
    if low_pct >= high_pct:
        raise ValueError("low_pct must be < high_pct.")

    mean_cols = [c for c in df.columns if c.startswith(mean_prefix)]
    if not mean_cols:
        raise ValueError(f"No columns found starting with prefix '{mean_prefix}'.")

    # Convert to numeric safely; non-numeric becomes NaN
    mean_values = df[mean_cols].apply(pd.to_numeric, errors="coerce")

    # Global percentiles across ALL cells x wavelengths (ignore NaNs)
    flat = mean_values.to_numpy().ravel()
    low_val, high_val = np.nanpercentile(flat, [low_pct, high_pct])

    # For each row, compute % of mean_ values outside [low_val, high_val]
    outside = (mean_values.lt(low_val) | mean_values.gt(high_val))
    row_outside_pct = outside.mean(axis=1) * 100  # mean ignores NaN by default

    # Keep rows with <= threshold (drop strictly greater than threshold)
    keep_mask = row_outside_pct.le(max_outside_pct)

    return df.loc[keep_mask].copy()
