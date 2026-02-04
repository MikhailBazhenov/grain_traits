import pandas as pd
import numpy as np

def scale_means_per_cell(
    df: pd.DataFrame,
    wavelength_list,
    mean_prefix: str = "mean_",
    scaled_prefix: str = "scaled_",
    low_wave: int = 460,
    high_wave: int = 1000,
) -> pd.DataFrame:
    """
    Scale mean_ values per row so that:
    mean_<low_wave> -> 0
    mean_<high_wave> -> 100

    Scaled values are stored as new columns: scaled_<wavelength>.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    wavelength_list : list
        List of wavelengths (e.g. [400, 410, ..., 1000]).
    mean_prefix : str
        Prefix for mean columns (default 'mean_').
    scaled_prefix : str
        Prefix for scaled columns (default 'scaled_').
    low_wave : int
        Wavelength that maps to 0 (default 460).
    high_wave : int
        Wavelength that maps to 100 (default 1000).

    Returns
    -------
    pd.DataFrame
        DataFrame with added scaled_ columns.
    """
    df = df.copy()

    low_col = f"{mean_prefix}{low_wave}"
    high_col = f"{mean_prefix}{high_wave}"

    if low_col not in df.columns or high_col not in df.columns:
        raise ValueError("Required reference wavelength columns not found.")

    denominator = df[high_col] - df[low_col]

    # Avoid division by zero
    denominator = denominator.replace(0, np.nan)

    for wave in wavelength_list:
        mean_col = f"{mean_prefix}{wave}"
        scaled_col = f"{scaled_prefix}{wave}"

        if mean_col not in df.columns:
            continue

        df[scaled_col] = (
            (df[mean_col] - df[low_col]) / denominator * 100
        )

    return df
