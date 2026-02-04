
def smoothing(in_file,
              out_file,
              window_length = 15,
              polyorder = 2):

    import re
    import numpy as np
    import pandas as pd
    from scipy.signal import savgol_filter

    # Function to apply Savitzky-Golay smoothing
    def smooth_block(data_2d: np.ndarray, wlen: int, porder: int) -> np.ndarray:
        """
        Smooth along axis=1 (wavelength axis) row-by-row, handling NaNs by linear interpolation.
        Keeps NaNs at positions that were originally NaN (so we don't invent values).
        """
        out = data_2d.copy().astype(float)

        for i in range(out.shape[0]):
            y = out[i, :]
            nan_mask = np.isnan(y)

            # If too few points to smooth, skip
            finite_idx = np.where(~nan_mask)[0]
            if finite_idx.size < max(wlen, porder + 2):
                continue

            # Interpolate NaNs (only internally) so SavGol can run
            x = np.arange(y.size)
            y_interp = y.copy()
            y_interp[nan_mask] = np.interp(x[nan_mask], x[~nan_mask], y[~nan_mask])

            # Window length must be odd and <= length
            w = wlen
            if w > y.size:
                w = y.size if y.size % 2 == 1 else y.size - 1
            if w < (porder + 2):
                # minimal viable odd window
                w = porder + 2
                if w % 2 == 0:
                    w += 1
            if w > y.size:
                # still not possible
                continue

            y_smooth = savgol_filter(y_interp, window_length=w, polyorder=porder, mode="interp")

            # Restore NaNs where they were originally NaN
            y_smooth[nan_mask] = np.nan
            out[i, :] = y_smooth

        return out


    # Load the filtered data (uploaded file path)
    df = pd.read_excel(in_file, index_col=0)

    # Prefixes for wavelength-related columns
    prefixes = ['mean', 'CV', 'scaled']

    # Initialize dictionary for columns based on prefixes (fixing this part)
    cols_by_prefix = {p: [] for p in prefixes}

    # Loop over columns and categorize them correctly
    for c in df.columns:
        # Skip non-wavelength-related columns like 'folder', 'cell', etc.
        if not isinstance(c, str) or c in ['folder', 'cell']:
            continue
        # Match columns like 'mean_460', 'scaled_470', 'CV_500', etc.
        parsed = re.match(r"^(mean|CV|scaled)_(\d+)$", str(c))
        if parsed:
            pref, wl = parsed.groups()
            cols_by_prefix[pref].append(c)

    # Sort columns by wavelength for each prefix
    for p in prefixes:
        cols_by_prefix[p] = [c for wl, c in sorted([(int(wl), c) for c in cols_by_prefix[p]], key=lambda x: x[0])]

    # Apply Savitzky-Golay smoothing to each prefix block
    df_smoothed = df.copy()

    for pref in prefixes:
        cols = cols_by_prefix[pref]
        if not cols:
            continue

        # Convert selected columns to numeric and apply smoothing
        # Convert selected columns to numeric and explicitly cast to float64
        df_smoothed[cols] = df_smoothed[cols].apply(pd.to_numeric, errors="coerce").astype(np.float64)
        block = df_smoothed[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        smoothed = smooth_block(block, wlen=window_length, porder=polyorder)
        df_smoothed.loc[:, cols] = smoothed.astype(np.float64)

    # Save the smoothed data to a new file
    df_smoothed.to_excel(out_file)

    return out_file  # return the file path for user to download

if __name__ == '__main__':
    
    smoothing_in_file = main_folder + '\\' + "grain_data_filtered.xlsx"
    smoothing_out_file = main_folder + '\\' + "grain_data_smoothed.xlsx"

    smoothing_new(in_file=smoothing_in_file, out_file=smoothing_out_file, window_length=15, polyorder=2)
