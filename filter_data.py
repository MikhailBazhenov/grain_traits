# Проводим фильтрацию данных

def filter_data(in_file,
                out_file,
                wavelength_list,
                exclude_waves = [],
                replace_with_neighbours_average = []
                ):
    
    import numpy as np
    import pandas as pd
    import re

    # -----------------------
    # Inputs
    # -----------------------
    # wavelength_list = [...]          # исходный список длин волн
    wavelength_list_new = [x for x in wavelength_list if x not in exclude_waves]     # НОВЫЙ список длин волн, которые нужно оставить

    # -----------------------
    # 1) Read combined_df
    # -----------------------
    combined_df = pd.read_excel(in_file, index_col=0)

    # -----------------------
    # 2) Recalculate 500 nm columns using wavelength_list neighbors
    # -----------------------
    def recalc_bad_wavelength(df: pd.DataFrame, prefix: str, wl_bad: int, wl_prev: int, wl_next: int):
        bad = f"{prefix}_{wl_bad}"
        prev = f"{prefix}_{wl_prev}"
        nxt = f"{prefix}_{wl_next}"
        if bad in df.columns and prev in df.columns and nxt in df.columns:
            df[bad] = (
                pd.to_numeric(df[prev], errors="coerce") +
                pd.to_numeric(df[nxt], errors="coerce")
            ) / 2.0

    for bad_wave in replace_with_neighbours_average:

        if bad_wave in wavelength_list:
            idx = wavelength_list.index(bad_wave)
            if 0 < idx < len(wavelength_list) - 1:
                wl_prev = wavelength_list[idx - 1]
                wl_next = wavelength_list[idx + 1]

                recalc_bad_wavelength(combined_df, "mean",   bad_wave, wl_prev, wl_next)
                recalc_bad_wavelength(combined_df, "scaled", bad_wave, wl_prev, wl_next)
                recalc_bad_wavelength(combined_df, "CV",     bad_wave, wl_prev, wl_next)
            else:
                print("Note: " + str(bad_wave) + " at edge of wavelength_list, skipping recalculation.")
        else:
            print("Note: " + str(bad_wave) + "  not in wavelength_list, skipping recalculation.")

    # -----------------------
    # 3) Replace outliers with NA within each folder
    # -----------------------
    if "folder" not in combined_df.columns:
        raise ValueError("Expected 'folder' column not found.")

    exclude_cols = {"cell"}

    numeric_cols = combined_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    def filter_group_outliers(g: pd.DataFrame) -> pd.DataFrame:
        mu = g[numeric_cols].mean()
        sd = g[numeric_cols].std(ddof=1).replace(0, np.nan)

        mask = (g[numeric_cols] < (mu - 3 * sd)) | (g[numeric_cols] > (mu + 3 * sd))
        g.loc[:, numeric_cols] = g.loc[:, numeric_cols].mask(mask)
        return g

    filtered_df = combined_df.groupby("folder", group_keys=False).apply(filter_group_outliers)

    # -----------------------
    # 4) Select wavelengths using wavelength_list_new
    # -----------------------
    def select_wavelengths(df: pd.DataFrame, wl_keep: list[int]) -> pd.DataFrame:
        """
        Keep only spectral columns (mean_, scaled_, CV_) whose wavelength is in wl_keep.
        Non-spectral columns are kept untouched.
        """
        spectral_pattern = re.compile(r"^(mean|scaled|CV)_(\d+)$")

        cols_to_keep = []
        for c in df.columns:
            m = spectral_pattern.match(str(c))
            if m:
                wl = int(m.group(2))
                if wl in wl_keep:
                    cols_to_keep.append(c)
            else:
                cols_to_keep.append(c)

        return df.loc[:, cols_to_keep]

    filtered_df = select_wavelengths(filtered_df, wavelength_list_new)

    # -----------------------
    # 5) Save
    # -----------------------
    filtered_df.to_excel(out_file)
    print(f"Saved filtered dataframe to: {out_file}")

if __name__ == '__main__':
    in_file = main_folder + '\\' + "grain_data_combined.xlsx"
    out_file = main_folder + '\\' + "grain_data_filtered.xlsx"
    exclude_waves = list(range(400, 459, 10))
    filter_data(in_file=in_file, out_file=out_file, wavelength_list=wavelength_list, exclude_waves=exclude_waves, replace_with_neighbours_average=[500])
