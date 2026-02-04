# Усредняем данные
import os

def averaging(wavelength_list,
              in_file,
              out_file):
    
    import re
    import numpy as np
    import pandas as pd

    try:
        from scipy.stats import t as t_dist
    except ImportError as e:
        raise ImportError(
            "This script needs scipy for t-distribution CI95. Install with: pip install scipy"
        ) from e


    # Assumed to exist in your environment:
    # folder_list = [...]
    
    combined_df = pd.read_excel(in_file, index_col=0)

    def ci95_t(series: pd.Series) -> float:
        """
        Returns a single +/- CI95 half-width using t-distribution.
        """
        x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
        n = x.size
        if n <= 1:
            return np.nan
        s = np.std(x, ddof=1)
        se = s / np.sqrt(n)
        tcrit = t_dist.ppf(0.975, df=n - 1)
        return float(tcrit * se)


    def rms(series: pd.Series) -> float:
        """
        Root Mean Square.
        """
        x = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
        if x.size == 0:
            return np.nan
        return float(np.sqrt(np.mean(x**2)))


    # -------------------------
    # 2) Rename / drop columns
    # -------------------------

    # Rename mean_460, mean_1000 -> Raw_460, Raw_1000 (if they exist)
    rename_map = {}
    if "mean_460" in combined_df.columns:
        rename_map["mean_460"] = "Raw_460"
    if "mean_1000" in combined_df.columns:
        rename_map["mean_1000"] = "Raw_1000"

    combined_df = combined_df.rename(columns=rename_map)

    # Rename scaled_{wl} -> T{wl}
    scaled_cols = [c for c in combined_df.columns if isinstance(c, str) and c.startswith("scaled_")]
    scaled_rename = {}
    for c in scaled_cols:
        m = re.match(r"^scaled_(\d+)$", c)
        if m:
            wl = int(m.group(1))
            scaled_rename[c] = f"T{wl}"

    combined_df = combined_df.rename(columns=scaled_rename)

    # Drop all remaining mean_ columns (but keep Raw_460/Raw_1000 which are no longer mean_)
    mean_cols_to_drop = [c for c in combined_df.columns if isinstance(c, str) and c.startswith("mean_")]
    combined_df = combined_df.drop(columns=mean_cols_to_drop, errors="ignore")


    # -------------------------
    # 3) Compute folder-level stats
    # -------------------------

    # Target mean+CI columns:
    geom_cols = ["area[mm²]", "perimeter[mm]", "length[mm]", "width[mm]"]
    raw_cols = ["Raw_460", "Raw_1000"]
    t_cols = [f"T{wl}" for wl in wavelength_list]

    mean_ci_cols = [c for c in (geom_cols + raw_cols + t_cols) if c in combined_df.columns]

    # CV columns -> RMS by folder
    cv_cols = [f"CV_{wl}" for wl in wavelength_list]

    # Compute means and CI95
    grp = combined_df.groupby("folder", dropna=False)

    means_df = grp[mean_ci_cols].mean(numeric_only=True)

    ci_df = grp[mean_ci_cols].apply(lambda g: pd.Series({col: ci95_t(g[col]) for col in mean_ci_cols}))
    ci_df = ci_df.rename(columns={c: f"{c}_CI95" for c in ci_df.columns})

    # Compute RMS for CV_ columns, output names "CV{wl}" (no underscore)
    cv_out = {}
    for c in cv_cols:
        m = re.match(r"^CV_(\d+)$", c)
        if m:
            wl = int(m.group(1))
            out_name = f"CV{wl}"
        else:
            # Fallback naming if column doesn't match CV_### exactly
            out_name = c.replace("CV_", "CV")
        cv_out[c] = out_name

    cv_rms_df = grp[cv_cols].apply(lambda g: pd.Series({cv_out[col]: rms(g[col]) for col in cv_cols}))


    # -------------------------
    # 4) Build output with required column order
    # -------------------------

    # Ordering rules:
    # - For area/perimeter/length/width and Raw_460/Raw_1000:
    #   mean column immediately followed by its _CI95 column
    # - For T columns:
    #   all T means first (in wavelength order), then all T _CI95 columns (same order)
    # - CV RMS columns:
    #   after all other columns, as CV{wavelength} order by wavelength if possible

    out_cols = []

    # Geom + Raw: mean then CI right after each
    for c in (geom_cols + raw_cols):
        if c in means_df.columns:
            out_cols.append(c)
            ci_name = f"{c}_CI95"
            if ci_name in ci_df.columns:
                out_cols.append(ci_name)

    # T means first
    for c in t_cols:
        if c in means_df.columns:
            out_cols.append(c)

    # then T CI95 columns in same order
    for c in t_cols:
        ci_name = f"{c}_CI95"
        if ci_name in ci_df.columns:
            out_cols.append(ci_name)

    # CV RMS columns after all other columns, ordered by wavelength if possible
    # First gather CV columns that match CV### and sort by numeric wavelength
    cv_sorted = []
    cv_unsorted = []
    for col in cv_rms_df.columns:
        m = re.match(r"^CV(\d+)$", str(col))
        if m:
            cv_sorted.append((int(m.group(1)), col))
        else:
            cv_unsorted.append(col)

    cv_sorted = [c for _, c in sorted(cv_sorted, key=lambda x: x[0])]

    out_cols.extend(cv_sorted)
    out_cols.extend(sorted(cv_unsorted))

    # Combine final output dataframe (index = folder)
    result_df = pd.concat([means_df, ci_df, cv_rms_df], axis=1)

    # Keep only columns we ordered (and that exist)
    result_df = result_df.loc[:, [c for c in out_cols if c in result_df.columns]]

    # Save
    
    result_df.to_excel(out_file)
    print("Saved: grain_data_averaged.xlsx")

if __name__ == '__main__':
    
    wavelength_list = list(range(460, 1001, 10)) # Для усреднения используем только часть надежных длин волн
    averaging_in_file = os.path.join(main_folder, "grain_data_smoothed.xlsx") # Используем сглаженные данные
    averaging_out_file = os.path.join(main_folder, "grain_data_averaged.xlsx")
    
    averaging(wavelength_list=wavelength_list, in_file=averaging_in_file, out_file=averaging_out_file)

