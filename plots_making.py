# Строим графики

def plots_making(input_file, out_folder):

    import os
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # -----------------------
    # User-defined variables
    # -----------------------
    
    # -----------------------
    # Load data
    # -----------------------
    df = pd.read_excel(input_file, index_col=0)
    df.index.name = df.index.name or "folder"
    folders = df.index.astype(str).tolist()

    # Color cycle
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # -----------------------
    # Helpers
    # -----------------------
    def get_wavelength_cols(prefix, columns):
        """
        prefix = 'T' or 'CV'
        Returns sorted wavelengths and column map
        """
        pat = re.compile(rf"^{prefix}(\d+)$")
        out = {}
        for c in columns:
            m = pat.match(str(c))
            if m:
                wl = int(m.group(1))
                out[wl] = c
        return sorted(out), out


    def get_mean_cols(columns):
        """
        mean_400 ... mean_1000
        """
        pat = re.compile(r"^Raw_(\d+)$")
        out = {}
        for c in columns:
            m = pat.match(str(c))
            if m:
                wl = int(m.group(1))
                out[wl] = c
        return sorted(out), out


    # ---------------------------------------------
    # 1) Barplots with CI (geom columns)
    # ---------------------------------------------
    geom_cols = ["area[mm²]", "perimeter[mm]", "length[mm]", "width[mm]"]

    for col in geom_cols:
        ci_col = f"{col}_CI95"
        if col not in df.columns:
            continue

        x = np.arange(len(folders))
        y = df[col].to_numpy(dtype=float)
        yerr = df[ci_col].to_numpy(dtype=float) if ci_col in df.columns else None

        plt.figure()
        plt.bar(x, y)
        if yerr is not None:
            plt.errorbar(x, y, yerr=yerr, fmt="none", capsize=4, ecolor='k')

        plt.xticks(x, folders, rotation=45, ha="right")
        plt.ylabel(col)
        plt.title(f"{col} (mean ± CI95)")
        plt.tight_layout()

        fname = f"bar_{col.replace('/', '_').replace('[','').replace(']','').replace('²','2')}.png"
        plt.savefig(os.path.join(out_folder, fname), dpi=300)
        plt.close()


    # -------------------------------------------------------
    # 2) mean_400 ... mean_1000 barplots (all folders together)
    # -------------------------------------------------------
    mean_wls, mean_map = get_mean_cols(df.columns)

    if mean_wls:
        plt.figure(figsize=(12, 4))

        for i, folder in enumerate(folders):
            y = [df.loc[folder, mean_map[wl]] for wl in mean_wls]
            ci = [
                df.loc[folder, f"{mean_map[wl]}_CI95"]
                if f"{mean_map[wl]}_CI95" in df.columns else np.nan
                for wl in mean_wls
            ]

            xpos = np.arange(len(mean_wls)) + i * 0.8 / len(folders)
            plt.bar(
                xpos,
                y,
                width=0.8 / len(folders),
                yerr=ci,
                capsize=2,
                label=folder,
                color=colors[i % len(colors)]
            )

        plt.xticks(np.arange(len(mean_wls)), mean_wls, rotation=90)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("mean")
        plt.title("Mean spectrum (mean ± CI95)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, "mean_spectrum_bar_all_folders.png"), dpi=300)
        plt.close()


    # ---------------------------------------------------
    # 3) T columns – line plots with CI ribbons (combined)
    # ---------------------------------------------------
    t_wls, t_map = get_wavelength_cols("T", df.columns)

    if t_wls:
        x = np.array(t_wls, dtype=float)
        plt.figure()

        for i, folder in enumerate(folders):
            y = np.array([df.loc[folder, t_map[wl]] for wl in t_wls], dtype=float)
            ci = np.array([
                df.loc[folder, f"{t_map[wl]}_CI95"]
                if f"{t_map[wl]}_CI95" in df.columns else np.nan
                for wl in t_wls
            ], dtype=float)

            color = colors[i % len(colors)]
            plt.plot(x, y, label=folder, color=color)

            if np.isfinite(ci).any():
                plt.fill_between(x, y - ci, y + ci, alpha=0.2, color=color)

        plt.xlabel("Wavelength (nm)")
        plt.ylabel("T")
        plt.title("T spectra (mean ± CI95)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, "T_spectra_all_folders.png"), dpi=300)
        plt.close()


    # ------------------------------------------
    # 4) CV columns – line plots (no CI ribbons)
    # ------------------------------------------
    cv_wls, cv_map = get_wavelength_cols("CV", df.columns)

    if cv_wls:
        x = np.array(cv_wls, dtype=float)
        plt.figure()

        for i, folder in enumerate(folders):
            y = np.array([df.loc[folder, cv_map[wl]] for wl in cv_wls], dtype=float)
            plt.plot(x, y, label=folder, color=colors[i % len(colors)])

        plt.xlabel("Wavelength (nm)")
        plt.ylabel("CV (RMS)")
        plt.title("CV spectra (RMS)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, "CV_spectra_all_folders.png"), dpi=300)
        plt.close()


    print("All plots created and saved to main_folder.")

if __name__ == '__main__':
    
    plots_input_file = os.path.join(main_folder, "grain_data_averaged.xlsx")

    plots_making(input_file=plots_input_file, out_folder=main_folder)

