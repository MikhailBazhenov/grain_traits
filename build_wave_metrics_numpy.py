import numpy as np
import pandas as pd

def nearest_neighbor_dists(A: np.ndarray, B: np.ndarray, chunk_size: int = 5000):
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    n = A.shape[0]
    m = B.shape[0]
    if n == 0 or m == 0:
        return np.empty((n,), dtype=np.float64), np.empty((n,), dtype=np.int64)

    B2 = np.sum(B * B, axis=1)  # (m,)

    best_d2 = np.full(n, np.inf, dtype=np.float64)
    best_idx = np.full(n, -1, dtype=np.int64)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        Ach = A[start:end]  # (k,2)
        A2 = np.sum(Ach * Ach, axis=1)  # (k,)

        d2 = A2[:, None] + B2[None, :] - 2.0 * (Ach @ B.T)
        np.maximum(d2, 0.0, out=d2)

        idx = np.argmin(d2, axis=1)
        min_d2 = d2[np.arange(end - start), idx]

        best_d2[start:end] = min_d2
        best_idx[start:end] = idx

    return np.sqrt(best_d2), best_idx


def compare_wave_to_base_numpy(df_points_new: pd.DataFrame, base_wave: int, other_wave: int,
                               chunk_size: int = 5000):
    base = df_points_new.loc[df_points_new["wave"] == base_wave, ["x", "y"]].to_numpy()
    other = df_points_new.loc[df_points_new["wave"] == other_wave, ["x", "y"]].to_numpy()

    if base.shape[0] < 2 or other.shape[0] < 2:
        return None

    dist_b2o, _ = nearest_neighbor_dists(base, other, chunk_size=chunk_size)  # base -> other
    dist_o2b, _ = nearest_neighbor_dists(other, base, chunk_size=chunk_size)  # other -> base

    all_dists = np.concatenate([dist_b2o, dist_o2b])
    mu = float(all_dists.mean())
    sd = float(all_dists.std(ddof=1)) if all_dists.size > 1 else 0.0
    cutoff = mu + 2.0 * sd

    keep_base = dist_b2o <= cutoff
    keep_other = dist_o2b <= cutoff

    base_f = base[keep_base]
    other_f = other[keep_other]

    if base_f.shape[0] < 2 or other_f.shape[0] < 2:
        return None

    # Means for the *kept* points (centers)
    x_mean_base = base_f[:, 0].mean()
    y_mean_base = base_f[:, 1].mean()
    x_mean_other = other_f[:, 0].mean()
    y_mean_other = other_f[:, 1].mean()

    # Std for the *kept* points
    sd_x_base = base_f[:, 0].std(ddof=1)
    sd_y_base = base_f[:, 1].std(ddof=1)
    sd_x_other = other_f[:, 0].std(ddof=1)
    sd_y_other = other_f[:, 1].std(ddof=1)

    sdx_ratio = np.nan if sd_x_other == 0 else (sd_x_base / sd_x_other)
    sdy_ratio = np.nan if sd_y_other == 0 else (sd_y_base / sd_y_other)

    return {
        "wave": other_wave,

        # center of valid points for THIS wavelength
        "x_mean": float(x_mean_other),
        "y_mean": float(y_mean_other),

        # comparison stats vs base
        "dx_mean": float(x_mean_base - x_mean_other),
        "dy_mean": float(y_mean_base - y_mean_other),
        "sdx_ratio": float(sdx_ratio) if np.isfinite(sdx_ratio) else np.nan,
        "sdy_ratio": float(sdy_ratio) if np.isfinite(sdy_ratio) else np.nan,

        # optional diagnostics
        "n_base": int(base.shape[0]),
        "n_other": int(other.shape[0]),
        "n_base_kept": int(base_f.shape[0]),
        "n_other_kept": int(other_f.shape[0]),
        "dist_mean": mu,
        "dist_sd": sd,
        "dist_cutoff": cutoff,
    }


def build_wave_metrics_numpy(df_points_new: pd.DataFrame, wavelength_list, base_wave: int = 1000,
                            chunk_size: int = 5000):
    rows = []

    # Add base row (1000 nm) so you can inspect x_mean/y_mean and see dx/dy=0
    base = df_points_new.loc[df_points_new["wave"] == base_wave, ["x", "y"]].to_numpy()
    if base.shape[0] >= 1:
        rows.append({
            "wave": base_wave,
            "x_mean": float(base[:, 0].mean()),
            "y_mean": float(base[:, 1].mean()),
            "dx_mean": 0.0,
            "dy_mean": 0.0,
            "sdx_ratio": 1.0,   # base/base
            "sdy_ratio": 1.0,
            "n_base": int(base.shape[0]),
            "n_other": int(base.shape[0]),
            "n_base_kept": int(base.shape[0]),
            "n_other_kept": int(base.shape[0]),
            "dist_mean": np.nan,
            "dist_sd": np.nan,
            "dist_cutoff": np.nan,
        })
    else:
        rows.append({"wave": base_wave, "x_mean": np.nan, "y_mean": np.nan,
                     "dx_mean": 0.0, "dy_mean": 0.0, "sdx_ratio": np.nan, "sdy_ratio": np.nan})

    for w in wavelength_list:
        if w == base_wave:
            continue

        out = compare_wave_to_base_numpy(df_points_new, base_wave, w, chunk_size=chunk_size)
        if out is None:
            rows.append({
                "wave": w,
                "x_mean": np.nan,
                "y_mean": np.nan,
                "dx_mean": np.nan,
                "dy_mean": np.nan,
                "sdx_ratio": np.nan,
                "sdy_ratio": np.nan,
            })
        else:
            rows.append(out)

    return pd.DataFrame(rows).sort_values("wave").reset_index(drop=True)


# ---- Usage ----
if __name__ == '__main__':

    wavelength_list = list(range(400, 1001, 10))
    df_metrics = build_wave_metrics_numpy(df_points_new, wavelength_list, base_wave=1000, chunk_size=5000)

    df_metrics
