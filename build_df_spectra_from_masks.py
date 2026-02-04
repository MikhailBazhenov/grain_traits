

def build_df_spectra_from_masks(
    main_folder,
    wavelength_list,
    df_box_prompts,
    wave_for_segmentation="460",  # <-- must exist externally; kept here as default too
    rectified_subdir="Spectral_Cube_Rectified",
    masks_subdir="masks",
):
    """
    Uses masks ONLY from wavelength `wave_for_segmentation` (string, e.g. '460'):
        main_folder/Spectral_Cube_Rectified/masks/image460_cell000.png, ...

    Applies each cell mask to EVERY wavelength image in:
        main_folder/Spectral_Cube_Rectified/image{wavelength}.{ext}

    Computes for each cell and each wavelength:
        mean intensity over masked pixels only
        sd intensity over masked pixels only

    Returns:
        df_spectra with columns:
          cell,
          mean_400..mean_1000,
          sd_400..sd_1000
    """
    from pathlib import Path
    import cv2
    import numpy as np
    import pandas as pd

    base_dir = Path(main_folder) / rectified_subdir
    masks_dir = base_dir / masks_subdir

    if not base_dir.exists():
        raise FileNotFoundError(f"Images folder not found: {base_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks folder not found: {masks_dir}")

    # Cells present in prompts
    cells = (
        df_box_prompts["cell"]
        .dropna()
        .astype(int)
        .drop_duplicates()
        .sort_values()
        .to_list()
    )

    # --- Helpers ---
    def _find_image_path(wavelength: int) -> Path:
        candidates = [
            base_dir / f"image{wavelength}.jpg",
            base_dir / f"image{wavelength}.png",
            base_dir / f"image{wavelength}.tif",
            base_dir / f"image{wavelength}.tiff",
            base_dir / f"image{wavelength}.bmp",
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(f"No image found for wavelength {wavelength} in {base_dir}")

    def _read_gray(path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read: {path}")

        if img.ndim == 2:
            return img

        if img.ndim == 3:
            c = img.shape[2]
            if c == 1:
                return img[:, :, 0]
            if c == 3:
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if c == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        raise ValueError(f"Unsupported image shape {img.shape} for file: {path}")

    def _read_mask_2d(path: Path) -> np.ndarray:
        mk = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mk is None:
            raise ValueError(f"Failed to read mask: {path}")

        if mk.ndim == 2:
            pass
        elif mk.ndim == 3:
            c = mk.shape[2]
            if c == 1:
                mk = mk[:, :, 0]
            elif c == 3:
                mk = cv2.cvtColor(mk, cv2.COLOR_BGR2GRAY)
            elif c == 4:
                mk = cv2.cvtColor(mk, cv2.COLOR_BGRA2GRAY)
            else:
                raise ValueError(f"Unsupported mask channels: {mk.shape} for {path}")
        else:
            raise ValueError(f"Unsupported mask shape: {mk.shape} for {path}")

        if mk.dtype != np.uint8:
            mk = (mk > 0).astype(np.uint8) * 255

        return mk

    # --- Load all images once (per wavelength) ---
    images_by_wave = {}
    for wavelength in wavelength_list:
        img_path = _find_image_path(wavelength)
        images_by_wave[wavelength] = _read_gray(img_path)

    # Use one reference image to size-check masks (first wavelength in list)
    ref_img = images_by_wave[wavelength_list[0]]
    H, W = ref_img.shape[:2]

    # --- Load all cell masks once (from the constant wavelength) ---
    seg_wave_int = int(wave_for_segmentation)  # masks are named like image460_cell000.png
    masks_by_cell = {}

    for cell in cells:
        mp = masks_dir / f"image{seg_wave_int}_cell{cell:03d}.png"
        if not mp.exists():
            masks_by_cell[cell] = None
            continue

        mk = _read_mask_2d(mp)

        if mk.shape[:2] != (H, W):
            mk = cv2.resize(mk, (W, H), interpolation=cv2.INTER_NEAREST)

        if mk.ndim == 3 and mk.shape[2] == 1:
            mk = mk[:, :, 0]

        masks_by_cell[cell] = (mk > 0)  # store boolean 2D mask

    # --- Build spectra as list-of-lists (avoids fragmentation warning) ---
    # Row format: [cell, mean_400..mean_1000, sd_400..sd_1000]
    spectra = []

    for cell in cells:
        m = masks_by_cell.get(cell, None)

        # Prepare row
        row = [cell]

        # Means
        for wavelength in wavelength_list:
            img = images_by_wave[wavelength]
            if m is None or not np.any(m):
                row.append(np.nan)
            else:
                vals = img[m].astype(np.float64)
                row.append(float(vals.mean()) if vals.size else np.nan)

        # SDs
        for wavelength in wavelength_list:
            img = images_by_wave[wavelength]
            if m is None or not np.any(m):
                row.append(np.nan)
            else:
                vals = img[m].astype(np.float64)
                row.append(float(vals.std(ddof=1)) if vals.size > 1 else (0.0 if vals.size == 1 else np.nan))

        spectra.append(row)

    # --- Column names ---
    mean_cols = [f"mean_{w}" for w in wavelength_list]
    sd_cols = [f"sd_{w}" for w in wavelength_list]
    columns = ["cell"] + mean_cols + sd_cols

    df_spectra = pd.DataFrame(spectra, columns=columns)
    return df_spectra


# Example usage:
# main_folder = r'grain_11.12.2025_B39-1-1_ostatok7.68'
# wavelength_list = list(range(400, 1001, 10))
# wave_for_segmentation = '460'
# df_spectra = build_df_spectra_from_masks(main_folder, wavelength_list, df_box_prompts, wave_for_segmentation=wave_for_segmentation)


# Example usage:
if __name__ == '__main__':

    main_folder = r'grain_11.12.2025_B39-1-1_ostatok7.68'
    wavelength_list = list(range(400, 1001, 10))
    df_spectra = build_df_spectra_from_masks(main_folder, wavelength_list, df_box_prompts)

    df_spectra
