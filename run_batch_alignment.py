import os
from pathlib import Path

import numpy as np
import cv2
import pandas as pd


# -----------------------------
# Core transforms
# -----------------------------
def shift_image(img: np.ndarray, dx: float, dy: float,
                border_mode=cv2.BORDER_CONSTANT, border_value=(0, 0, 0)) -> np.ndarray:
    """
    Shift image by (dx, dy) in pixels.
    Positive dx -> right, positive dy -> down.
    """
    h, w = img.shape[:2]
    M = np.array([[1.0, 0.0, float(dx)],
                  [0.0, 1.0, float(dy)]], dtype=np.float32)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=border_mode, borderValue=border_value)


def scale_about_point(img: np.ndarray, center_xy: tuple[float, float],
                      sx: float, sy: float,
                      output_size: tuple[int, int] | None = None,
                      border_mode=cv2.BORDER_CONSTANT, border_value=(0, 0, 0)) -> np.ndarray:
    """
    Anisotropic scale (sx, sy) about an arbitrary point (cx, cy) in pixel coordinates.

    This keeps the chosen center fixed in the output coordinate system (i.e., that pixel
    location corresponds to the same "point" after scaling), while pixels farther from
    center move according to the scale factors.

    If output_size is None, output matches input size (w, h).
    """
    h, w = img.shape[:2]
    if output_size is None:
        out_w, out_h = w, h
    else:
        out_w, out_h = output_size

    cx, cy = map(float, center_xy)

    # Mapping from source -> destination:
    # x' = sx*(x - cx) + cx
    # y' = sy*(y - cy) + cy
    #
    # For cv2.warpAffine we need destination -> source mapping,
    # so invert (diagonal inverse):
    # x = (x' - cx)/sx + cx
    # y = (y' - cy)/sy + cy
    if sx == 0 or sy == 0:
        raise ValueError("sx and sy must be non-zero.")

    M_inv = np.array([
        [1.0 / sx, 0.0, cx - cx / sx],
        [0.0, 1.0 / sy, cy - cy / sy],
    ], dtype=np.float32)

    return cv2.warpAffine(img, M_inv, (out_w, out_h), flags=cv2.INTER_LINEAR,
                          borderMode=border_mode, borderValue=border_value)


# -----------------------------
# Pipeline per wavelength row
# -----------------------------
def process_one_image(img: np.ndarray,
                      x_mean: float, y_mean: float,
                      dx_mean: float, dy_mean: float,
                      sdx_ratio: float, sdy_ratio: float,
                      border_mode=cv2.BORDER_CONSTANT, border_value=(0, 0, 0)) -> np.ndarray:
    """
    Steps:
      1) shift image by (dx_mean, dy_mean)
      2) compute new center = (x_mean + dx_mean, y_mean + dy_mean)
      3) INVERT scaling about that new center
         (enlarge <-> shrink)
    """

    # 1) shift
    shifted = shift_image(img, dx_mean, dy_mean,
                          border_mode=border_mode, border_value=border_value)

    # 2) new center
    new_center = (x_mean + dx_mean, y_mean + dy_mean)

    # 3) invert scaling direction
    sx_inv = 1.0 / sdx_ratio
    sy_inv = 1.0 / sdy_ratio

    scaled = scale_about_point(
        shifted,
        new_center,
        sx_inv,
        sy_inv,
        output_size=None,
        border_mode=border_mode,
        border_value=border_value
    )

    return scaled


# -----------------------------
# Batch runner
# -----------------------------
def run_batch_alignment(df: pd.DataFrame,
              main_folder: str | os.PathLike,
              subfolder: str = "Spectral_Cube",
              out_subfolder: str = "Spectral_Cube_Processed",
              ext_priority: tuple[str, ...] = (".png", ".jpg", ".jpeg"),
              keep_alpha_if_png: bool = True,
              border_mode=cv2.BORDER_CONSTANT,
              border_value=(0, 0, 0)) -> Path:
    """
    Expects df columns: wave, x_mean, y_mean, dx_mean, dy_mean, sdx_ratio, sdy_ratio.
    Notes:
      - You said 'y_min' is the y center; using it as y_mean here.
      - Images named 'imageNUMBER.png/jpg' where NUMBER == wave.
    """
    df_required = {"wave", "x_mean", "y_mean", "dx_mean", "dy_mean", "sdx_ratio", "sdy_ratio"}
    missing = df_required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {sorted(missing)}")

    main_folder = Path(main_folder)
    in_dir = main_folder / subfolder
    out_dir = main_folder / out_subfolder
    out_dir.mkdir(parents=True, exist_ok=True)

    def find_image_path(wave: int) -> Path | None:
        stem = f"image{int(wave)}"
        for ext in ext_priority:
            p = in_dir / f"{stem}{ext}"
            if p.exists():
                return p
        return None

    for _, row in df.iterrows():
        wave = int(row["wave"])
        x_mean = float(row["x_mean"])
        y_mean = float(row["y_mean"])  # per your description
        dx = float(row["dx_mean"])
        dy = float(row["dy_mean"])
        sx = float(row["sdx_ratio"])
        sy = float(row["sdy_ratio"])

        img_path = find_image_path(wave)
        if img_path is None:
            print(f"[WARN] Missing image for wave={wave} (looked for {ext_priority})")
            continue

        # Read image (preserve alpha if requested)
        if keep_alpha_if_png:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)  # can be HxWx4 for PNG
        else:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

        if img is None:
            print(f"[WARN] Failed to read: {img_path}")
            continue

        # If border_value is a 3-tuple but image has 4 channels, extend it
        bv = border_value
        if img.ndim == 3 and img.shape[2] == 4:
            if isinstance(border_value, (tuple, list)) and len(border_value) == 3:
                bv = tuple(border_value) + (0,)
            elif isinstance(border_value, (int, float)):
                bv = (border_value, border_value, border_value, 0)

        out = process_one_image(
            img, x_mean, y_mean, dx, dy, sx, sy,
            border_mode=border_mode, border_value=bv
        )

        # Save with same extension as input
        out_path = out_dir / img_path.name
        ok = cv2.imwrite(str(out_path), out)
        if not ok:
            print(f"[WARN] Failed to write: {out_path}")

    return out_dir


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # wavelength_list = list(range(400, 1001, 10))  # if you need it

    # df should already exist; otherwise load it:
    # df = pd.read_csv("your_table.csv")

    # main_folder = r"/path/to/main_folder"

    # Process all rows in df
    out_dir = run_batch_alignment(
        df=df_metrics,
        main_folder=main_folder,
        subfolder="Spectral_Cube",
        out_subfolder="Spectral_Cube_Processed",
        ext_priority=(".png", ".jpg", ".jpeg"),
        keep_alpha_if_png=True,
        border_mode=cv2.BORDER_CONSTANT,
        border_value=(0, 0, 0),  # fill color for newly exposed areas
    )

    print(f"Done. Output in: {out_dir}")
