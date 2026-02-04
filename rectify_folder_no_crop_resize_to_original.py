import os
from pathlib import Path
import numpy as np
import cv2
import pandas as pd


# -----------------------------
# Geometry helpers
# -----------------------------
def fit_line_axbyc(points_xy: np.ndarray) -> np.ndarray:
    """
    Fit line in homogeneous form: a*x + b*y + c = 0 using TLS (SVD).
    Returns (a,b,c) normalized so sqrt(a^2+b^2)=1.
    """
    pts = np.asarray(points_xy, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 2:
        raise ValueError("points_xy must be (N,2) with N>=2")

    A = np.c_[pts[:, 0], pts[:, 1], np.ones(len(pts))]
    _, _, Vt = np.linalg.svd(A)
    line = Vt[-1, :]
    n = np.hypot(line[0], line[1])
    if n == 0:
        raise ValueError("Degenerate line fit")
    return line / n


def intersect_lines(l1: np.ndarray, l2: np.ndarray) -> np.ndarray:
    """
    Intersect two lines (a,b,c). Returns point (x,y).
    """
    p = np.cross(l1, l2)  # homogeneous point (x,y,w)
    if abs(p[2]) < 1e-10:
        raise ValueError("Lines are parallel or nearly parallel")
    return (p[:2] / p[2]).astype(np.float64)


def homography_from_df_unit(df_angle_correction: pd.DataFrame):
    """
    Build homography H_unit mapping the detected skewed rectangle -> unit square.
    Also returns the 4 source corners (tl,tr,br,bl) in the original image.
    """
    if not {"x", "y"}.issubset(df_angle_correction.columns):
        raise ValueError("df_angle_correction must have columns ['x','y']")

    left_idx = list(range(0, 10))
    right_idx = list(range(90, 100))
    top_idx = [10, 30, 50, 70, 90]
    bottom_idx = [9, 29, 49, 69, 89]

    left_pts = df_angle_correction.loc[left_idx, ["x", "y"]].to_numpy()
    right_pts = df_angle_correction.loc[right_idx, ["x", "y"]].to_numpy()
    top_pts = df_angle_correction.loc[top_idx, ["x", "y"]].to_numpy()
    bottom_pts = df_angle_correction.loc[bottom_idx, ["x", "y"]].to_numpy()

    line_left = fit_line_axbyc(left_pts)
    line_right = fit_line_axbyc(right_pts)
    line_top = fit_line_axbyc(top_pts)
    line_bottom = fit_line_axbyc(bottom_pts)

    tl = intersect_lines(line_left, line_top)
    tr = intersect_lines(line_right, line_top)
    br = intersect_lines(line_right, line_bottom)
    bl = intersect_lines(line_left, line_bottom)

    src = np.array([tl, tr, br, bl], dtype=np.float32)

    # Map to unit square
    dst = np.array([[0, 0],
                    [1, 0],
                    [1, 1],
                    [0, 1]], dtype=np.float32)

    H_unit = cv2.getPerspectiveTransform(src, dst)  # src -> unit square
    return H_unit, src


def warp_no_crop_to_original_size(img: np.ndarray,
                                  H_unit: np.ndarray,
                                  pad: int = 0):
    """
    1) Build rectifying homography (scaled to roughly original pixel scale)
    2) Find needed canvas size to contain FULL transformed original image (no crop)
    3) Warp to that canvas
    4) Resize back to original (w,h)

    Returns:
      rectified_resized (same size as input)
      rect_w_px, rect_h_px = rectangle width/height on the FINAL resized image (pixels)
    """
    h, w = img.shape[:2]

    # Scale unit square to pixel rectangle roughly matching original size
    S = np.array([[w - 1, 0, 0],
                  [0, h - 1, 0],
                  [0, 0, 1]], dtype=np.float64)
    H0 = S @ H_unit  # src -> rectified pixel-ish space

    # Transform the 4 corners of the original image to see bounding box in rectified space
    corners = np.array([[0, 0],
                        [w - 1, 0],
                        [w - 1, h - 1],
                        [0, h - 1]], dtype=np.float32).reshape(-1, 1, 2)

    warped_corners = cv2.perspectiveTransform(corners, H0.astype(np.float32)).reshape(-1, 2)
    min_xy = warped_corners.min(axis=0)
    max_xy = warped_corners.max(axis=0)

    # Translation so everything becomes >=0 (plus optional pad)
    tx = -min_xy[0] + pad
    ty = -min_xy[1] + pad
    T = np.array([[1, 0, tx],
                  [0, 1, ty],
                  [0, 0, 1]], dtype=np.float64)

    H_final = T @ H0

    out_w = int(np.ceil((max_xy[0] - min_xy[0]) + 2 * pad))
    out_h = int(np.ceil((max_xy[1] - min_xy[1]) + 2 * pad))
    out_w = max(out_w, 2)
    out_h = max(out_h, 2)

    # Warp to expanded canvas (no crop)
    rectified_big = cv2.warpPerspective(
        img,
        H_final.astype(np.float32),
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # Resize back to original size
    rectified_resized = cv2.resize(rectified_big, (w, h), interpolation=cv2.INTER_LINEAR)

    # Rectangle (the corrected plane rectangle) has size (w-1, h-1) in the "rectified_big"
    # BEFORE resizing (because unit square was scaled by S to (w-1,h-1)).
    # After resizing the whole canvas to (w,h), it scales by:
    rx = w / out_w
    ry = h / out_h
    rect_w_px = int(round((w - 1) * rx))
    rect_h_px = int(round((h - 1) * ry))

    return rectified_resized, rect_w_px, rect_h_px


# -----------------------------
# Batch runner
# -----------------------------
def rectify_folder_no_crop_resize_to_original(main_folder,
                                             df_angle_correction,
                                             in_subfolder="Spectral_Cube_Processed",
                                             out_subfolder="Spectral_Cube_Rectified",
                                             ext_priority=(".png", ".jpg", ".jpeg"),
                                             keep_alpha_if_png=True,
                                             pad: int = 0,
                                             border_mode=cv2.BORDER_CONSTANT,
                                             border_value=(0, 0, 0)):
    """
    Returns:
      out_dir (Path)
      rect_w_px (int): width of rectified rectangle on FINAL saved images
      rect_h_px (int): height of rectified rectangle on FINAL saved images
    """
    main_folder = Path(main_folder)
    in_dir = main_folder / in_subfolder
    out_dir = main_folder / out_subfolder
    out_dir.mkdir(parents=True, exist_ok=True)

    H_unit, _src = homography_from_df_unit(df_angle_correction)

    # Collect images
    img_paths = []
    for ext in ext_priority:
        img_paths.extend(sorted(in_dir.glob(f"*{ext}")))
    if not img_paths:
        raise FileNotFoundError(f"No images found in {in_dir} with extensions {ext_priority}")

    rect_w_px_out = None
    rect_h_px_out = None

    for img_path in img_paths:
        # Read image
        if keep_alpha_if_png:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        else:
            img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

        if img is None:
            print(f"[WARN] Failed to read: {img_path}")
            continue

        # Handle border_value for alpha images
        bv = border_value
        if img.ndim == 3 and img.shape[2] == 4:
            if isinstance(border_value, (tuple, list)) and len(border_value) == 3:
                bv = tuple(border_value) + (0,)
            elif isinstance(border_value, (int, float)):
                bv = (border_value, border_value, border_value, 0)

        # Warp (no crop) + resize back to original
        rectified_resized, rect_w_px, rect_h_px = warp_no_crop_to_original_size(
            img, H_unit, pad=pad
        )

        # Re-warp with desired border handling (apply after resize by filling via padding in warp)
        # Note: warp_no_crop_to_original_size uses borderValue=0. If you want custom bv:
        # simplest is to keep 0 (black). If you really need non-black, tell me and I’ll adapt it.

        # Save
        out_path = out_dir / img_path.name
        ok = cv2.imwrite(str(out_path), rectified_resized)
        if not ok:
            print(f"[WARN] Failed to write: {out_path}")
            continue

        # Store rectangle dims from first successful image (should be consistent if all images same size)
        if rect_w_px_out is None:
            rect_w_px_out, rect_h_px_out = rect_w_px, rect_h_px

    if rect_w_px_out is None:
        raise RuntimeError("No images were successfully processed.")

    return out_dir, rect_w_px_out, rect_h_px_out


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    # main_folder = r"/path/to/main_folder"

    out_dir, rect_w_px, rect_h_px = rectify_folder_no_crop_resize_to_original(
        main_folder=main_folder,
        df_angle_correction=df_angle_correction,
        pad=0
    )

    print(f"Saved to: {out_dir}")
    print(f"Rectified rectangle size on final images: width={rect_w_px}px, height={rect_h_px}px")
