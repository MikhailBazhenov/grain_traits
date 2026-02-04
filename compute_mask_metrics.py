from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class ShapeMetrics:
    area_px2: float
    perimeter_px: float
    length_px: float
    width_px: float
    farthest_points_xy: Tuple[Tuple[float, float], Tuple[float, float]]  # (A, B)
    width_points_xy: Tuple[Tuple[float, float], Tuple[float, float]]    # (W1, W2)
    center_xy: Tuple[float, float]


def _load_binary_mask(mask_path: str | Path, threshold: int = 127) -> np.ndarray:
    mask_path = Path(mask_path)
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    _, bw = cv2.threshold(m, threshold, 255, cv2.THRESH_BINARY)
    return bw


def _largest_external_contour(binary_mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contours found in mask.")
    return max(contours, key=cv2.contourArea)  # (N,1,2)


def _farthest_pair_bruteforce(points_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    points_xy: (M,2) float array
    Returns: (A, B, max_dist)
    """
    if len(points_xy) < 2:
        raise ValueError("Need at least 2 points to compute farthest pair.")

    # Brute force on hull points (usually small). O(M^2).
    diffs = points_xy[:, None, :] - points_xy[None, :, :]
    d2 = np.sum(diffs * diffs, axis=2)  # (M,M)
    i, j = np.unravel_index(np.argmax(d2), d2.shape)
    A = points_xy[i].copy()
    B = points_xy[j].copy()
    return A, B, float(np.sqrt(d2[i, j]))


def _point_in_mask(mask: np.ndarray, x: float, y: float) -> bool:
    h, w = mask.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if xi < 0 or xi >= w or yi < 0 or yi >= h:
        return False
    return mask[yi, xi] > 0


def _nearest_contour_point(contour_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    # contour_xy: (N,2), target_xy: (2,)
    d2 = np.sum((contour_xy - target_xy[None, :]) ** 2, axis=1)
    return contour_xy[int(np.argmin(d2))].copy()


def _ray_boundary_intersection(
    mask: np.ndarray,
    start_xy: np.ndarray,
    dir_xy: np.ndarray,
    step: float = 0.5,
    max_steps: Optional[int] = None,
) -> np.ndarray:
    """
    March from start along dir until leaving the mask; return last inside point.
    Assumes start is inside; if not, will try to march until it enters then exits (still works often).
    """
    h, w = mask.shape[:2]
    if max_steps is None:
        max_steps = int(2 * np.hypot(h, w) / step) + 10

    x, y = float(start_xy[0]), float(start_xy[1])
    dx, dy = float(dir_xy[0]), float(dir_xy[1])

    last_inside = None
    inside = _point_in_mask(mask, x, y)

    for _ in range(max_steps):
        if inside:
            last_inside = (x, y)
        x += dx * step
        y += dy * step

        # If we go out of image bounds, stop
        if x < 0 or x >= w or y < 0 or y >= h:
            break

        inside = _point_in_mask(mask, x, y)

        # If we were inside and now outside, we just crossed boundary
        if last_inside is not None and not inside:
            break

    if last_inside is None:
        # Could not find an inside point along ray
        return start_xy.copy()

    return np.array(last_inside, dtype=np.float32)


def compute_mask_metrics(
    mask_path: str | Path,
    threshold: int = 127,
    pixel_size: Optional[float] = None,
) -> dict:
    """
    Computes:
      - area (px^2 or units^2)
      - perimeter (px or units)
      - length: largest distance between any two perimeter points (px or units)
      - width: at midpoint of the farthest pair, measure thickness along perpendicular ray (px or units)

    pixel_size:
      - If provided (e.g., mm per pixel), outputs additional scaled metrics.
    """
    mask = _load_binary_mask(mask_path, threshold=threshold)
    contour = _largest_external_contour(mask)

    # OpenCV contour format -> (N,2)
    contour_xy = contour.reshape(-1, 2).astype(np.float32)

    area_px2 = float(cv2.contourArea(contour))
    perimeter_px = float(cv2.arcLength(contour, closed=True))

    # Use convex hull for farthest pair
    hull = cv2.convexHull(contour_xy.astype(np.float32), returnPoints=True)
    hull_xy = hull.reshape(-1, 2).astype(np.float32)

    A, B, length_px = _farthest_pair_bruteforce(hull_xy)

    # Midpoint between farthest points
    C = (A + B) / 2.0

    # If midpoint is not inside mask (possible for concave shapes), snap to nearest contour point
    if not _point_in_mask(mask, C[0], C[1]):
        C = _nearest_contour_point(contour_xy, C)

    # Unit direction along length and its perpendicular
    v = (B - A).astype(np.float32)
    v_norm = float(np.hypot(v[0], v[1]))
    if v_norm < 1e-6:
        raise ValueError("Degenerate length vector (A and B too close).")
    d = v / v_norm
    p = np.array([-d[1], d[0]], dtype=np.float32)  # perpendicular

    # Intersections of perpendicular line through C with the object boundary
    W1 = _ray_boundary_intersection(mask, C, +p, step=0.5)
    W2 = _ray_boundary_intersection(mask, C, -p, step=0.5)
    width_px = float(np.hypot(*(W1 - W2)))

    metrics = ShapeMetrics(
        area_px2=area_px2,
        perimeter_px=perimeter_px,
        length_px=length_px,
        width_px=width_px,
        farthest_points_xy=((float(A[0]), float(A[1])), (float(B[0]), float(B[1]))),
        width_points_xy=((float(W1[0]), float(W1[1])), (float(W2[0]), float(W2[1]))),
        center_xy=(float(C[0]), float(C[1])),
    )

    out = [] # area, perimeter, length, width

    # Optional scaled metrics
    if pixel_size is not None:
        ps = float(pixel_size)
        out = [metrics.area_px2 * (ps ** 2), metrics.perimeter_px * ps, metrics.length_px * ps, metrics.width_px * ps]

    return out


if __name__ == "__main__":
    # Example:
    #   python metrics_from_mask.py
    # Adjust path and optionally pixel_size (e.g., mm/px)
    geometry = []
    for grain in range(len(df_box_prompts)):
        cell = int(df_box_prompts.iloc[grain]['cell'])
        mask_path = main_folder + '/Spectral_Cube_Rectified/masks/' + 'image' + wave_for_segmentation + '_cell' + str(cell).zfill(3) + '.png'
        result = compute_mask_metrics(mask_path, threshold=127, pixel_size=1)
        geometry.append([cell] + result)

    df_geometry = pd.DataFrame(geometry, columns=['cell', 'area', 'perimeter', 'length', 'width'])
    print(df_geometry)
    
