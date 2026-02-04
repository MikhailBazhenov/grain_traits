from pathlib import Path

import cv2
import numpy as np
from ultralytics import SAM  # pip install ultralytics


def segment_with_sam_bbox(
    image_path: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    cell: int,
    model_path: str = "sam2.1_s.pt",
) -> str:
    """
    Segments an object in `image_path` using a bounding-box prompt (x1,y1,x2,y2),
    saves a full-size binary mask to ./masks/<image_stem>_mask.png,
    and returns the saved mask path.

    Mask format:
      - Same width/height as original image
      - 8-bit PNG, values {0, 255}
    """
    image_path = Path(image_path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")

    h, w = img.shape[:2]

    # Normalize/validate bbox to xyxy within image bounds
    x1i, x2i = sorted([int(x1), int(x2)])
    y1i, y2i = sorted([int(y1), int(y2)])
    x1i = max(0, min(x1i, w - 1))
    x2i = max(0, min(x2i, w - 1))
    y1i = max(0, min(y1i, h - 1))
    y2i = max(0, min(y2i, h - 1))

    if x2i <= x1i or y2i <= y1i:
        raise ValueError(f"Invalid bbox after clamping: {(x1i, y1i, x2i, y2i)} for image (w={w}, h={h}).")

    # Create output directory ./masks next to the image
    masks_dir = image_path.parent / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Load SAM model
    model = SAM(model_path)

    # Run segmentation with bbox prompt
    # ultralytics SAM expects bboxes as [x1, y1, x2, y2] (xyxy) list(s)
    results = model(
        source=img,
        bboxes=[[x1i, y1i, x2i, y2i]],
        verbose=False,
    )

    # Build a full-size mask (HxW) from the best predicted mask
    full_mask = np.zeros((h, w), dtype=np.uint8)

    r0 = results[0]
    if getattr(r0, "masks", None) is None or r0.masks is None:
        out_path = masks_dir / (f"{image_path.stem}_cell" + str(cell).zfill(3) + ".png")
        cv2.imwrite(str(out_path), full_mask)
        return str(out_path)

    masks_data = r0.masks.data  # typically (N, H, W) torch tensor
    try:
        masks_np = masks_data.cpu().numpy()
    except Exception:
        masks_np = np.array(masks_data)

    if masks_np.ndim != 3 or masks_np.shape[1] != h or masks_np.shape[2] != w:
        raise RuntimeError(f"Unexpected mask shape: {masks_np.shape}, expected (N, {h}, {w}).")

    # Choose the mask with largest area (simple heuristic)
    areas = masks_np.reshape(masks_np.shape[0], -1).sum(axis=1)
    best_idx = int(np.argmax(areas))
    best_mask = masks_np[best_idx]

    full_mask = (best_mask > 0.5).astype(np.uint8) * 255

    out_path = masks_dir / (f"{image_path.stem}_cell" + str(cell).zfill(3) + ".png")
    cv2.imwrite(str(out_path), full_mask)
    return str(out_path)


if __name__ == "__main__":
    # Example usage:
    image_path = image_path
    for grain in range(len(df_box_prompts)):
        boxes = df_box_prompts.iloc[grain][['x_min', 'y_min', 'x_max', 'y_max']].to_list()
        cell = int(df_box_prompts.iloc[grain]['cell'])
        x1, y1, x2, y2 = boxes
        saved_mask_path = segment_with_sam_bbox(image_path, x1, y1, x2, y2, cell, model_path="sam2.1_s.pt")
        print("Saved mask:", saved_mask_path)
