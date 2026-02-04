import os
from pathlib import Path
import cv2
import numpy as np


def _find_existing_image(folder: Path, stem: str, exts=(".png", ".jpg", ".jpeg")) -> Path | None:
    for ext in exts:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def _read_bgr_3ch(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.ndim == 3:
        c = img.shape[2]
        if c == 1:
            img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
        elif c == 3:
            pass
        elif c == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            raise ValueError(f"Unsupported channel count {c} for {path}: shape={img.shape}")
    else:
        raise ValueError(f"Unsupported image ndim {img.ndim} for {path}: shape={img.shape}")

    return img


def _pad_to_size(img: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
    h, w = img.shape[:2]
    if (w, h) == (target_w, target_h):
        return img

    scale = min(target_w / w, target_h / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top
    left = (target_w - new_w) // 2
    right = target_w - new_w - left
    return cv2.copyMakeBorder(resized, top, bottom, left, right,
                              borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))


def _compose_side_by_side(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    H = max(ha, hb)

    if ha != H:
        a = cv2.copyMakeBorder(a, (H - ha) // 2, H - ha - (H - ha) // 2, 0, 0,
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))
    if hb != H:
        b = cv2.copyMakeBorder(b, (H - hb) // 2, H - hb - (H - hb) // 2, 0, 0,
                               cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return np.hstack([a, b])


def _overlay_labels(frame: np.ndarray, wave: int) -> np.ndarray:
    out = frame.copy()
    H, W = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(out, f"wave = {wave} nm", (15, 35), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, "Original", (15, H - 15), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, "Processed", (W // 2 + 15, H - 15), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _try_open_writer(out_path: Path, fps: float, frame_size: tuple[int, int]):
    W, H = frame_size
    candidates = [
        (out_path.with_suffix(".mp4"), "avc1"),
        (out_path.with_suffix(".mp4"), "mp4v"),
        (out_path.with_suffix(".mp4"), "H264"),
        (out_path.with_suffix(".avi"), "MJPG"),
        (out_path.with_suffix(".avi"), "XVID"),
    ]

    for p, fourcc_str in candidates:
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        writer = cv2.VideoWriter(str(p), fourcc, float(fps), (W, H), True)
        ok = writer.isOpened()
        print(f"[VideoWriter] try {p.name} codec={fourcc_str} opened={ok}")
        if ok:
            return writer, p, fourcc_str
        writer.release()

    raise RuntimeError("Could not open VideoWriter with any tried codec/container combination.")


def make_comparison_video(
    main_folder: str | os.PathLike,
    in_subfolder: str = "Spectral_Cube",
    out_subfolder: str = "Spectral_Cube_Processed",
    video_name: str = "comparison",
    fps: int = 10,
    hold_frames: int = 1,
    wavelength_list=range(400, 1001, 10),
    exts=(".png", ".jpg", ".jpeg"),
    target_frame_size: tuple[int, int] | None = None,
):
    main_folder = Path(main_folder)
    in_dir = main_folder / in_subfolder
    proc_dir = main_folder / out_subfolder

    pairs = []
    for wave in wavelength_list:
        stem = f"image{int(wave)}"
        p1 = _find_existing_image(in_dir, stem, exts)
        p2 = _find_existing_image(proc_dir, stem, exts)
        if p1 and p2:
            pairs.append((int(wave), p1, p2))

    if not pairs:
        raise RuntimeError(f"No matching image pairs found.\nIN:  {in_dir}\nOUT: {proc_dir}\nEXTS: {exts}")

    w0, p10, p20 = pairs[0]
    a0 = _read_bgr_3ch(p10)
    b0 = _read_bgr_3ch(p20)
    composed0 = _compose_side_by_side(a0, b0)
    composed0 = _overlay_labels(composed0, w0)

    if target_frame_size is None:
        H0, W0 = composed0.shape[:2]
        frame_size = (W0, H0)
    else:
        frame_size = (int(target_frame_size[0]), int(target_frame_size[1]))

    out_path_base = main_folder / video_name
    writer, used_path, used_fourcc = _try_open_writer(out_path_base, fps, frame_size)

    W, H = frame_size
    frames_written = 0

    try:
        for wave, p1, p2 in pairs:
            a = _read_bgr_3ch(p1)
            b = _read_bgr_3ch(p2)

            frame = _compose_side_by_side(a, b)
            frame = _overlay_labels(frame, wave)

            frame = _pad_to_size(frame, W, H)

            # enforce 3-ch just in case
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.ndim == 3 and frame.shape[2] == 1:
                frame = cv2.cvtColor(frame[:, :, 0], cv2.COLOR_GRAY2BGR)

            for _ in range(max(1, int(hold_frames))):
                writer.write(frame)
                frames_written += 1

    finally:
        writer.release()

    size_bytes = used_path.stat().st_size if used_path.exists() else 0
    print(f"[DONE] codec={used_fourcc} frames={frames_written} file={used_path} size={size_bytes} bytes")

    if frames_written == 0 or size_bytes < 1024:
        raise RuntimeError(
            f"Video seems empty (frames={frames_written}, size={size_bytes} bytes). "
            f"Try AVI output (MJPG/XVID) or ensure your OpenCV build has video codecs."
        )

    return used_path


if __name__ == "__main__":
    # main_folder = r"/path/to/main_folder"

    out_video = make_comparison_video(
        main_folder=main_folder,
        in_subfolder="Spectral_Cube",
        out_subfolder="Spectral_Cube_Rectified",
        video_name="comparison",
        fps=10,
        hold_frames=2,
        wavelength_list=range(400, 1001, 10),
    )
    print(f"Saved video: {out_video}")
