# roi_real_processing.py
from __future__ import annotations
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence, Tuple

import cv2 as cv
import numpy as np

Point = Tuple[int, int]
Roi = Sequence[Point]
Milestone = Tuple[float, Roi]

# --------------------------
# Core helpers (ROI + thresh)
# --------------------------
def make_mask(shape: tuple[int, int], roi_pts: Optional[Roi]) -> np.ndarray:
    """shape=(H,W). If roi_pts is None, mask is all-ones (full image)."""
    H, W = shape
    mask = np.zeros((H, W), dtype=np.uint8)
    if roi_pts and len(roi_pts) >= 3:
        cv.fillPoly(mask, [np.array(roi_pts, dtype=np.int32)], 255)
    else:
        mask[:] = 255
    return mask


def process_roi(img_gray: np.ndarray, roi_pts: Optional[Roi], manual_thresh: Optional[int] = None) -> np.ndarray:
    """
    Apply thresholding inside ROI and invert only within the ROI.
    Returns uint8 image (0/255).
    """
    mask = make_mask(img_gray.shape, roi_pts)
    masked = cv.bitwise_and(img_gray, img_gray, mask=mask)

    # denoise (keep it minimal to preserve edges)
    # kernel size 1 is a no-op in OpenCV, use 3 for a tiny median smooth
    blurred = cv.medianBlur(masked, 3)

    # threshold
    if manual_thresh is None:
        _, th = cv.threshold(blurred, 0, 255, cv.THRESH_BINARY | cv.THRESH_OTSU)
    else:
        mt = int(np.clip(manual_thresh, 0, 255))
        _, th = cv.threshold(blurred, mt, 255, cv.THRESH_BINARY)

    # invert only inside ROI
    th_inv = th.copy()
    roi_idx = (mask == 255)
    th_inv[roi_idx] = 255 - th[roi_idx]
    return th_inv


# --------------------------
# ROI selection from milestones
# --------------------------
def select_roi_for_z(
    z: Optional[float],
    roi_milestones: Iterable[Milestone],
    layer_height: float,
    tol: Optional[float] = None,
) -> Optional[Roi]:
    """
    Pick the ROI whose milestone z is within tolerance of current z.
    Tolerance defaults to layer_height/2.
    """
    if z is None:
        return None
    if tol is None:
        tol = layer_height / 2.0

    best: Optional[Roi] = None
    best_delta = float("inf")
    for mz, roi in roi_milestones:
        try:
            delta = abs(float(mz) - float(z))
        except Exception:
            continue
        if delta <= tol and delta < best_delta:
            best = roi
            best_delta = delta
    return best


# --------------------------------------
# Public function 1:
# capture + process with assigned ROI
# --------------------------------------
def process_current_real_image(
    out_path: Path,
    *,
    capture_fn: Callable[[Path], bool],
    get_z_fn: Callable[[], Optional[float]],
    roi_milestones: Iterable[Milestone],
    layer_height: float,
    manual_thresh: Optional[int] = None,
    read_after_capture: bool = True,
) -> dict:
    """
    Captures a real image via `capture_fn`, picks ROI based on current Z from `get_z_fn`,
    thresholds+inverts within the ROI, writes the processed image to `out_path`,
    and returns an info dict.

    Returns:
      {
        "z": float|None,
        "roi_used": list[(x,y)]|None,
        "captured": bool,
        "saved_path": str|None
      }
    """
    z = get_z_fn()
    roi = select_roi_for_z(z, roi_milestones, layer_height)

    ok = capture_fn(out_path)
    if not ok:
        return {"z": z, "roi_used": roi, "captured": False, "saved_path": None}

    if not read_after_capture:
        # You saved the raw camera frame only
        return {"z": z, "roi_used": roi, "captured": True, "saved_path": str(out_path)}

    # Read what we just captured, process, and overwrite out_path with the processed image
    img_gray = cv.imread(str(out_path), cv.IMREAD_GRAYSCALE)
    if img_gray is None:
        return {"z": z, "roi_used": roi, "captured": True, "saved_path": None}

    processed = process_roi(img_gray, roi, manual_thresh=manual_thresh)
    cv.imwrite(str(out_path), processed)
    return {"z": z, "roi_used": roi, "captured": True, "saved_path": str(out_path)}


# --------------------------------------
# Public function 2:
# log-Hu moments + differential
# --------------------------------------
def _hu_log(bin_img: np.ndarray) -> np.ndarray:
    """
    Compute log-scaled Hu moments for a binary (0/255) image.
    Returns shape (7,) float64.
    """
    # Ensure binary for moments
    _, bin_img = cv.threshold(bin_img, 0, 255, cv.THRESH_BINARY)
    m = cv.moments(bin_img, binaryImage=True)
    hu = cv.HuMoments(m).flatten()

    # log transform with sign; add epsilon for stability
    eps = 1e-12
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + eps)
    return hu_log


def log_hu_moment_diff(
    slicer_img: np.ndarray,
    real_img: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Compute log-Hu moments for both slicer and real images and return:
      (hu_slicer, hu_real, abs_diff, L1_sum)

    Both inputs should be single-channel uint8 images where foreground is 255.
    """
    if slicer_img.ndim == 3:
        slicer_img = cv.cvtColor(slicer_img, cv.COLOR_BGR2GRAY)
    if real_img.ndim == 3:
        real_img = cv.cvtColor(real_img, cv.COLOR_BGR2GRAY)

    hu_s = _hu_log(slicer_img)
    hu_r = _hu_log(real_img)
    diff = np.abs(hu_s - hu_r)
    l1 = float(np.sum(diff))
    return hu_s, hu_r, diff, l1
