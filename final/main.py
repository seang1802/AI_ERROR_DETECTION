# main.py
import QIDI_Z_AXIS
import face_view
import img_process
from pathlib import Path
from typing import Optional, Tuple, List
import time
import datetime
import cv2 as cv
from picamera2 import Picamera2

# ------------------------------
# G-code & layer height
# ------------------------------
_gcode_candidates = list(Path(".").glob("**/*.gcode"))
GCODE_PATH = _gcode_candidates[0] if _gcode_candidates else None
LAYER_HEIGHT = face_view.get_layer_height(GCODE_PATH, fallback=0.20) if GCODE_PATH else 0.20

# ------------------------------
# Current Z (best effort)
# ------------------------------
try:
    CURR_Z = QIDI_Z_AXIS.get_z_height()
except Exception:
    CURR_Z = None


def layer_from_z(z: Optional[float], lh: float) -> Optional[int]:
    """Map a Z height to the nearest layer index (robust rounding)."""
    if z is None or lh <= 0:
        return None
    return int((z + 0.5 * lh) // lh)


CURR_LAYER = layer_from_z(CURR_Z, LAYER_HEIGHT)

# ------------------------------
# Globals
# ------------------------------
CURRENT_ROI: Optional[List[Tuple[int, int]]] = None
LAST_RENDERED_LAYER: Optional[int] = None
LAST_RENDER_INFO: Optional[dict] = None
LAST_CAPTURE_INFO: Optional[dict] = None
LAST_CAPTURED_LAYER: Optional[int] = None  # guard to avoid duplicate capture

# ------------------------------
# Camera capture
# ------------------------------
def capture_image(out_path: Path, width: int = 1280, height: int = 720, settle_s: float = 1.0) -> bool:
    """
    Capture a single image from the Raspberry Pi Camera using Picamera2.
    Saves BGR .png/.jpg to out_path. Returns True on success.
    """
    picam = None
    try:
        picam = Picamera2()
        picam.configure(
            picam.create_still_configuration(
                main={"size": (width, height), "format": "RGB888"}
            )
        )
        picam.start()
        time.sleep(settle_s)  # allow AE/AG to settle

        frame_rgb = picam.capture_array()
        if frame_rgb is None:
            return False

        frame_bgr = cv.cvtColor(frame_rgb, cv.COLOR_RGB2BGR)
        cv.imwrite(str(out_path), frame_bgr)
        return True

    except Exception:
        return False

    finally:
        try:
            if picam is not None:
                picam.stop()
                if hasattr(picam, "close"):
                    picam.close()
        except Exception:
            pass


# ------------------------------
# ROI selection (with wide tolerance)
# ------------------------------
def find_roi_for_z(z: float, tol: float = None) -> Optional[List[Tuple[int, int]]]:
    if z is None:
        return None

    # Default tolerance: ±0.2 mm (one layer height for 0.2mm)
    if tol is None:
        tol = 0.2

    best = None
    best_delta = float("inf")
    for mz, roi in ROI_MILESTONES:
        try:
            delta = abs(float(mz) - float(z))
        except Exception:
            continue
        if delta <= tol and delta < best_delta:
            best = roi
            best_delta = delta
    return best


# ------------------------------
# Poller (core): capture+ROI+render+HU once per match
# ------------------------------
def monitor_roi(
    poll_interval: float = 1.0,
    tol: float = None,
    max_attempts: Optional[int] = None,
    render_on_match: bool = True,
    out_dir: Optional[Path] = None,
    manual_thresh: Optional[int] = None,
) -> Optional[List[Tuple[int, int]]]:

    global CURR_Z, CURR_LAYER, CURRENT_ROI
    global LAST_RENDERED_LAYER, LAST_RENDER_INFO, LAST_CAPTURE_INFO, LAST_CAPTURED_LAYER

    attempts = 0
    if out_dir is None:
        out_dir = Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ensure tolerance is applied here too
    if tol is None:
        tol = 0.2

    while True:
        if max_attempts is not None and attempts >= max_attempts:
            return None
        attempts += 1

        z = QIDI_Z_AXIS.get_z_height()
        CURR_Z = z
        CURR_LAYER = layer_from_z(CURR_Z, LAYER_HEIGHT)

        print(f"[DEBUG] Z={CURR_Z if CURR_Z is not None else 'None'}  tol=±{tol}  layer={CURR_LAYER}")

        roi = find_roi_for_z(CURR_Z, tol=tol)
        if roi is not None:
            CURRENT_ROI = roi

            # Prevent capturing the same layer twice
            if CURR_LAYER is not None and LAST_CAPTURED_LAYER == CURR_LAYER:
                time.sleep(poll_interval)
                continue

            LAST_CAPTURED_LAYER = CURR_LAYER

            # ----------------------
            # 1) Capture RAW and apply ROI HERE (no re-read of Z)
            # ----------------------
            if render_on_match and CURR_LAYER is not None:
                try:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    raw_path  = out_dir / f"real_raw_L{CURR_LAYER}_{ts}.png"
                    proc_path = out_dir / f"real_processed_L{CURR_LAYER}_{ts}.png"

                    ok = capture_image(raw_path)
                    if not ok:
                        LAST_CAPTURE_INFO = {"path": None, "success": False}
                    else:
                        img_gray = cv.imread(str(raw_path), cv.IMREAD_GRAYSCALE)
                        if img_gray is None:
                            LAST_CAPTURE_INFO = {"path": None, "success": False}
                        else:
                            # decide threshold to use
                            if manual_thresh is None:
                                mt = 140  # default if caller doesn't specify
                            else:
                                mt = int(max(0, min(255, manual_thresh)))

                            print(f"[DEBUG] Using manual_thresh={mt}")

                            processed = img_process.process_roi(
                                img_gray,
                                roi_pts=roi,
                                manual_thresh=mt,
                            )
                            cv.imwrite(str(proc_path), processed)
                            LAST_CAPTURE_INFO = {
                                "path": str(proc_path),
                                "success": True,
                                "timestamp": ts,
                                "layer": CURR_LAYER,
                                "z": CURR_Z,
                                "roi": roi,
                            }
                            print(f"[DEBUG] ROI applied? YES  points={len(roi)}  saved={proc_path}")
                except Exception as e:
                    print(f"[DEBUG] capture/process error: {e}")
                    LAST_CAPTURE_INFO = {"path": None, "success": False}

            # ----------------------
            # 2) Render slicer front-view once per layer
            # ----------------------
            if render_on_match and GCODE_PATH is not None and CURR_LAYER is not None:
                try:
                    if LAST_RENDERED_LAYER != CURR_LAYER:
                        slicer_out = out_dir / f"front_view_layer_{CURR_LAYER}.png"
                        _, info = face_view.render_front_view_by_layer(GCODE_PATH, slicer_out, CURR_LAYER)
                        LAST_RENDERED_LAYER = CURR_LAYER
                        LAST_RENDER_INFO = info
                        print(f"[DEBUG] Rendered slicer view: {slicer_out}")
                except Exception as e:
                    print(f"[DEBUG] render error: {e}")

            # ----------------------
            # 3) Hu-moment differential (if we have both images)
            # ----------------------
            if LAST_CAPTURE_INFO and LAST_CAPTURE_INFO.get("success") and CURR_LAYER is not None:
                slicer_png = out_dir / f"front_view_layer_{CURR_LAYER}.png"
                real_png = Path(LAST_CAPTURE_INFO["path"])

                slicer_img = cv.imread(str(slicer_png), cv.IMREAD_GRAYSCALE)
                real_img   = cv.imread(str(real_png), cv.IMREAD_GRAYSCALE)

                if slicer_img is not None and real_img is not None:
                    hu_s, hu_r, diff, score = img_process.log_hu_moment_diff(slicer_img, real_img)
                    print(f"[Layer {CURR_LAYER}] Hu L1 score: {score:.4f}")
                else:
                    print("⚠️ Missing slicer or real image for Hu diff.")

            return roi

        time.sleep(poll_interval)


# ------------------------------
# ROI milestones (in mm @ 1280×720)
# ------------------------------
ROI_MILESTONES: list[Tuple[float, list[Tuple[int, int]]]] = [
    (2.0,  [(382, 447), (385, 466), (910, 461), (908, 442), (383, 447)]),
    (3.0,  [(383, 467), (384, 441), (892, 438), (894, 463), (384, 467)]),
    (4.0,  [(379, 465), (900, 462), (895, 431), (379, 434), (379, 466)]),
    (5.0,  [(380, 430), (380, 467), (906, 463), (906, 426), (381, 428)]),
    (6.0,  [(374, 466), (911, 462), (908, 421), (376, 425), (374, 466)]),
    (7.0,  [(377, 466), (916, 462), (916, 421), (376, 424), (374, 464)]),
    (8.0,  [(373, 466), (912, 462), (908, 412), (375, 412), (372, 465)]),
    (9.0,  [(377, 467), (910, 462), (909, 405), (377, 408), (376, 468)]),
    (10.0, [(370, 466), (921, 461), (916, 402), (374, 407), (372, 465)]),
]



