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

        time.sleep(settle_s)

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
# ROI selection
# ------------------------------
def find_roi_for_z(z: float, tol: float = None) -> Optional[List[Tuple[int, int]]]:
    if z is None:
        return None
    if tol is None:
        tol = LAYER_HEIGHT / 2.0

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
# Poller (core)
# ------------------------------
def monitor_roi(
    poll_interval: float = 1.0,
    tol: float = None,
    max_attempts: Optional[int] = None,
    render_on_match: bool = True,
    out_dir: Optional[Path] = None,
) -> Optional[List[Tuple[int, int]]]:

    global CURR_Z, CURR_LAYER, CURRENT_ROI
    global LAST_RENDERED_LAYER, LAST_RENDER_INFO, LAST_CAPTURE_INFO, LAST_CAPTURED_LAYER

    attempts = 0
    if out_dir is None:
        out_dir = Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)

    while True:
        if max_attempts is not None and attempts >= max_attempts:
            return None
        attempts += 1

        z = QIDI_Z_AXIS.get_z_height()
        CURR_Z = z
        CURR_LAYER = layer_from_z(CURR_Z, LAYER_HEIGHT)

        roi = find_roi_for_z(CURR_Z, tol=tol)
        if roi is not None:
            CURRENT_ROI = roi

            # Prevent capturing same layer twice
            if CURR_LAYER is not None and LAST_CAPTURED_LAYER == CURR_LAYER:
                time.sleep(poll_interval)
                continue

            LAST_CAPTURED_LAYER = CURR_LAYER

            # ----------------------
            # 1) Process real image
            # ----------------------
            if render_on_match and CURR_LAYER is not None:
                try:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    proc_path = out_dir / f"real_processed_L{CURR_LAYER}_{timestamp}.png"

                    info = img_process.process_current_real_image(
                        out_path=proc_path,
                        capture_fn=capture_image,
                        get_z_fn=QIDI_Z_AXIS.get_z_height,
                        roi_milestones=ROI_MILESTONES,
                        layer_height=LAYER_HEIGHT,
                        manual_thresh=180,
                    )

                    LAST_CAPTURE_INFO = {
                        "path": info.get("saved_path"),
                        "success": bool(info.get("captured")),
                        "timestamp": timestamp,
                        "layer": CURR_LAYER,
                        "z": info.get("z"),
                        "roi": info.get("roi_used"),
                    }
                except Exception:
                    LAST_CAPTURE_INFO = {"path": None, "success": False}

            # ----------------------
            # 2) Render slicer view
            # ----------------------
            if render_on_match and GCODE_PATH is not None and CURR_LAYER is not None:
                try:
                    if LAST_RENDERED_LAYER != CURR_LAYER:
                        slicer_out = out_dir / f"front_view_layer_{CURR_LAYER}.png"
                        _, info = face_view.render_front_view_by_layer(
                            GCODE_PATH, slicer_out, CURR_LAYER
                        )
                        LAST_RENDERED_LAYER = CURR_LAYER
                        LAST_RENDER_INFO = info
                except Exception:
                    pass

            # ----------------------
            # 3) Hu moment comparison
            # ----------------------
            if LAST_CAPTURE_INFO and LAST_CAPTURE_INFO.get("success"):
                slicer_png = out_dir / f"front_view_layer_{CURR_LAYER}.png"
                real_png = Path(LAST_CAPTURE_INFO["path"])

                slicer_img = cv.imread(str(slicer_png), cv.IMREAD_GRAYSCALE)
                real_img = cv.imread(str(real_png), cv.IMREAD_GRAYSCALE)

                if slicer_img is not None and real_img is not None:
                    hu_s, hu_r, diff, score = img_process.log_hu_moment_diff(
                        slicer_img, real_img
                    )
                    print(f"[Layer {CURR_LAYER}] Hu L1 score: {score:.4f}")
                else:
                    print("⚠️ Missing slicer or real image for Hu diff.")

            return roi

        time.sleep(poll_interval)


# ------------------------------
# ROI landmarks
# ------------------------------
ROI_MILESTONES: list[Tuple[float, list[Tuple[int, int]]]] = [
    (2.0,   [(373,460),(932,456),(931,450),(375,453),(371,460)]),
    (3.0,   [(373,461),(937,457),(933,447),(373,447),(374,461)]),
    (4.0,   [(372,459),(932,456),(931,442),(373,442),(373,459)]),
    (5.0,   [(371,461),(934,456),(930,438),(374,440),(370,460)]),
    (6.0,   [(373,460),(932,456),(931,434),(375,436),(371,460)]),
    (7.0,   [(373,460),(932,456),(931,430),(375,432),(371,460)]),
    (8.0,   [(373,460),(932,456),(931,426),(375,428),(371,460)]),
    (9.0,   [(373,460),(932,456),(931,422),(375,424),(371,460)]),
    (10.0,  [(373,460),(932,456),(931,418),(375,420),(371,460)]),
]


# ------------------------------
# MAIN (one-shot test)
# ------------------------------
if __name__ == "__main__":
    print("🔎 Waiting for ROI match and layer event...")
    roi = monitor_roi(
        poll_interval=1.0,
        max_attempts=5,
        out_dir=Path("out")  # processed images + slicer PNGs stored here
    )

    print("✅ Done (or out of attempts).")
