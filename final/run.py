

# run.py
import time
from pathlib import Path
import cv2 as cv

import main        # your main.py logic
import img_process # for HU diff

OUT_DIR = Path("out")
OUT_DIR.mkdir(exist_ok=True)

print("=== SIMPLE RUN LOOP STARTED ===")
print("Press CTRL+C to exit.\n")

try:
    while True:
        roi = main.monitor_roi(
            poll_interval=1.0,
            tol=None,            # use main.py's default tolerance
            max_attempts=1,      # return after one poll
            out_dir=OUT_DIR
            manual_thresh=150,
        )

        if roi is None:
            # No matching ROI for current Z
            time.sleep(0.5)
            continue

        info = main.LAST_CAPTURE_INFO
        layer = info.get("layer")
        real_path = info.get("path")

        if not info.get("success") or real_path is None:
            print(f"[Layer {layer}] Capture failed")
            time.sleep(0.5)
            continue

        # -------------------------
        # Load images
        # -------------------------
        slicer_png = OUT_DIR / f"front_view_layer_{layer}.png"
        real_png = Path(real_path)

        slicer_img = cv.imread(str(slicer_png), cv.IMREAD_GRAYSCALE)
        real_img   = cv.imread(str(real_png), cv.IMREAD_GRAYSCALE)

        if slicer_img is None or real_img is None:
            print(f"[Layer {layer}] Missing slicer or real img.")
            time.sleep(0.5)
            continue

        # -------------------------
        # Compute HU diff (log-scaled)
        # -------------------------
        hu_s, hu_r, diff, score = img_process.log_hu_moment_diff(slicer_img, real_img)

        # Detailed per-component printout
        print(f"\n[Layer {layer}] Hu log components (slicer | real | |diff|):")
        for i in range(7):
            print(f"  h{i+1}: {hu_s[i]:9.5f} | {hu_r[i]:9.5f} | {diff[i]:9.5f}")

        # One-number summary + simple decision
        status = "PASS" if score < 0.75 else "FAIL"
        print(f"[Layer {layer}] L1 Sum = {score:.4f} | {status}\n")

        # Small pause before next iteration
        time.sleep(0.5)

except KeyboardInterrupt:
    print("\n=== STOPPED BY USER ===")

