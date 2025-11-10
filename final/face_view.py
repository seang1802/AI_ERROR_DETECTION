import re
from pathlib import Path
import numpy as np
import cv2 as cv

detected_layer_h = None
FIRST_LAYER_Z    = None

# params
bed_w = 245.0   # mm
scale = 8.0     # px/mm

def get_layer_height(gcode_path, fallback=0.20):
    txt = Path(gcode_path).read_text(encoding="utf-8", errors="ignore")
    m = re.search(r'layer[\s_-]*height\s*[:=]\s*([0-9]*\.?[0-9]+)', txt, re.I)
    return float(m.group(1)) if m else fallback


def render_front_view_by_layer(gcode_path, out_path, target_layer):
    """
    Render a ***FILLED*** XZ silhouette of the model up to and including target_layer.
    This version produces a SOLID shape instead of horizontal line gaps.
    """
    txt = Path(gcode_path).read_text(encoding="utf-8", errors="ignore")
    layer_h = get_layer_height(gcode_path, fallback=0.20)
    z_top = (target_layer + 1) * layer_h

    x = y = z = e = 0.0
    abs_xy = abs_e = True
    segs = []
    z_max = 0.0

    for raw in txt.splitlines():
        s = raw.split(";", 1)[0].strip()
        if not s:
            continue

        words = {}
        for tok in s.split():
            if tok[0].isalpha() and len(tok) > 1:
                try:
                    words[tok[0].upper()] = float(tok[1:])
                except:
                    pass

        gcode = int(words["G"]) if "G" in words else None
        mcode = int(words["M"]) if "M" in words else None

        if gcode == 90: abs_xy = True
        elif gcode == 91: abs_xy = False
        if mcode == 82: abs_e = True
        elif mcode == 83: abs_e = False

        prev_x, prev_z, prev_e = x, z, e
        if "X" in words: x = words["X"] if abs_xy else x + words["X"]
        if "Y" in words: y = words["Y"] if abs_xy else y + words["Y"]
        if "Z" in words: z = words["Z"] if abs_xy else z + words["Z"]
        if "E" in words: e = words["E"] if abs_e else e + words["E"]

        # Extrusion move within current z-top
        if gcode == 1 and e > prev_e and max(prev_z, z) <= z_top:
            x0, z0 = max(0.0, min(bed_w, prev_x)), max(0.0, prev_z)
            x1, z1 = max(0.0, min(bed_w, x)),      max(0.0, z)
            segs.append(((x0, z0), (x1, z1)))
            z_max = max(z_max, z0, z1, z_top)

    if z_max <= 0.0:
        z_max = max(layer_h, z_top)

    img_w = int(round(bed_w * scale))
    img_h = int(round(z_max * scale))
    img = np.zeros((img_h, img_w), np.uint8)

    # thickness in pixels (height of filament)
    thick_px = int(round(layer_h * scale))
    if thick_px < 1: thick_px = 1

    def to_px(xx, zz):
        """Convert (x,z) mm to pixel coords (col,row)."""
        return (
            int(round(xx * scale)),
            int(round((z_max - zz) * scale))
        )


    for (p0, p1) in segs:
        x0, y0 = to_px(*p0)
        x1, y1 = to_px(*p1)

        # draw a filled rectangle covering extrusion path
        # vertical extent = filament thickness
        top = min(y0, y1)
        bottom = top + thick_px
        left = min(x0, x1)
        right = max(x0, x1)

        # clamp safely
        top = max(0, min(img_h - 1, top))
        bottom = max(0, min(img_h - 1, bottom))
        left = max(0, min(img_w - 1, left))
        right = max(0, min(img_w - 1, right))

        cv.rectangle(img, (left, top), (right, bottom), 255, -1)

    cv.imwrite(str(out_path), img)

    info = {
        "segments_drawn": len(segs),
        "z_top": z_top,
        "layer_h": layer_h,
        "target_layer": target_layer,
        "out_path": str(out_path),
        "z_max": z_max,
    }

    return img, info
