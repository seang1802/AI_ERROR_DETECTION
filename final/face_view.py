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
    Render an XZ front view of the model up to and including target_layer.
    Returns (img, info_dict) and writes the result to out_path.
    """
    txt = Path(gcode_path).read_text(encoding="utf-8", errors="ignore")
    layer_h = get_layer_height(gcode_path, fallback=0.20)
    z_top = (target_layer + 1) * layer_h

    x = y = z = e = 0.0
    abs_xy = abs_e = True
    segs, z_max = [], 0.0

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

        px, pz, pe = x, z, e
        if "X" in words: x = words["X"] if abs_xy else x + words["X"]
        if "Y" in words: y = words["Y"] if abs_xy else y + words["Y"]
        if "Z" in words: z = words["Z"] if abs_xy else z + words["Z"]
        if "E" in words: e = words["E"] if abs_e else e + words["E"]

        if gcode == 1 and e > pe and max(pz, z) <= z_top:
            x0, z0 = max(0.0, min(bed_w, px)), max(0.0, pz)
            x1, z1 = max(0.0, min(bed_w, x)), max(0.0, z)
            segs.append(((x0, z0), (x1, z1)))
            z_max = max(z_max, z0, z1, z_top)

    if z_max <= 0.0:
        z_max = max(layer_h, z_top)

    w = int(round(bed_w * scale))
    h = int(round(z_max * scale))
    img = np.zeros((h, w), np.uint8)
    thick = max(1, int(round(layer_h * scale)))

    def to_px(xx, zz):
        return (
            int(round(max(0.0, min(bed_w, xx)) * scale)),
            int(round((z_max - max(0.0, zz)) * scale))
        )

    for (p0, p1) in segs:
        cv.line(img, to_px(*p0), to_px(*p1), 255, thick, cv.LINE_8)

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
