import cv2
import numpy as np
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import BLUEPRINTS_DIR, SPECS_FILE, MIN_FEATURE_AREA


def preprocess_blueprint(img):
    """Convert blueprint PNG to clean binary image."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=4
    )
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    return thresh


def extract_module_region(thresh, img_shape):
    """
    Find the module's OUTER contour — the solid body boundary.

    We score candidates by (area × solidity) so that a solid outer body
    wins over a high-contrast inner hollow (e.g. a square cutout inside
    a circular module).
    """
    h, w = img_shape[:2]
    image_area = h * w
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours
             if MIN_FEATURE_AREA < cv2.contourArea(c) < image_area * 0.85]
    if not valid:
        valid = contours
    if not valid:
        # fallback: return a dummy
        dummy = np.array([[[0, 0]], [[w, 0]], [[w, h]], [[0, h]]], dtype=np.int32)
        return dummy, (0, 0, w, h)

    def solidity(c):
        area = cv2.contourArea(c)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        return area / hull_area if hull_area > 0 else 0.0

    largest = max(valid, key=lambda c: cv2.contourArea(c) * solidity(c))
    x, y, cw, ch = cv2.boundingRect(largest)
    return largest, (x, y, cw, ch)


def count_holes_in_blueprint(thresh):
    """Count circular holes in blueprint for front/rear identification."""
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return 0
    holes = 0
    for i, cnt in enumerate(contours):
        if hierarchy[0][i][3] != -1:
            area = cv2.contourArea(cnt)
            if area > MIN_FEATURE_AREA:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter > 0:
                    circ = (4 * np.pi * area) / (perimeter ** 2)
                    if circ > 0.6:
                        holes += 1
    return holes


def orient_blueprint_crop(cropped, spec_length_mm, spec_width_mm):
    """
    Rotate the cropped blueprint image 90° if its aspect ratio disagrees
    with the physical module's spec dimensions.

    The blueprint drawing tool sometimes saves the image in portrait orientation
    even when the module is landscape (or vice versa).  We fix that here so
    the reference contour and the overlay are always in the correct orientation.

    spec_length_mm / spec_width_mm: the longer and shorter physical dimensions.
    If both are 0 (e.g. circular module) we leave the crop unchanged.
    """
    if spec_length_mm <= 0 or spec_width_mm <= 0:
        return cropped  # circular / unknown — don't touch

    ch, cw = cropped.shape[:2]
    spec_landscape = spec_length_mm >= spec_width_mm   # True → wider than tall
    img_landscape  = cw >= ch                           # True → wider than tall

    if spec_landscape != img_landscape:
        # Rotate 90° clockwise to flip orientation
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

    return cropped


def extract_blueprint_overlay(img, target_w, target_h, module_bbox=None,
                               spec_length_mm=0, spec_width_mm=0):
    """
    Extract module drawing from blueprint PNG.
    If module_bbox provided → scale blueprint box to match detected module size.
    Otherwise → fixed centered box (70% of screen height).
    spec_length_mm / spec_width_mm are used to correct orientation if the
    blueprint image was saved rotated.
    """
    thresh = preprocess_blueprint(img)
    _, (x, y, cw, ch) = extract_module_region(thresh, img.shape)

    pad = 5
    x  = max(0, x - pad)
    y  = max(0, y - pad)
    cw = min(img.shape[1] - x, cw + 2 * pad)
    ch = min(img.shape[0] - y, ch + 2 * pad)
    cropped = img[y:y+ch, x:x+cw]

    # Fix orientation: rotate if blueprint image aspect ≠ spec aspect
    cropped = orient_blueprint_crop(cropped, spec_length_mm, spec_width_mm)
    # Update cw/ch after possible rotation
    ch, cw = cropped.shape[:2]

    if module_bbox is not None:
        # Scale blueprint to fit over detected module — preserve aspect ratio
        mx, my, mw, mh = module_bbox

        # Scale blueprint to fit within module bounding box (preserve aspect ratio)
        scale   = min(mw / cw, mh / ch)
        box_w   = int(cw * scale)
        box_h   = int(ch * scale)

        # Center blueprint over module center
        mod_cx  = mx + mw // 2
        mod_cy  = my + mh // 2
        ox      = max(0, mod_cx - box_w // 2)
        oy      = max(0, mod_cy - box_h // 2)

        # Clamp to frame
        box_w = min(box_w, target_w - ox)
        box_h = min(box_h, target_h - oy)
    else:
        # Fixed centered box: 70% of screen height
        box_h = int(target_h * 0.70)
        scale = box_h / ch
        box_w = int(cw * scale)
        if box_w > int(target_w * 0.85):
            box_w = int(target_w * 0.85)
            scale = box_w / cw
            box_h = int(ch * scale)
        ox = (target_w - box_w) // 2
        oy = (target_h - box_h) // 2

    bp_resized = cv2.resize(cropped, (box_w, box_h), interpolation=cv2.INTER_LINEAR)

    overlay = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    overlay[:] = (30, 30, 30)
    overlay[oy:oy+box_h, ox:ox+box_w] = bp_resized

    return overlay, (ox, oy, box_w, box_h)


def load_all_blueprints():
    """
    Load all blueprints from specs.json.
    Returns list of blueprint dicts.
    """
    with open(SPECS_FILE, "r", encoding="utf-8") as f:
        specs = json.load(f)

    blueprints = []

    for module_id, module_data in specs.items():
        for view_name, view_data in module_data["views"].items():
            bp_filename = view_data["blueprint"]
            bp_path = os.path.join(BLUEPRINTS_DIR, bp_filename)

            if not os.path.exists(bp_path):
                print(f"[WARNING] Blueprint not found: {bp_path}")
                continue

            img = cv2.imread(bp_path)
            if img is None:
                print(f"[WARNING] Could not load: {bp_path}")
                continue

            # Physical dimensions from spec (used for orientation correction)
            spec_length = view_data.get("overall_length_mm", 0)
            spec_width  = view_data.get("overall_width_mm",  0)

            thresh = preprocess_blueprint(img)

            # Extract and orient the blueprint crop, then re-derive the contour
            # from the correctly-oriented image so matchShapes works properly.
            raw_contour, (bx, by, bcw, bch) = extract_module_region(thresh, img.shape)
            pad = 5
            bx  = max(0, bx - pad)
            by  = max(0, by - pad)
            bcw = min(img.shape[1] - bx, bcw + 2 * pad)
            bch = min(img.shape[0] - by, bch + 2 * pad)
            cropped_img = img[by:by+bch, bx:bx+bcw]
            oriented    = orient_blueprint_crop(cropped_img, spec_length, spec_width)

            # Re-extract contour from the oriented crop
            oriented_thresh = preprocess_blueprint(oriented)
            contour, bbox   = extract_module_region(oriented_thresh, oriented.shape)

            hole_count = count_holes_in_blueprint(thresh)

            blueprints.append({
                "module_id":      module_id,
                "view":           view_name,
                "description":    module_data.get("description", ""),
                "shape":          module_data.get("shape", "unknown"),
                "blueprint_img":  img,          # original full image (for overlay)
                "oriented_crop":  oriented,     # correctly-oriented crop
                "thresh":         thresh,
                "contour":        contour,      # contour from oriented crop
                "bbox":           bbox,
                "hole_count":     hole_count,
                "spec_length_mm": spec_length,
                "spec_width_mm":  spec_width,
                "specs":          view_data
            })

            oh, ow = oriented.shape[:2]
            print(f"[INFO] Loaded: {module_id}/{view_name} — "
                  f"oriented crop: {ow}x{oh}  holes: {hole_count}")

    return blueprints
