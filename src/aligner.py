import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MIN_FEATURE_AREA, ALIGNMENT_THRESHOLD, OVERLAY_ALPHA

# Fraction of frame height to black-out at the bottom before contour detection.
# Droidcam watermark sits in roughly the bottom 8% of the frame.
WATERMARK_MASK_BOTTOM = 0.08

# Fraction of frame width to black-out on the left edge (droidcam side bar).
WATERMARK_MASK_LEFT = 0.04


def _mask_watermark(gray, h, w):
    """Zero-out the regions where the droidcam watermark appears."""
    masked = gray.copy()
    # bottom strip
    cut_y = int(h * (1.0 - WATERMARK_MASK_BOTTOM))
    masked[cut_y:, :] = 0
    # left strip
    cut_x = int(w * WATERMARK_MASK_LEFT)
    masked[:, :cut_x] = 0
    return masked


def preprocess_frame(frame):
    """
    Extract the module's outer contour from the live camera frame.

    - Masks the droidcam watermark region before processing.
    - Combines Canny edges + adaptive threshold so the outer body is
      captured even when it has low contrast against the background.
    - Picks the candidate with the highest (area × solidity) score so
      a solid outer body always wins over a high-contrast inner hollow.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # ── mask watermark regions ──────────────────────────────────────
    gray = _mask_watermark(gray, h, w)

    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # edge-based mask
    edges = cv2.Canny(blur, 20, 80)
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=3)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # threshold-based mask (catches low-contrast outer body)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=21, C=6
    )
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel, iterations=1)

    combined = cv2.bitwise_or(edges, thresh)

    image_area = h * w
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # must be at least 0.5% and at most 85% of frame
    valid = [c for c in contours
             if image_area * 0.005 < cv2.contourArea(c) < image_area * 0.85]

    if not valid:
        return None

    def solidity(c):
        area = cv2.contourArea(c)
        hull_area = cv2.contourArea(cv2.convexHull(c))
        return area / hull_area if hull_area > 0 else 0.0

    # prefer large AND solid (outer body beats hollow inner feature)
    return max(valid, key=lambda c: cv2.contourArea(c) * solidity(c))


def get_contour_angle(contour):
    """
    Get the rotation angle of a contour using minAreaRect.
    Returns a value in (-45, 45] degrees.
    Positive = clockwise tilt.
    """
    if contour is None or len(contour) < 5:
        return 0.0
    rect  = cv2.minAreaRect(contour)
    angle = rect[-1]   # OpenCV returns (-90, 0]
    if angle < -45:
        angle += 90
    return angle


def compute_shape_score(contour_a, contour_b):
    """Returns 0.0–1.0 similarity between two contours (Hu moments)."""
    if contour_a is None or contour_b is None:
        return 0.0
    raw = cv2.matchShapes(contour_a, contour_b, cv2.CONTOURS_MATCH_I1, 0.0)
    return round(float(np.exp(-5.0 * raw)), 4)


def contour_circularity(contour):
    """Return circularity [0,1] of a contour (1.0 = perfect circle)."""
    if contour is None or len(contour) < 5:
        return 0.0
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0.0
    return (4 * np.pi * area) / (perimeter ** 2)


def check_camera_angle(contour, module_shape):
    """
    Check whether the camera is held roughly perpendicular (top-down) to the module.

    For CIRCULAR modules: the contour should be nearly circular (aspect ≈ 1.0).
    For RECTANGULAR/SQUARE modules: we can't use aspect ratio because the module
    IS rectangular. Instead we check that the contour's convex hull solidity is
    high (≥ 0.80) — a tilted rectangular object foreshortens and its projected
    silhouette becomes less solid / more trapezoidal.
    Returns True if the camera angle looks acceptable.
    """
    if contour is None or len(contour) < 5:
        return False

    from config import CAMERA_TILT_TOLERANCE

    if module_shape == "circular":
        _, (cw2, ch2), _ = cv2.minAreaRect(contour)
        if cw2 > 0 and ch2 > 0:
            aspect = min(cw2, ch2) / max(cw2, ch2)
            return aspect >= (1.0 - CAMERA_TILT_TOLERANCE)
        return False
    else:
        # For rectangular modules: solidity check
        area = cv2.contourArea(contour)
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if hull_area == 0:
            return False
        solidity = area / hull_area
        # Also check the contour is large enough to be the whole module
        # (not just a fragment)
        return solidity >= 0.75


def draw_searching(frame):
    """Phase 1 display — searching for module."""
    output = frame.copy()
    fh, fw = frame.shape[:2]
    cv2.putText(output, "SEARCHING FOR MODULE...",
                (fw//2 - 220, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 165, 255), 2)
    cv2.putText(output, "Point camera at module on white background",
                (fw//2 - 280, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    return output


def draw_aligning(frame, blueprint_entry, alignment_score,
                  overlay_img, placement, module_angle, contour=None):
    """
    Phase 2 display.
    Scales and positions the oriented blueprint crop over the detected module.
    Blueprint rotates to match module orientation.
    """
    output = frame.copy()
    fh, fw = frame.shape[:2]

    if contour is not None:
        mx, my, mw, mh = cv2.boundingRect(contour)
        mod_cx = mx + mw // 2
        mod_cy = my + mh // 2

        # Use the pre-oriented blueprint crop (orientation-corrected at load time)
        bp_cropped = blueprint_entry.get("oriented_crop", blueprint_entry["blueprint_img"])

        # Scale to fit the module bounding box, preserving aspect ratio
        scale = min(mw / bp_cropped.shape[1], mh / bp_cropped.shape[0])
        new_w = int(bp_cropped.shape[1] * scale)
        new_h = int(bp_cropped.shape[0] * scale)
        bp_scaled = cv2.resize(bp_cropped, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Place blueprint centered on module
        bp_ox = mod_cx - new_w // 2
        bp_oy = mod_cy - new_h // 2

        # Build full-frame canvas for rotation
        full_bp = np.full((fh, fw, 3), 200, dtype=np.uint8)

        x1 = max(0, bp_ox);      y1 = max(0, bp_oy)
        x2 = min(fw, bp_ox + new_w); y2 = min(fh, bp_oy + new_h)
        sx1 = x1 - bp_ox;        sy1 = y1 - bp_oy
        sx2 = sx1 + (x2 - x1);   sy2 = sy1 + (y2 - y1)

        if x2 > x1 and y2 > y1:
            full_bp[y1:y2, x1:x2] = bp_scaled[sy1:sy2, sx1:sx2]

        # Rotate blueprint to match module tilt
        # module_angle > 0 means CW tilt; warpAffine positive = CCW, so negate
        M = cv2.getRotationMatrix2D((float(mod_cx), float(mod_cy)), -module_angle, 1.0)
        full_bp_rotated = cv2.warpAffine(full_bp, M, (fw, fh),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(200, 200, 200))

        # Darken area outside the module
        vignette_mask = np.zeros((fh, fw), dtype=np.uint8)
        cv2.drawContours(vignette_mask, [contour], -1, 255, -1)
        k = np.ones((15, 15), np.uint8)
        vignette_mask = cv2.dilate(vignette_mask, k, iterations=2)
        dark = (output * 0.4).astype(np.uint8)
        output = np.where(vignette_mask[:, :, np.newaxis] == 255, output, dark)

        # Overlay only the dark blueprint lines (not the grey background)
        bp_gray2  = cv2.cvtColor(full_bp_rotated, cv2.COLOR_BGR2GRAY)
        line_mask = (bp_gray2 < 150).astype(np.uint8)
        for ch in range(3):
            output[:, :, ch] = np.where(
                line_mask == 1,
                (full_bp_rotated[:, :, ch] * 0.8 + output[:, :, ch] * 0.2).astype(np.uint8),
                output[:, :, ch]
            )

        placement = (mx, my, mw, mh)

    # Border / status color
    if alignment_score >= ALIGNMENT_THRESHOLD:
        color  = (0, 255, 0)
        status = "ALIGNED - HOLD STEADY"
    elif alignment_score >= 0.55:
        color  = (0, 255, 255)
        status = "GETTING CLOSE"
    else:
        color  = (0, 0, 255)
        status = "ALIGN MODULE TO BLUEPRINT"

    if contour is not None:
        cv2.drawContours(output, [contour], -1, color, 2)
    else:
        ox, oy, bw, bh = placement
        cv2.rectangle(output, (ox, oy), (ox+bw, oy+bh), color, 2)

    bar_w = int(fw * alignment_score)
    cv2.rectangle(output, (0, fh-18), (fw, fh), (30, 30, 30), -1)
    cv2.rectangle(output, (0, fh-18), (bar_w, fh), color, -1)

    pct = int(alignment_score * 100)
    cv2.putText(output, f"{status}  {pct}%",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(output,
                f"{blueprint_entry['module_id']} / {blueprint_entry['view'].upper()}  "
                f"Angle: {module_angle:.1f}",
                (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return output, placement


def crop_blueprint_box(frame, placement):
    """Crop exactly the blueprint box region from frame."""
    ox, oy, bw, bh = placement
    return frame[oy:oy+bh, ox:ox+bw].copy()
