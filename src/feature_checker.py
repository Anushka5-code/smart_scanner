import cv2
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import CIRCULARITY_THRESHOLD, MIN_FEATURE_AREA


def compute_circularity(contour):
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return 0
    return (4 * np.pi * area) / (perimeter ** 2)


def detect_features(captured_img):
    """
    Detect all features (holes + cutouts) from captured image.
    Returns dict with detected features.
    """
    gray = cv2.cvtColor(captured_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15, C=4
    )

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=3)

    # Get all contours with hierarchy
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    holes   = []
    cutouts = []

    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            # Inner contours only (has parent)
            if hierarchy[0][i][3] != -1:
                area = cv2.contourArea(cnt)
                if area < MIN_FEATURE_AREA:
                    continue

                circ = compute_circularity(cnt)
                x, y, w, h = cv2.boundingRect(cnt)

                if circ >= CIRCULARITY_THRESHOLD:
                    (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                    holes.append({
                        "center": (int(cx), int(cy)),
                        "radius": int(radius),
                        "circularity": round(circ, 3)
                    })
                else:
                    # Filter out too-large rectangles (outer boundary noise)
                    img_h, img_w = captured_img.shape[:2]
                    if w < img_w * 0.85 and h < img_h * 0.85:
                        cutouts.append({
                            "bbox": (x, y, w, h),
                            "circularity": round(circ, 3)
                        })

    return {
        "holes":        holes,
        "cutouts":      cutouts,
        "hole_count":   len(holes),
        "cutout_count": len(cutouts)
    }


def run_feature_check(detected, specs):
    """
    Compare detected features against specs.
    Returns report dict with PASS/DEFECTIVE status.
    """
    checks  = []
    passed  = 0

    # Check hole count
    expected_holes = len(specs.get("holes", []))
    detected_holes = detected["hole_count"]
    hole_ok = expected_holes == detected_holes
    checks.append({
        "check":    "Hole Count",
        "expected": expected_holes,
        "measured": detected_holes,
        "pass":     hole_ok
    })
    if hole_ok:
        passed += 1

    # Check cutout presence
    has_cutout_spec = specs.get("square_cutout") is not None
    has_cutout_det  = detected["cutout_count"] > 0
    if has_cutout_spec:
        cutout_ok = has_cutout_det
        checks.append({
            "check":    "Square Cutout",
            "expected": "present",
            "measured": "present" if has_cutout_det else "missing",
            "pass":     cutout_ok
        })
        if cutout_ok:
            passed += 1

    failed  = [c for c in checks if not c["pass"]]
    status  = "PASS" if len(failed) == 0 else "DEFECTIVE"

    return {
        "status":       status,
        "total_checks": len(checks),
        "passed":       passed,
        "failed":       len(failed),
        "checks":       checks,
        "failed_checks": failed
    }


def draw_features(img, detected):
    """Draw detected features on image for result display."""
    vis = img.copy()

    for i, hole in enumerate(detected["holes"]):
        cx, cy = hole["center"]
        r = hole["radius"]
        cv2.circle(vis, (cx, cy), r, (255, 0, 0), 2)
        cv2.putText(vis, f"H{i+1}", (cx - 10, cy - r - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    for i, cutout in enumerate(detected["cutouts"]):
        x, y, w, h = cutout["bbox"]
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(vis, f"C{i+1}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return vis
