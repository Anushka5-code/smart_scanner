import cv2
import numpy as np
import os
import datetime
import json

from src.blueprint_loader import load_all_blueprints, extract_blueprint_overlay
from src.aligner import (preprocess_frame, compute_shape_score,
                         get_contour_angle, draw_searching,
                         draw_aligning, crop_blueprint_box,
                         contour_circularity, check_camera_angle)
from config import (CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT,
                    ALIGNMENT_THRESHOLD, ALIGNMENT_HOLD_FRAMES,
                    DETECTION_THRESHOLD, FILL_THRESHOLD,
                    CAMERA_TILT_TOLERANCE, WINDOW_NAME)


def save_capture(img, module_id, view):
    os.makedirs("captures", exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"captures/{module_id}_{view}_{ts}"
    path = f"{base}.png"
    ok   = cv2.imwrite(path, img)
    if ok:
        print(f"[SAVED] {path}")
        with open(f"{base}.json", "w") as f:
            json.dump({"module_id": module_id, "view": view,
                       "timestamp": ts, "image": path}, f, indent=2)
    else:
        print(f"[ERROR] Could not save {path}")
    return path


def show_flash(frame, module_id, view):
    """Flash green screen for 1 second to confirm capture."""
    display = frame.copy()
    h, w = display.shape[:2]
    cv2.rectangle(display, (0, 0), (w, h), (0, 255, 0), 25)
    cv2.putText(display, "CAPTURED!",
                (w//2 - 130, h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
    cv2.putText(display, f"{module_id} / {view.upper()}",
                (w//2 - 160, h//2 + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.imshow(WINDOW_NAME, display)
    cv2.waitKey(800)


def select_view(blueprints):
    """Show FRONT / REAR selection screen."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    views = []
    for bp in blueprints:
        if bp["view"] not in views:
            views.append(bp["view"])

    selected = None
    while selected is None:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        fh, fw  = display.shape[:2]
        dark    = display.copy()
        dark[:] = (20, 20, 20)
        cv2.addWeighted(dark, 0.65, display, 0.35, 0, display)

        cv2.putText(display, "SELECT FACE TO SCAN",
                    (fw//2 - 190, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 2)

        for i, v in enumerate(views):
            cv2.putText(display, f"[{i+1}]  {v.upper()}",
                        (fw//2 - 80, 180 + i * 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)

        cv2.putText(display, "Module will be auto-identified",
                    (fw//2 - 200, fh - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF

        for i, v in enumerate(views):
            if key == ord(str(i + 1)):
                selected = v

        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return None

    cap.release()
    return selected


def run_scanner():
    print("[INFO] Loading blueprints...")
    blueprints = load_all_blueprints()
    if not blueprints:
        print("[ERROR] No blueprints loaded.")
        return

    selected_view = select_view(blueprints)
    if selected_view is None:
        return

    view_bps = [bp for bp in blueprints if bp["view"] == selected_view]
    print(f"[INFO] Scanning view: {selected_view.upper()} "
          f"({len(view_bps)} blueprint(s) to match)")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}")
        return

    ret, test = cap.read()
    if not ret:
        print("[ERROR] Cannot read from camera")
        return

    fh, fw = test.shape[:2]

    for bp in view_bps:
        ov, pl = extract_blueprint_overlay(
            bp["blueprint_img"], fw, fh,
            spec_length_mm=bp.get("spec_length_mm", 0),
            spec_width_mm=bp.get("spec_width_mm", 0))
        bp["overlay_img"]       = ov
        bp["overlay_placement"] = pl

    print("[INFO] Scanner ready.")
    print("  S → manual capture   B → back   Q → quit\n")

    PHASE_SEARCHING = 0
    PHASE_ALIGNING  = 1

    phase          = PHASE_SEARCHING
    detected_bp    = None
    aligned_frames = 0
    smooth_angle   = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        contour = preprocess_frame(frame)

        # ── PHASE 1: SEARCHING ──────────────────────────
        if phase == PHASE_SEARCHING:
            display = draw_searching(frame)

            if contour is not None:
                live_circ = contour_circularity(contour)
                best_bp    = None
                best_score = 0.0
                for bp in view_bps:
                    # Shape-type pre-filter: skip if live contour circularity
                    # is incompatible with the blueprint's declared shape.
                    # circular blueprint  → live_circ should be > 0.6
                    # square/rect blueprint → live_circ should be < 0.75
                    bp_shape = bp.get("shape", "unknown")
                    if bp_shape == "circular" and live_circ < 0.55:
                        continue
                    if bp_shape in ("square", "rectangle") and live_circ > 0.80:
                        continue

                    s = compute_shape_score(contour, bp["contour"])
                    if s > best_score:
                        best_score = s
                        best_bp    = bp

                pct = int(best_score * 100)
                cv2.putText(display, f"Match: {pct}%  Circ: {live_circ:.2f}",
                            (10, fh - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

                if best_score >= DETECTION_THRESHOLD and best_bp is not None:
                    detected_bp    = best_bp
                    phase          = PHASE_ALIGNING
                    aligned_frames = 0
                    smooth_angle   = get_contour_angle(contour)
                    print(f"[DETECTED] {detected_bp['module_id']} / "
                          f"{detected_bp['view']} (score: {best_score:.3f}, "
                          f"circ: {live_circ:.2f})")

        # ── PHASE 2: ALIGNING ───────────────────────────
        elif phase == PHASE_ALIGNING:
            alignment_score = 0.0
            module_angle    = smooth_angle
            fill_ratio      = 0.0
            camera_90       = False

            if contour is not None:
                alignment_score = compute_shape_score(contour, detected_bp["contour"])

                raw_angle    = get_contour_angle(contour)
                smooth_angle = smooth_angle * 0.7 + raw_angle * 0.3
                module_angle = smooth_angle

                # Resize blueprint box to match detected module
                mx, my, mw, mh = cv2.boundingRect(contour)
                ov, pl = extract_blueprint_overlay(
                    detected_bp["blueprint_img"], fw, fh,
                    module_bbox=(mx, my, mw, mh),
                    spec_length_mm=detected_bp.get("spec_length_mm", 0),
                    spec_width_mm=detected_bp.get("spec_width_mm", 0))
                detected_bp["overlay_img"]       = ov
                detected_bp["overlay_placement"] = pl

                # Fill ratio
                ox, oy, bw, bh = detected_bp["overlay_placement"]
                box_area = bw * bh
                cnt_mask = np.zeros((fh, fw), dtype=np.uint8)
                cv2.drawContours(cnt_mask, [contour], -1, 255, -1)
                box_mask = np.zeros((fh, fw), dtype=np.uint8)
                box_mask[oy:oy+bh, ox:ox+bw] = 255
                overlap_area = np.count_nonzero(cv2.bitwise_and(cnt_mask, box_mask))
                fill_ratio   = min(overlap_area / box_area, 1.0) if box_area > 0 else 0.0

                # Camera perpendicularity check — method depends on module shape
                camera_90 = check_camera_angle(contour, detected_bp.get("shape", "unknown"))

            display, _ = draw_aligning(
                frame, detected_bp, alignment_score,
                detected_bp["overlay_img"],
                detected_bp["overlay_placement"],
                module_angle, contour)

            # Status indicators
            cam_color = (0, 255, 0) if camera_90 else (0, 0, 255)
            cv2.putText(display,
                        f"90°: {'OK' if camera_90 else 'Hold camera top-down'}",
                        (fw - 320, 82),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, cam_color, 2)

            cv2.putText(display, f"Fill: {int(fill_ratio*100)}%",
                        (fw - 130, fh - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)

            # Module lost
            if contour is None or alignment_score < 0.25:
                phase          = PHASE_SEARCHING
                detected_bp    = None
                aligned_frames = 0

            # ALL THREE conditions: alignment + fill + camera at 90°
            elif alignment_score >= ALIGNMENT_THRESHOLD \
                    and fill_ratio >= FILL_THRESHOLD \
                    and camera_90:
                aligned_frames += 1
                hold_pct = min(aligned_frames / ALIGNMENT_HOLD_FRAMES, 1.0)
                bar_w    = int(fw * hold_pct)
                cv2.rectangle(display, (0, fh-40), (bar_w, fh-20), (0, 255, 0), -1)
                cv2.putText(display,
                            f"Hold... {aligned_frames}/{ALIGNMENT_HOLD_FRAMES}",
                            (10, fh - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                aligned_frames = 0
                if not camera_90:
                    cv2.putText(display, "Hold camera directly above module",
                                (10, fh - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                elif fill_ratio < FILL_THRESHOLD:
                    cv2.putText(display,
                                f"Move closer  Fill:{int(fill_ratio*100)}%",
                                (10, fh - 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            # AUTO CAPTURE
            if aligned_frames >= ALIGNMENT_HOLD_FRAMES:
                aligned_frames = 0
                print(f"\n[CAPTURE] {detected_bp['module_id']} / {detected_bp['view']}")
                if contour is not None:
                    mx, my, mw, mh = cv2.boundingRect(contour)
                    pad = 10
                    mx = max(0, mx - pad)
                    my = max(0, my - pad)
                    mw = min(fw - mx, mw + 2*pad)
                    mh = min(fh - my, mh + 2*pad)
                    cropped = frame[my:my+mh, mx:mx+mw].copy()
                else:
                    ox, oy, bw, bh = detected_bp["overlay_placement"]
                    cropped = crop_blueprint_box(frame, (ox, oy, bw, bh))
                save_capture(cropped, detected_bp["module_id"], detected_bp["view"])
                show_flash(display, detected_bp["module_id"], detected_bp["view"])
                phase       = PHASE_SEARCHING
                detected_bp = None
                print("[INFO] Ready for next scan...")

        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and phase == PHASE_ALIGNING and detected_bp is not None:
            print(f"\n[MANUAL CAPTURE] {detected_bp['module_id']} / {detected_bp['view']}")
            if contour is not None:
                mx, my, mw, mh = cv2.boundingRect(contour)
                pad = 10
                mx = max(0, mx - pad)
                my = max(0, my - pad)
                mw = min(fw - mx, mw + 2*pad)
                mh = min(fh - my, mh + 2*pad)
                cropped = frame[my:my+mh, mx:mx+mw].copy()
            else:
                ox, oy, bw, bh = detected_bp["overlay_placement"]
                cropped = crop_blueprint_box(frame, (ox, oy, bw, bh))
            save_capture(cropped, detected_bp["module_id"], detected_bp["view"])
            show_flash(display, detected_bp["module_id"], detected_bp["view"])
            phase       = PHASE_SEARCHING
            detected_bp = None

        elif key == ord('b'):
            cap.release()
            cv2.destroyAllWindows()
            run_scanner()
            return

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_scanner()
