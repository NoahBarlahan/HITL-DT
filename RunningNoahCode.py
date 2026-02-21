import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


# ---------------- Camera ----------------

def open_camera(max_index: int = 5) -> cv2.VideoCapture:
    backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    for backend in backends:
        for idx in range(max_index):
            cap = cv2.VideoCapture(idx, backend)
            if cap.isOpened():
                print(f"Opened camera index {idx} using backend {backend}")
                return cap
            cap.release()
    raise RuntimeError(f"Could not open any webcam (tried indices 0-{max_index-1}).")


# ---------------- BLUE HSV ----------------
# OpenCV HSV: H 0-179, S 0-255, V 0-255
BLUE_LOOSE = ((85, 40, 30), (145, 255, 255))
BLUE_TIGHT = ((95, 90, 60), (135, 255, 255))


def in_range(hsv: np.ndarray, low, high) -> np.ndarray:
    return cv2.inRange(hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8))


def build_blue_masks(hsv: np.ndarray, morph_kernel: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    loose = in_range(hsv, *BLUE_LOOSE)
    tight = in_range(hsv, *BLUE_TIGHT)

    k = morph_kernel
    kernel = np.ones((k, k), np.uint8)

    # A bit more cleanup to reduce flicker / broken blobs
    loose = cv2.morphologyEx(loose, cv2.MORPH_OPEN, kernel, iterations=1)
    loose = cv2.morphologyEx(loose, cv2.MORPH_CLOSE, kernel, iterations=2)

    tight = cv2.morphologyEx(tight, cv2.MORPH_OPEN, kernel, iterations=1)
    return loose, tight


# ---------------- Temporal smoothing of mask ----------------

class MaskSmoother:
    """Running average of a binary mask to reduce frame-to-frame flicker."""
    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha
        self.acc: Optional[np.ndarray] = None

    def update(self, mask_u8: np.ndarray) -> np.ndarray:
        m = mask_u8.astype(np.float32)
        if self.acc is None:
            self.acc = m
        else:
            self.acc = self.alpha * m + (1.0 - self.alpha) * self.acc

        # Threshold back to binary
        out = (self.acc >= 128.0).astype(np.uint8) * 255
        return out


# ---------------- Blob -> circle candidates ----------------

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def blue_confidence_for_blob(blue_loose: np.ndarray, blue_tight: np.ndarray, blob_mask: np.ndarray) -> float:
    loose_in = cv2.bitwise_and(blue_loose, blue_loose, mask=blob_mask)
    tight_in = cv2.bitwise_and(blue_tight, blue_tight, mask=blob_mask)

    loose_px = int(cv2.countNonZero(loose_in))
    tight_px = int(cv2.countNonZero(tight_in))
    blob_px = int(cv2.countNonZero(blob_mask))

    if blob_px == 0 or loose_px == 0:
        return 0.0

    core_ratio = tight_px / loose_px
    coverage = clamp01(loose_px / blob_px)
    return clamp01(0.65 * core_ratio + 0.35 * coverage)


def shape_confidence_from_fill(fill_ratio: float) -> float:
    """
    fill_ratio = blob_area / (pi*r^2) where r is minEnclosingCircle radius.
    Circles typically give higher fill_ratio; irregular blobs lower.
    """
    # Map fill_ratio into 0..1 softly. (0.55 -> ~0, 0.85 -> ~1)
    return clamp01((fill_ratio - 0.55) / (0.85 - 0.55))


@dataclass
class DetectedCircle:
    cx: float
    cy: float
    r: float
    blue_conf: float
    shape_conf: float
    final_conf: float
    area: int


def detect_blue_circles_from_mask(blue_mask: np.ndarray,
                                  blue_loose: np.ndarray,
                                  blue_tight: np.ndarray,
                                  min_area: int = 220,
                                  min_radius: int = 6,
                                  max_radius: int = 300) -> List[DetectedCircle]:
    """
    Uses connected components to get stable blobs (often fixes the 'mask sees it but detector doesn't' issue).
    """
    num, labels, stats, _centroids = cv2.connectedComponentsWithStats(blue_mask, connectivity=8)

    h, w = blue_mask.shape[:2]
    circles: List[DetectedCircle] = []

    for label in range(1, num):  # 0 is background
        x, y, ww, hh, area = stats[label]
        if area < min_area:
            continue

        # Extract blob mask
        blob = (labels == label).astype(np.uint8) * 255

        # Get contour for minEnclosingCircle
        cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)

        (cx, cy), r = cv2.minEnclosingCircle(c)
        if r < min_radius or r > max_radius:
            continue

        # Fill ratio for circle-likeness
        circle_area = np.pi * r * r
        fill_ratio = float(area) / float(circle_area) if circle_area > 0 else 0.0

        blue_conf = blue_confidence_for_blob(blue_loose, blue_tight, blob)
        shp_conf = shape_confidence_from_fill(fill_ratio)

        # Final confidence (continuous)
        final_conf = clamp01(0.75 * blue_conf + 0.25 * shp_conf)

        circles.append(DetectedCircle(cx=float(cx), cy=float(cy), r=float(r),
                                      blue_conf=float(blue_conf),
                                      shape_conf=float(shp_conf),
                                      final_conf=float(final_conf),
                                      area=int(area)))

    # Sort big first
    circles.sort(key=lambda c: c.area, reverse=True)
    return circles


# ---------------- Simple tracking to reduce flicker ----------------

@dataclass
class Track:
    id: int
    cx: float
    cy: float
    r: float
    conf: float
    last_seen: float


def match_and_update_tracks(tracks: List[Track],
                            detections: List[DetectedCircle],
                            now: float,
                            max_dist: float = 60.0,
                            smooth: float = 0.6,
                            keep_alive_s: float = 0.35) -> List[Track]:
    """
    Nearest-neighbor association + smoothing.
    Keeps tracks alive briefly even if a detection drops for a frame.
    """
    used = [False] * len(detections)

    # Update existing tracks
    for t in tracks:
        best_i = -1
        best_d = 1e9
        for i, det in enumerate(detections):
            if used[i]:
                continue
            d = np.hypot(det.cx - t.cx, det.cy - t.cy)
            if d < best_d:
                best_d = d
                best_i = i

        if best_i >= 0 and best_d <= max_dist:
            det = detections[best_i]
            used[best_i] = True

            # Smooth position/radius/confidence
            t.cx = smooth * t.cx + (1.0 - smooth) * det.cx
            t.cy = smooth * t.cy + (1.0 - smooth) * det.cy
            t.r  = smooth * t.r  + (1.0 - smooth) * det.r
            t.conf = smooth * t.conf + (1.0 - smooth) * det.final_conf
            t.last_seen = now

    # Create new tracks for unmatched detections
    next_id = (max([t.id for t in tracks], default=0) + 1)
    for i, det in enumerate(detections):
        if not used[i]:
            tracks.append(Track(id=next_id, cx=det.cx, cy=det.cy, r=det.r, conf=det.final_conf, last_seen=now))
            next_id += 1

    # Drop stale tracks
    tracks = [t for t in tracks if (now - t.last_seen) <= keep_alive_s]
    return tracks


# ---------------- Center logic ----------------

def centered_confidence(dist: float, r_circle: float, r_center_zone: float) -> float:
    """
    Confidence the entire detected circle is inside the center zone:
      dist + r_circle <= r_center_zone
    """
    margin = r_center_zone - (dist + r_circle)
    if margin <= 0:
        return 0.0
    return clamp01(margin / max(1.0, 0.5 * r_center_zone))


def draw_center_reference(frame, cx, cy, zone_r):
    cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (255, 255, 255), 2)
    cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (255, 255, 255), 2)
    cv2.circle(frame, (cx, cy), int(zone_r), (255, 255, 255), 2)


def draw_hud(frame, fps, n_tracks, best_center_conf, hold_s, hold_need_s, paused):
    x, y = 10, 10
    w, h = 420, 140
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    status = "PAUSED" if paused else "RUNNING"
    lines = [
        f"{status} | FPS: {fps:.1f}",
        f"Blue circles (stable): {n_tracks}",
        f"Best 'centered' conf: {best_center_conf*100:.0f}%",
        f"Hold: {min(hold_s, hold_need_s):.2f}/{hold_need_s:.2f}s",
        "Keys: D=mask  SPACE=pause  Q=quit",
    ]

    yy = y + 28
    for line in lines:
        cv2.putText(frame, line, (x + 12, yy), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
        yy += 24


# ---------------- Main ----------------

def main():
    cap = open_camera(max_index=5)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # Center zone & stop rule
    CENTER_ZONE_R = 20
    HOLD_REQUIRED_S = 2.0

    # Detection knobs (start reasonable)
    MIN_AREA = 220
    MIN_R = 6
    MAX_R = 260

    paused = False
    frozen_frame = None
    show_mask = False

    mask_smoother = MaskSmoother(alpha=0.35)
    tracks: List[Track] = []
    hold_start = None
    best_hold_report = None

    t_prev = time.time()
    fps = 0.0

    print("Blue-only stable circle detection + center hold.")
    print("Mask may show 4; this version should also keep 4 stable if they are blobs.")
    print("Keys: D mask, SPACE pause, Q quit.")

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame.")
                break
        else:
            frame = frozen_frame.copy()

        h, w = frame.shape[:2]
        cam_cx, cam_cy = w // 2, h // 2

        # FPS
        if not paused:
            t_now = time.time()
            dt = t_now - t_prev
            t_prev = t_now
            if dt > 0:
                inst = 1.0 / dt
                fps = inst if fps == 0 else (0.9 * fps + 0.1 * inst)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        blue_loose, blue_tight = build_blue_masks(hsv, morph_kernel=3)

        # Smooth mask to reduce flicker
        blue_mask = mask_smoother.update(blue_loose)

        detections = detect_blue_circles_from_mask(
            blue_mask,
            blue_loose=blue_loose,
            blue_tight=blue_tight,
            min_area=MIN_AREA,
            min_radius=MIN_R,
            max_radius=MAX_R,
        )

        now = time.time()
        tracks = match_and_update_tracks(tracks, detections, now,
                                         max_dist=60.0,
                                         smooth=0.6,
                                         keep_alive_s=0.35)

        # Draw
        out = frame.copy()
        draw_center_reference(out, cam_cx, cam_cy, CENTER_ZONE_R)

        best_centered_conf = 0.0
        best_centered_track = None

        for t in tracks:
            cx, cy, r = int(t.cx), int(t.cy), int(t.r)
            dx = cx - cam_cx
            dy = cy - cam_cy
            dist = float(np.hypot(dx, dy))

            center_fit = centered_confidence(dist, t.r, CENTER_ZONE_R)
            centered_conf = clamp01(center_fit * t.conf)  # combine "is circle blue" with "fully in center"

            if centered_conf > best_centered_conf:
                best_centered_conf = centered_conf
                best_centered_track = (t, dx, dy, dist, center_fit, centered_conf)

            # circle + id + offsets + conf
            cv2.circle(out, (cx, cy), r, (255, 0, 0), 2)
            cv2.circle(out, (cx, cy), 3, (255, 0, 0), -1)
            cv2.putText(out, f"ID{t.id}  {t.conf*100:.0f}%", (cx + r + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
            cv2.putText(out, f"dx={dx} dy={dy}", (cx + r + 6, cy + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Hold logic: stop when a circle is COMPLETELY inside center zone for >= 2s
        if best_centered_track is not None and not paused:
            t, dx, dy, dist, center_fit, centered_conf = best_centered_track
            is_completely_inside = (center_fit > 0.0)

            if is_completely_inside:
                if hold_start is None:
                    hold_start = now
                held = now - hold_start

                best_hold_report = (t, dx, dy, dist, center_fit, centered_conf)

                if held >= HOLD_REQUIRED_S:
                    paused = True
                    frozen_frame = frame.copy()

                    t, dx, dy, dist, center_fit, centered_conf = best_hold_report
                    print("\n=== BLUE CIRCLE IN CENTER FOR 2s ===")
                    print(f"Track ID: {t.id}")
                    print(f"Offset: dx={dx}, dy={dy} px; dist={dist:.2f}px")
                    print(f"Blue-circle confidence: {t.conf*100:.1f}%")
                    print(f"Inside-center confidence: {center_fit*100:.1f}%")
                    print(f"FINAL confidence it is fully inside center: {centered_conf*100:.1f}%")
            else:
                hold_start = None
        else:
            hold_start = None

        hold_s = (now - hold_start) if hold_start is not None else 0.0
        draw_hud(out, fps, len(tracks), best_centered_conf, hold_s, HOLD_REQUIRED_S, paused)

        if paused and best_hold_report is not None:
            cv2.putText(out, f"CENTERED! Conf={best_hold_report[5]*100:.0f}%",
                        (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        cv2.imshow("Blue Circles (Stable)", out)

        if show_mask:
            cv2.imshow("BLUE_MASK", blue_mask)
        else:
            try:
                cv2.destroyWindow("BLUE_MASK")
            except cv2.error:
                pass

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("d"):
            show_mask = not show_mask
        if key == ord(" "):
            if not paused:
                paused = True
                frozen_frame = frame.copy()
                hold_start = None
                best_hold_report = None
                print("Paused.")
            else:
                paused = False
                frozen_frame = None
                hold_start = None
                best_hold_report = None
                print("Resumed.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
