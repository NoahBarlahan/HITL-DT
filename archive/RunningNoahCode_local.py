import cv2
import numpy as np
from typing import Dict, List, Tuple


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


# ---------------- Color ranges (HSV) ----------------
COLOR_RANGES = {
    # Blue (unchanged)
    "B": [((95, 80, 50), (135, 255, 255))],

    # Yellow: "lighter" = allow lower saturation but require brighter value
    # (helps pick up pale/light yellow)
    "Y": [((18, 50, 170), (40, 255, 255))],

    # Orange: include darker orange by lowering V min, and slightly lowering S min
    # (also widened H a bit so it doesn't miss orange that shifts)
    "O": [((3, 80, 35), (22, 255, 255))],

    # White (unchanged-ish)
    "W": [((0, 0, 200), (179, 50, 255))],
}


DRAW_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "B": (255, 0, 0),
    "Y": (0, 255, 255),
    "O": (0, 165, 255),
    "W": (255, 255, 255),
}


# ---------------- Helpers ----------------

def build_mask(hsv: np.ndarray, ranges: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in ranges:
        lo = np.array(low, dtype=np.uint8)
        hi = np.array(high, dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    return mask


def clean_mask(mask: np.ndarray, k: int = 5) -> np.ndarray:
    # Blur helps reduce speckle noise before morphology
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    kernel = np.ones((k, k), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    return mask


def circularity(contour: np.ndarray) -> float:
    area = float(cv2.contourArea(contour))
    per = float(cv2.arcLength(contour, True))
    if per <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (per * per))


def extract_circle_candidates(mask: np.ndarray,
                              min_area: int,
                              min_radius: int,
                              max_radius: int,
                              min_circularity: float,
                              min_fill: float) -> List[Tuple[float, float, float, int]]:
    """
    Returns list of (cx, cy, r, area) that pass basic size + shape filters.
    """
    circles: List[Tuple[float, float, float, int]] = []
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for label in range(1, num):
        x, y, w, h, area = stats[label]
        if area < min_area:
            continue

        blob = (labels == label).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        c = max(cnts, key=cv2.contourArea)

        circ = circularity(c)
        if circ < min_circularity:
            continue

        (cx, cy), r = cv2.minEnclosingCircle(c)
        if r < min_radius or r > max_radius:
            continue

        # Fill ratio: blob area vs enclosing circle area
        circle_area = np.pi * r * r
        fill = float(area) / float(circle_area) if circle_area > 1e-6 else 0.0
        if fill < min_fill:
            continue

        circles.append((float(cx), float(cy), float(r), int(area)))

    # stable-ish order: top-to-bottom, then left-to-right
    circles.sort(key=lambda t: (t[1], t[0]))
    return circles


def filter_by_similar_radius(circles: List[Tuple[float, float, float, int]],
                             tol_ratio: float = 0.25) -> List[Tuple[float, float, float, int]]:
    """
    Keep only circles with radius close to the median radius.
    tol_ratio=0.25 means keep radii within ±25% of median.
    """
    if not circles:
        return circles

    radii = np.array([c[2] for c in circles], dtype=np.float32)
    med = float(np.median(radii))

    lo = med * (1.0 - tol_ratio)
    hi = med * (1.0 + tol_ratio)

    filtered = [c for c in circles if lo <= c[2] <= hi]
    return filtered


# ---------------- Main ----------------

def main():
    cap = open_camera(max_index=5)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # ---- Tuning knobs ----
    # Hard limits to reject obvious background detections:
    MIN_AREA = 250

    # If your pegs are always about the same size on screen, set tighter bounds here:
    MIN_R = 10
    MAX_R = 40

    # Shape filters:
    MIN_CIRCULARITY = 0.65   # 1.0 is perfect circle; 0.65 is fairly round
    MIN_FILL = 0.55          # blob should fill a decent amount of the enclosing circle

    # Similar-size filter (per color):
    RADIUS_TOL = 0.25        # +/- 25% around median radius
    # ----------------------

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame.")
            break

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        out = frame.copy()

        for code, ranges in COLOR_RANGES.items():
            mask = build_mask(hsv, ranges)
            mask = clean_mask(mask, k=5)

            circles = extract_circle_candidates(
                mask,
                min_area=MIN_AREA,
                min_radius=MIN_R,
                max_radius=MAX_R,
                min_circularity=MIN_CIRCULARITY,
                min_fill=MIN_FILL
            )

            # NEW: enforce similar peg sizes (kills big background blobs)
            circles = filter_by_similar_radius(circles, tol_ratio=RADIUS_TOL)

            for i, (cx, cy, r, _area) in enumerate(circles, start=1):
                x, y, rr = int(cx), int(cy), int(r)
                color = DRAW_COLORS_BGR.get(code, (0, 255, 0))

                cv2.circle(out, (x, y), rr, color, 2)
                cv2.circle(out, (x, y), 3, color, -1)

                label = f"{code}{i}"
                cv2.putText(out, label, (x + rr + 6, y - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Peg Color Detection", out)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
