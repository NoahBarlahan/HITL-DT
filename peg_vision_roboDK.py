import cv2
import numpy as np
from typing import Dict, List, Tuple

from robodk import robolink, robomath


# ========================= USER SETTINGS =========================

BOARD_FRAME_NAME = "BoardFrame"

# Z heights in BoardFrame (mm)
Z_PICK_MM = 0.0
Z_APPROACH_MM = 30.0

# Detection tuning
MIN_AREA = 250
MIN_R = 10
MAX_R = 45
MIN_CIRC = 0.70
MIN_FILL = 0.55
RAD_TOL = 0.25

# Camera settings
CAMERA_MAX_INDEX = 5
FRAME_W = 640
FRAME_H = 480

# Homography file (pixel -> board mm). You will create this later.
H_FILE = r"C:\Users\noah\Desktop\Python Reference\HITLDT\homography.npy"

# ================================================================


# ---------------- HSV color ranges ----------------
COLOR_RANGES: Dict[str, List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = {
    "B": [((95, 80, 50), (135, 255, 255))],
    "Y": [((18, 50, 170), (40, 255, 255))],   # lighter yellow
    "O": [((3, 80, 50), (22, 255, 255))],     # darker orange allowed
    "W": [((0, 0, 200), (179, 50, 255))],     # tighter white
}

DRAW_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "B": (255, 0, 0),
    "Y": (0, 255, 255),
    "O": (0, 165, 255),
    "W": (255, 255, 255),
}


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


# ---------------- Vision helpers ----------------

def build_mask(hsv: np.ndarray, ranges) -> np.ndarray:
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in ranges:
        lo = np.array(low, dtype=np.uint8)
        hi = np.array(high, dtype=np.uint8)
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
    return mask


def clean_mask(mask: np.ndarray, k: int = 5) -> np.ndarray:
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


def extract_circle_candidates(mask: np.ndarray) -> List[Tuple[float, float, float, int]]:
    circles: List[Tuple[float, float, float, int]] = []
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    for label in range(1, num):
        x, y, w, h, area = stats[label]
        if area < MIN_AREA:
            continue

        blob = (labels == label).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue

        c = max(cnts, key=cv2.contourArea)
        if circularity(c) < MIN_CIRC:
            continue

        (cx, cy), r = cv2.minEnclosingCircle(c)
        if r < MIN_R or r > MAX_R:
            continue

        circle_area = np.pi * r * r
        fill = float(area) / float(circle_area) if circle_area > 1e-6 else 0.0
        if fill < MIN_FILL:
            continue

        circles.append((float(cx), float(cy), float(r), int(area)))

    circles.sort(key=lambda t: (t[1], t[0]))
    return circles


def filter_by_similar_radius(circles: List[Tuple[float, float, float, int]], tol_ratio: float = RAD_TOL):
    if not circles:
        return circles
    radii = np.array([c[2] for c in circles], dtype=np.float32)
    med = float(np.median(radii))
    lo, hi = med * (1.0 - tol_ratio), med * (1.0 + tol_ratio)
    return [c for c in circles if lo <= c[2] <= hi]


# ---------------- Homography (pixel -> board mm) ----------------

def load_homography() -> np.ndarray:
    try:
        H = np.load(H_FILE)
        if H.shape != (3, 3):
            raise ValueError("homography.npy is not 3x3")
        print("Loaded homography from:", H_FILE)
        return H.astype(np.float64)
    except Exception as e:
        print("WARNING: Failed to load homography:", e)
        print("Using identity homography (targets will be WRONG until you calibrate).")
        return np.eye(3, dtype=np.float64)


def apply_homography(H: np.ndarray, uv: Tuple[float, float]) -> Tuple[float, float]:
    u, v = uv
    pt = np.array([[[u, v]]], dtype=np.float32)
    xy = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(xy[0]), float(xy[1])


# ---------------- RoboDK helpers ----------------

def require_frame(rdk: robolink.Robolink, name: str):
    f = rdk.Item(name, robolink.ITEM_TYPE_FRAME)
    if not f.Valid():
        raise RuntimeError(
            f"Frame '{name}' not found.\n"
            f"Create a Reference Frame in RoboDK named '{name}' located at the pegboard origin."
        )
    return f


def get_or_create_target(rdk: robolink.Robolink, name: str):
    t = rdk.Item(name, robolink.ITEM_TYPE_TARGET)
    if not t.Valid():
        t = rdk.AddTarget(name)
    return t


# ---------------- Main ----------------

def main():
    rdk = robolink.Robolink()

    robot = rdk.ItemUserPick("Select the robot", robolink.ITEM_TYPE_ROBOT)
    if not robot.Valid():
        raise RuntimeError("No robot selected.")

    board_frame = require_frame(rdk, BOARD_FRAME_NAME)
    base_T_board = board_frame.PoseAbs()

    H = load_homography()

    cap = open_camera(max_index=CAMERA_MAX_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("Running. Q/ESC quit.")

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

            circles = extract_circle_candidates(mask)
            circles = filter_by_similar_radius(circles, tol_ratio=RAD_TOL)

            for i, (u, v, r, _area) in enumerate(circles, start=1):
                Xmm, Ymm = apply_homography(H, (u, v))

                # draw
                color = DRAW_COLORS_BGR.get(code, (0, 255, 0))
                cv2.circle(out, (int(u), int(v)), int(r), color, 2)
                cv2.circle(out, (int(u), int(v)), 3, color, -1)
                cv2.putText(out, f"{code}{i}", (int(u) + int(r) + 6, int(v) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                # update targets
                pick_name = f"{code}{i}"
                app_name = f"{code}{i}_APP"
                pick_t = get_or_create_target(rdk, pick_name)
                app_t = get_or_create_target(rdk, app_name)

                board_T_pick = robomath.transl(Xmm, Ymm, Z_PICK_MM)
                board_T_app  = robomath.transl(Xmm, Ymm, Z_PICK_MM + Z_APPROACH_MM)

                pick_t.setPoseAbs(base_T_board * board_T_pick)
                app_t.setPoseAbs(base_T_board * board_T_app)

        cv2.imshow("Peg Detection", out)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
