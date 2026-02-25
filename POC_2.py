import cv2
import numpy as np
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from robodk import robolink, robomath

# ============================================================
# USER SETTINGS (RoboDK)
# ============================================================

BOARD_FRAME_NAME = "BoardFrame"
PICK_REF_TARGET_NAME = "PickRef"
H_FILE = r"C:\Users\noah\Desktop\HITL-DT\homography.npy"  # <-- update

# Z heights in mm (relative to BoardFrame)
Z_PICK_MM = 50.0
Z_APPROACH_MM = 100.0

# Motion style
DO_PICK_MOTION = True
DO_DROP_MOTION = True

# Two labels
LABEL_TARGET = "TARGET"
LABEL_OTHER = "OTHER"

# Optional drop targets in RoboDK station (if missing, drop is skipped)
LABEL_TO_DROP_TARGETS = {
    LABEL_TARGET: ("TARGET_DROP_APP", "TARGET_DROP"),
    LABEL_OTHER: ("OTHER_DROP_APP", "OTHER_DROP"),
}

# Names for dynamic pick targets we will update each time
DYN_PICK_APP_NAME = "CUR_PICK_APP"
DYN_PICK_NAME = "CUR_PICK"

# ============================================================
# USER SETTINGS (Camera)
# ============================================================

CAMERA_INDEX = 0
FRAME_W = 640
FRAME_H = 480

# ============================================================
# VISION SETTINGS (Box-draw training + detection)
# ============================================================

WINDOW_NAME = "RoboDK + Box Learn (TARGET vs OTHER) [No Save]"

# Candidate region detection (bboxes from edges)
MIN_AREA_PX = 2500
MAX_AREA_PX = 250000

# Tracking
MATCH_DIST_PX = 90
MISSING_FRAMES_TO_REMOVE = 20

# MUST see object for this many seconds before it can be treated as a real object
SEEN_SECONDS_TO_ACT = 2.0

# Confidence thresholds (TARGET-vs-OTHER)
CONF_THRESH_TARGET = 0.80
CONF_THRESH_OTHER = 0.30

# If it matches NOT-OBJECT this confidently, ignore automatically
CONF_THRESH_NOT_OBJECT = 0.75

# Similarity scaling (higher => higher confidence)
DIST_SCALE = 1.2

# Template/feature sizes
ROI_SIZE = (96, 96)  # normalization size for descriptors
HOG_WIN = (96, 96)

# In-memory learning caps
MAX_TARGET_SAMPLES = 80
MAX_NOT_OBJECT_SAMPLES = 200

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class TrackedObj:
    obj_id: int
    cx: int
    cy: int
    w: int
    h: int
    last_seen_frame: int
    first_seen_t: float
    last_update_t: float

    feat: np.ndarray
    label: str
    acted: bool

# ============================================================
# RoboDK helpers
# ============================================================

def require_frame(rdk: robolink.Robolink, name: str):
    f = rdk.Item(name, robolink.ITEM_TYPE_FRAME)
    if not f.Valid():
        raise RuntimeError(f"Frame '{name}' not found. Create it in RoboDK.")
    return f

def require_target(rdk: robolink.Robolink, name: str):
    t = rdk.Item(name, robolink.ITEM_TYPE_TARGET)
    return t if t.Valid() else None

def get_or_create_target(rdk: robolink.Robolink, name: str):
    t = rdk.Item(name, robolink.ITEM_TYPE_TARGET)
    if not t.Valid():
        t = rdk.AddTarget(name)
        print(f"[RoboDK] Created target '{name}'. You may need to set it up if it's a fixed drop target.")
    return t

def load_homography(path: str) -> np.ndarray:
    try:
        H = np.load(path)
        if H.shape != (3, 3):
            raise ValueError("Homography must be 3x3")
        print("[INFO] Loaded homography:", path)
        return H.astype(np.float64)
    except Exception as e:
        print("[WARN] Could not load homography. Using identity. ERROR:", e)
        return np.eye(3, dtype=np.float64)

def apply_homography(H: np.ndarray, uv: Tuple[float, float]) -> Tuple[float, float]:
    u, v = uv
    pt = np.array([[[u, v]]], dtype=np.float32)
    xy = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(xy[0]), float(xy[1])

def pose_with_ref_rotation(ref_pose_abs, new_pose_abs):
    out = ref_pose_abs.copy()
    out.setPos(new_pose_abs.Pos())
    return out

def move_pick_and_optional_drop(
    robot,
    base_T_board,
    pick_ref_pose_abs,
    Xmm: float,
    Ymm: float,
    t_app,
    t_pick,
    drop_app_t,
    drop_t,
    do_pick: bool,
    do_drop: bool
):
    board_T_pick = robomath.transl(Xmm, Ymm, Z_PICK_MM)
    board_T_app  = robomath.transl(Xmm, Ymm, Z_PICK_MM + Z_APPROACH_MM)

    pick_pose_abs = base_T_board * board_T_pick
    app_pose_abs  = base_T_board * board_T_app

    pick_pose_abs = pose_with_ref_rotation(pick_ref_pose_abs, pick_pose_abs)
    app_pose_abs  = pose_with_ref_rotation(pick_ref_pose_abs, app_pose_abs)

    t_pick.setPoseAbs(pick_pose_abs)
    t_app.setPoseAbs(app_pose_abs)

    try:
        if do_pick:
            robot.MoveJ(t_app)
            robot.MoveL(t_pick)
            robot.MoveL(t_app)

        if do_drop and (drop_app_t is not None) and (drop_t is not None):
            robot.MoveJ(drop_app_t)
            robot.MoveL(drop_t)
            robot.MoveL(drop_app_t)

    except Exception as e:
        print("[RoboDK] Move error:", e)
        print("Common causes: unreachable target, wrong BoardFrame pose, wrong homography, wrong tool.")

# ============================================================
# Feature extraction (HOG + HSV hist)
# ============================================================

def make_hog():
    return cv2.HOGDescriptor(
        _winSize=HOG_WIN,
        _blockSize=(32, 32),
        _blockStride=(16, 16),
        _cellSize=(16, 16),
        _nbins=9
    )

HOG = make_hog()

def extract_feature(frame_bgr: np.ndarray, box: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    x, y, w, h = box
    Hh, Ww = frame_bgr.shape[:2]
    x = max(0, min(x, Ww - 1))
    y = max(0, min(y, Hh - 1))
    w = max(1, min(w, Ww - x))
    h = max(1, min(h, Hh - y))
    roi = frame_bgr[y:y+h, x:x+w]
    if roi.size == 0:
        return None

    roi = cv2.resize(roi, ROI_SIZE, interpolation=cv2.INTER_AREA)

    # HOG on gray
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hog = HOG.compute(gray).reshape(-1).astype(np.float32)

    # HSV histogram (appearance)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist_h = cv2.calcHist([hsv], [0], None, [24], [0, 180]).reshape(-1)
    hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).reshape(-1)
    hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).reshape(-1)
    hist = np.concatenate([hist_h, hist_s, hist_v]).astype(np.float32)

    hist_sum = float(np.sum(hist)) + 1e-6
    hist /= hist_sum

    feat = np.concatenate([hog, hist]).astype(np.float32)
    n = float(np.linalg.norm(feat)) + 1e-6
    feat /= n
    return feat

def feature_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))

def dist_to_conf(dist: float) -> float:
    return float(np.exp(-dist / max(1e-6, DIST_SCALE)))

# ============================================================
# Candidate bbox detection (edges -> blobs -> boxes)
# ============================================================

def detect_candidate_boxes(frame_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    med = np.median(gray)
    lo = int(max(0, 0.90 * med))
    hi = int(min(255, 1.80 * med))
    edges = cv2.Canny(gray, lo, hi)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    blobs = cv2.dilate(edges, k, iterations=1)
    blobs = cv2.morphologyEx(blobs, cv2.MORPH_CLOSE, k, iterations=1)

    cnts, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < MIN_AREA_PX or area > MAX_AREA_PX:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w < 10 or h < 10:
            continue
        boxes.append((x, y, w, h))

    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return boxes

# ============================================================
# HITL prompt
# ============================================================

def ask_user_action_lowconf(obj_id: int, conf: float) -> str:
    """
    Returns: TARGET / OTHER / NOT_OBJECT / SKIP / CLEAR
    """
    print("\n[HITL] Low confidence classification.")
    print(f"[HITL] Object id={obj_id} | confidence={conf*100:.1f}%")
    print("  t) It's TARGET (learn it)")
    print("  o) It's OTHER")
    print("  n) NOT an object (learn ignore)")
    print("  s) Skip (no motion)")
    print("  c) Clear target model (forget target)")
    while True:
        ans = input("Choice (t/o/n/s/c): ").strip().lower()
        if ans in ("t", "o", "n", "s", "c"):
            return {
                "t": LABEL_TARGET,
                "o": LABEL_OTHER,
                "n": "NOT_OBJECT",
                "s": "SKIP",
                "c": "CLEAR",
            }[ans]
        print("Invalid input. Try again.")

# ============================================================
# MAIN
# ============================================================

def main():
    cap = None

    # ---- RoboDK init ----
    rdk = robolink.Robolink()
    robot = rdk.ItemUserPick("Select the robot", robolink.ITEM_TYPE_ROBOT)
    if not robot.Valid():
        raise RuntimeError("No robot selected.")

    board_frame = require_frame(rdk, BOARD_FRAME_NAME)
    base_T_board = board_frame.PoseAbs()

    pick_ref = rdk.Item(PICK_REF_TARGET_NAME, robolink.ITEM_TYPE_TARGET)
    if not pick_ref.Valid():
        raise RuntimeError(f"Create a RoboDK target named '{PICK_REF_TARGET_NAME}' with correct pick orientation.")
    pick_ref_pose_abs = pick_ref.PoseAbs()

    H = load_homography(H_FILE)

    # Dynamic targets for pick
    t_pick_app = get_or_create_target(rdk, DYN_PICK_APP_NAME)
    t_pick = get_or_create_target(rdk, DYN_PICK_NAME)

    # Drop targets (optional)
    drop_targets: Dict[str, Tuple[Optional[robolink.Item], Optional[robolink.Item]]] = {}
    for label in (LABEL_TARGET, LABEL_OTHER):
        app_name, drop_name = LABEL_TO_DROP_TARGETS[label]
        drop_app_t = require_target(rdk, app_name)
        drop_t = require_target(rdk, drop_name)
        if drop_app_t is None or drop_t is None:
            print(f"[WARN] Missing drop targets for {label}: {app_name}, {drop_name} (drop will be skipped for this label)")
        drop_targets[label] = (drop_app_t, drop_t)

    # ---- Camera init (must be before loop) ----
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # ---- In-memory models (reset on quit) ----
    target_samples: List[np.ndarray] = []
    target_center: Optional[np.ndarray] = None

    not_object_samples: List[np.ndarray] = []
    not_object_center: Optional[np.ndarray] = None

    def update_center_from_samples(samples: List[np.ndarray]) -> Optional[np.ndarray]:
        if len(samples) == 0:
            return None
        arr = np.vstack(samples).astype(np.float32)
        c = arr.mean(axis=0).astype(np.float32)
        c /= (float(np.linalg.norm(c)) + 1e-6)
        return c

    def clear_target_model():
        nonlocal target_samples, target_center
        target_samples = []
        target_center = None
        print("[INFO] Cleared TARGET model (in-memory).")

    # ---- Tracking ----
    tracked: Dict[int, TrackedObj] = {}
    next_id = 1
    frame_idx = 0

    # ---- Freeze + drag ROI ----
    frozen = False
    frozen_frame = None
    dragging = False
    drag_start = (0, 0)
    drag_end = (0, 0)
    roi_box: Optional[Tuple[int, int, int, int]] = None

    def normalize_box(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        x = min(x1, x2)
        y = min(y1, y2)
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        return x, y, w, h

    def on_mouse(event, x, y, flags, param):
        nonlocal dragging, drag_start, drag_end, roi_box
        if not frozen:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            dragging = True
            drag_start = (x, y)
            drag_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and dragging:
            drag_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP and dragging:
            dragging = False
            drag_end = (x, y)
            x0, y0, w0, h0 = normalize_box(drag_start, drag_end)
            roi_box = (x0, y0, w0, h0) if (w0 >= 10 and h0 >= 10) else None
            if roi_box is not None:
                print(f"[INFO] ROI selected: x={x0}, y={y0}, w={w0}, h={h0}")

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print("[INFO] Keys: q/ESC quit | f freeze (drag ROI to teach TARGET) | c clear TARGET model")
    print("[INFO] Must see a candidate for ~2 seconds before it can trigger.")
    print("[INFO] Low confidence => highlighted in magenta + terminal prompt (t/o/n/s/c).")

    try:
        while True:
            now = time.time()
            if not frozen:
                ok, frame = cap.read()
                if not ok:
                    break
            else:
                frame = frozen_frame.copy()

            frame_idx += 1

            # Candidates
            boxes = detect_candidate_boxes(frame)

            # Candidate features
            dets: List[Tuple[int, int, int, int, int, int, np.ndarray]] = []  # x,y,w,h,cx,cy,feat
            for (x, y, w, h) in boxes[:25]:
                feat = extract_feature(frame, (x, y, w, h))
                if feat is None:
                    continue
                cx = x + w // 2
                cy = y + h // 2
                dets.append((x, y, w, h, cx, cy, feat))

            # Associate detections to tracked objects
            det_used = [False] * len(dets)
            ids = list(tracked.keys())
            pos = {oid: (tracked[oid].cx, tracked[oid].cy) for oid in ids}

            matched: Dict[int, int] = {}
            for di, (_, _, _, _, cx, cy, _) in enumerate(dets):
                best_id = None
                best_d = 1e9
                for oid in ids:
                    d = ((cx - pos[oid][0]) ** 2 + (cy - pos[oid][1]) ** 2) ** 0.5
                    if d < best_d:
                        best_d = d
                        best_id = oid
                if best_id is not None and best_d <= MATCH_DIST_PX and best_id not in matched:
                    matched[best_id] = di
                    det_used[di] = True

            # Update matched
            for oid, di in matched.items():
                x, y, w, h, cx, cy, feat = dets[di]
                obj = tracked[oid]
                obj.cx, obj.cy, obj.w, obj.h = cx, cy, w, h
                obj.last_seen_frame = frame_idx
                obj.last_update_t = now
                obj.feat = feat

            # Create new objects
            for di, used in enumerate(det_used):
                if used:
                    continue
                x, y, w, h, cx, cy, feat = dets[di]
                oid = next_id
                next_id += 1
                tracked[oid] = TrackedObj(
                    obj_id=oid,
                    cx=cx, cy=cy, w=w, h=h,
                    last_seen_frame=frame_idx,
                    first_seen_t=now,
                    last_update_t=now,
                    feat=feat,
                    label="UNASSIGNED",
                    acted=False
                )

            # Remove stale
            stale = [oid for oid, o in tracked.items() if frame_idx - o.last_seen_frame > MISSING_FRAMES_TO_REMOVE]
            for oid in stale:
                del tracked[oid]

            # Learn TARGET from ROI (freeze) and auto-unfreeze
            if frozen and roi_box is not None:
                x, y, w, h = roi_box
                roi_box = None
                feat = extract_feature(frame, (x, y, w, h))
                if feat is not None:
                    target_samples.append(feat.astype(np.float32))
                    if len(target_samples) > MAX_TARGET_SAMPLES:
                        target_samples = target_samples[-MAX_TARGET_SAMPLES:]
                    target_center = update_center_from_samples(target_samples)
                    print(f"[LEARN] Added TARGET sample. Samples={len(target_samples)}")

                    frozen = False
                    frozen_frame = None
                    dragging = False
                    print("[INFO] Unfrozen (auto) — resume detection.")

                    # mark currently-visible candidates as SEEN so only NEW arrivals trigger
                    for o in tracked.values():
                        o.acted = True
                        o.label = "SEEN"
                else:
                    print("[WARN] Could not extract features from ROI.")

            # ---------- Base visualization ----------
            vis = frame.copy()
            if target_center is None:
                msg = "NO TARGET: press F to freeze, drag ROI around TARGET"
            else:
                msg = f"TARGET learned: {len(target_samples)} samples | waiting for NEW objects (>=2s)"
            if frozen:
                msg = "FROZEN: drag ROI around TARGET (auto-unfreeze after selection)"
            cv2.putText(vis, msg, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            for (x, y, w, h) in boxes[:25]:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 2)

            for oid, o in tracked.items():
                x1 = int(o.cx - o.w / 2)
                y1 = int(o.cy - o.h / 2)
                age = max(0.0, now - o.first_seen_t)
                cv2.putText(
                    vis,
                    f"id{oid} {o.label} age={age:.1f}s acted={o.acted}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1
                )

            if frozen and dragging:
                x0, y0, w0, h0 = normalize_box(drag_start, drag_end)
                cv2.rectangle(vis, (x0, y0), (x0 + w0, y0 + h0), (0, 255, 255), 2)

            cv2.imshow(WINDOW_NAME, vis)

            # ---------- Classification / motion ----------
            if target_center is not None:
                for oid, o in list(tracked.items()):
                    if o.acted:
                        continue

                    # Must be visible for 2 seconds before it counts
                    if (now - o.first_seen_t) < SEEN_SECONDS_TO_ACT:
                        continue

                    # Auto-ignore if matches NOT_OBJECT model
                    if not_object_center is not None:
                        d_no = feature_distance(o.feat, not_object_center)
                        conf_no = dist_to_conf(d_no)
                        if conf_no >= CONF_THRESH_NOT_OBJECT:
                            o.label = "IGNORED"
                            o.acted = True
                            continue

                    # Target distance/conf
                    dist = feature_distance(o.feat, target_center)
                    conf = dist_to_conf(dist)

                    if conf >= CONF_THRESH_TARGET:
                        final_label = LABEL_TARGET
                    elif conf <= CONF_THRESH_OTHER:
                        final_label = LABEL_OTHER
                    else:
                        # Highlight before prompting
                        vis2 = frame.copy()
                        x1 = int(o.cx - o.w / 2)
                        y1 = int(o.cy - o.h / 2)
                        cv2.rectangle(vis2, (x1, y1), (x1 + o.w, y1 + o.h), (255, 0, 255), 4)
                        cv2.putText(vis2, f"HITL id={oid}", (x1, max(20, y1 - 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
                        cv2.putText(vis2, f"age={(now-o.first_seen_t):.1f}s conf={conf*100:.1f}%",
                                    (x1, y1 + o.h + 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
                        cv2.imshow(WINDOW_NAME, vis2)
                        cv2.waitKey(1)

                        action = ask_user_action_lowconf(obj_id=oid, conf=conf)

                        if action == "CLEAR":
                            clear_target_model()
                            o.label = "CLEARED"
                            o.acted = True
                            continue
                        if action == "SKIP":
                            o.label = "SKIP"
                            o.acted = True
                            continue
                        if action == "NOT_OBJECT":
                            not_object_samples.append(o.feat.astype(np.float32))
                            if len(not_object_samples) > MAX_NOT_OBJECT_SAMPLES:
                                not_object_samples = not_object_samples[-MAX_NOT_OBJECT_SAMPLES:]
                            not_object_center = update_center_from_samples(not_object_samples)
                            o.label = "IGNORED_TRAINED"
                            o.acted = True
                            continue

                        final_label = action

                    o.label = final_label
                    o.acted = True

                    # Update models from label (in-memory only)
                    if final_label == LABEL_TARGET:
                        target_samples.append(o.feat.astype(np.float32))
                        if len(target_samples) > MAX_TARGET_SAMPLES:
                            target_samples = target_samples[-MAX_TARGET_SAMPLES:]
                        target_center = update_center_from_samples(target_samples)

                    # RoboDK action
                    Xmm, Ymm = apply_homography(H, (float(o.cx), float(o.cy)))
                    drop_app_t, drop_t = (None, None)
                    if final_label in LABEL_TO_DROP_TARGETS:
                        app_name, drop_name = LABEL_TO_DROP_TARGETS[final_label]
                        drop_app_t, drop_t = drop_targets.get(final_label, (None, None))

                    print(f"[ACTION] id={oid} -> {final_label} (age={now-o.first_seen_t:.1f}s conf={conf*100:.1f}%, dist={dist:.3f}) at XY=({Xmm:.1f},{Ymm:.1f})")

                    move_pick_and_optional_drop(
                        robot=robot,
                        base_T_board=base_T_board,
                        pick_ref_pose_abs=pick_ref_pose_abs,
                        Xmm=Xmm,
                        Ymm=Ymm,
                        t_app=t_pick_app,
                        t_pick=t_pick,
                        drop_app_t=drop_app_t,
                        drop_t=drop_t,
                        do_pick=DO_PICK_MOTION,
                        do_drop=DO_DROP_MOTION
                    )

            # ---------- keys ----------
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

            if key in (ord("f"), ord("F")):
                frozen = True
                frozen_frame = frame.copy()
                dragging = False
                roi_box = None
                print("[INFO] Frozen. Drag ROI around TARGET. (Auto-unfreeze after selection)")

            if key in (ord("c"), ord("C")):
                clear_target_model()
                # allow re-processing
                for o in tracked.values():
                    o.acted = False
                    o.label = "UNASSIGNED"

    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()