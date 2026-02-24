import cv2
import numpy as np
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from robodk import robolink, robomath


# ============================================================
# USER SETTINGS (RoboDK)
# ============================================================

BOARD_FRAME_NAME = "BoardFrame"
PICK_REF_TARGET_NAME = "PickRef"
H_FILE = r"C:\Users\noah\Desktop\Python Reference\HITLDT\homography.npy"  # <-- update

# Z heights in mm (relative to BoardFrame)
Z_PICK_MM = 50.0
Z_APPROACH_MM = 100.0

# Motion style
DO_PICK_MOTION = True         # approach->pick->retract at object location
DO_DROP_MOTION = True         # then move to bin drop targets (if they exist)

# Targets for bin drop (must exist in RoboDK station, or drop is skipped)
BIN_TO_DROP_TARGETS = {
    "BIN_1": ("BIN_1_DROP_APP", "BIN_1_DROP"),
    "BIN_2": ("BIN_2_DROP_APP", "BIN_2_DROP"),
    "BIN_3": ("BIN_3_DROP_APP", "BIN_3_DROP"),
}

# Names for dynamic pick targets we will update each time
DYN_PICK_APP_NAME = "CUR_PICK_APP"
DYN_PICK_NAME = "CUR_PICK"


# ============================================================
# USER SETTINGS (Camera)
# ============================================================

CAMERA_INDEX = 1
FRAME_W = 640
FRAME_H = 480


# ============================================================
# VISION SETTINGS (Edge-based detection + robust color sampling)
# ============================================================

# Geometry filters
MIN_AREA_PX = 150
MAX_AREA_PX = 150000
ASPECT_MIN = 0.70
ASPECT_MAX = 1.35
SOLIDITY_MIN = 0.90
APPROX_EPS_FRAC = 0.03

# Color sampling: sample only the interior (avoid edge/background bleeding)
INTERIOR_ERODE_PX = 8
SAT_MIN = 15
VAL_MIN = 15

# Tracking
MATCH_DIST_PX = 90
MISSING_FRAMES_TO_REMOVE = 20
STABLE_FRAMES_TO_ACT = 6  # require object to be stable before triggering motion

# Training
N_BINS = 3
TRAIN_SECONDS = 3.0
TRAIN_MIN_OBJECTS = 3
TRAIN_MIN_SAMPLES_PER_OBJ = 25
MIN_CENTER_SEPARATION = 0.18  # in feature space; prevents “3 bins = same color”

# Unknown detection
UNKNOWN_DIST_THRESH = 0.35  # feature-space threshold; tune
RULES_JSON = "unknown_color_rules.json"


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
    stable_frames: int
    last_update_t: float

    mean_hsv: np.ndarray          # (3,)
    feat: np.ndarray              # (4,) circular hue embedding + S + V
    bin_name: str
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
    # Build pick poses (in abs)
    board_T_pick = robomath.transl(Xmm, Ymm, Z_PICK_MM)
    board_T_app  = robomath.transl(Xmm, Ymm, Z_PICK_MM + Z_APPROACH_MM)

    pick_pose_abs = base_T_board * board_T_pick
    app_pose_abs  = base_T_board * board_T_app

    pick_pose_abs = pose_with_ref_rotation(pick_ref_pose_abs, pick_pose_abs)
    app_pose_abs  = pose_with_ref_rotation(pick_ref_pose_abs, app_pose_abs)

    # Update dynamic targets
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
# Rules for unknown colors
# ============================================================

def load_rules() -> Dict[str, str]:
    try:
        with open(RULES_JSON, "r") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception:
        return {}

def save_rules(rules: Dict[str, str]):
    with open(RULES_JSON, "w") as f:
        json.dump(rules, f, indent=2)

def make_sig(mean_hsv: np.ndarray) -> str:
    h = int(mean_hsv[0] // 8)
    s = int(mean_hsv[1] // 16)
    v = int(mean_hsv[2] // 16)
    return f"h{h}_s{s}_v{v}"

def ask_user_unknown_action(sig: str, suggested: str) -> str:
    """
    Returns:
      - BIN_1/2/3
      - SKIP
      - RETRAIN
    """
    print(f"\n[HITL] Unknown color encountered (signature={sig}).")
    print(f"[HITL] Nearest bin guess: {suggested}")
    print("[HITL] What should I do?")
    print("  1) Assign to BIN_1")
    print("  2) Assign to BIN_2")
    print("  3) Assign to BIN_3")
    print("  s) Skip (no motion)")
    print("  r) Retrain bins")
    while True:
        ans = input("Choice (1/2/3/s/r): ").strip().lower()
        if ans in ("1", "2", "3"):
            return f"BIN_{ans}"
        if ans == "s":
            return "SKIP"
        if ans == "r":
            return "RETRAIN"
        print("Invalid input. Try again.")


# ============================================================
# Vision: edge-based detection + interior color feature
# ============================================================

def detect_objects_edges(frame_bgr) -> List[Tuple[int, int, int, int, np.ndarray]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    med = np.median(gray)
    lo = int(max(0, 0.66 * med))
    hi = int(min(255, 1.33 * med))
    edges = cv2.Canny(gray, lo, hi)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, k, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dets = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA_PX or area > MAX_AREA_PX:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 0 or h <= 0:
            continue

        aspect = w / float(h)
        if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
            continue

        hull = cv2.convexHull(cnt)
        hull_area = max(1.0, cv2.contourArea(hull))
        sol = area / hull_area
        if sol < SOLIDITY_MIN:
            continue

        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, APPROX_EPS_FRAC * peri, True)
        # allow some noise; blocks are often ~4 corners but can be more under blur
        if not (4 <= len(approx) <= 20):
            continue

        cx = x + w // 2
        cy = y + h // 2
        dets.append((cx, cy, w, h, cnt))

    return dets

def roi_hsv_and_mask_from_contour(frame_bgr, contour: np.ndarray):
    H, W = frame_bgr.shape[:2]
    x, y, bw, bh = cv2.boundingRect(contour)
    pad = 2
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + bw + pad, W - 1)
    y2 = min(y + bh + pad, H - 1)

    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None, None

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    shifted = contour.copy()
    shifted[:, 0, 0] -= x1
    shifted[:, 0, 1] -= y1

    mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [shifted], -1, 255, thickness=-1)
    return hsv_roi, mask

def mean_hsv_interior(hsv_roi: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    if hsv_roi is None or mask is None:
        return None

    if INTERIOR_ERODE_PX > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * INTERIOR_ERODE_PX + 1, 2 * INTERIOR_ERODE_PX + 1)
        )
        mask_i = cv2.erode(mask, k, iterations=1)
    else:
        mask_i = mask

    pix = hsv_roi[mask_i > 0]
    if pix.size == 0:
        return None

    sat = pix[:, 1]
    val = pix[:, 2]
    keep = (sat >= SAT_MIN) & (val >= VAL_MIN)
    pix = pix[keep]
    if pix.shape[0] < 50:
        return None

    return pix.mean(axis=0).astype(np.float32)

def hsv_to_feature(mean_hsv: np.ndarray) -> np.ndarray:
    """
    Circular hue embedding to prevent 'red wrap' issues:
      feature = [cos(theta), sin(theta), S_norm, V_norm]
    """
    H, S, V = float(mean_hsv[0]), float(mean_hsv[1]), float(mean_hsv[2])
    theta = 2.0 * np.pi * (H / 180.0)
    return np.array([np.cos(theta), np.sin(theta), S / 255.0, V / 255.0], dtype=np.float32)

def feature_to_hsv(center_feat: np.ndarray) -> np.ndarray:
    """
    Convert feature center back to HSV for display swatch.
    """
    c, s = float(center_feat[0]), float(center_feat[1])
    ang = np.arctan2(s, c)
    if ang < 0:
        ang += 2.0 * np.pi
    H = (ang * 180.0 / (2.0 * np.pi))
    S = float(np.clip(center_feat[2] * 255.0, 0, 255))
    V = float(np.clip(center_feat[3] * 255.0, 0, 255))
    return np.array([H, S, V], dtype=np.float32)

def hsv_patch_to_bgr(hsv_center: np.ndarray):
    hsv_px = np.uint8([[[int(hsv_center[0]) % 180, int(hsv_center[1]), int(hsv_center[2])]]])
    return cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0, 0].tolist()

def train_bins_kmeans(features: np.ndarray) -> np.ndarray:
    """
    features: (N,4)
    returns centers: (3,4), sorted by hue angle
    """
    features = features.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.2)
    _, _, centers = cv2.kmeans(features, N_BINS, None, criteria, 8, cv2.KMEANS_PP_CENTERS)

    # sort by hue angle for stable BIN numbering
    hues = []
    for c in centers:
        hsv = feature_to_hsv(c)
        hues.append(hsv[0])
    order = np.argsort(np.array(hues))
    return centers[order].astype(np.float32)

def assign_bin(feature: np.ndarray, centers: np.ndarray) -> Tuple[str, float]:
    d = np.linalg.norm(centers - feature[None, :], axis=1)
    idx = int(np.argmin(d))
    return f"BIN_{idx+1}", float(d[idx])


# ============================================================
# MAIN
# ============================================================

def main():
    rules = load_rules()

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
    for b in ("BIN_1", "BIN_2", "BIN_3"):
        app_name, drop_name = BIN_TO_DROP_TARGETS[b]
        drop_app_t = require_target(rdk, app_name)
        drop_t = require_target(rdk, drop_name)
        if drop_app_t is None or drop_t is None:
            print(f"[WARN] Missing drop targets for {b}: {app_name}, {drop_name} (drop will be skipped for this bin)")
        drop_targets[b] = (drop_app_t, drop_t)

    # ---- Camera init ----
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # ---- State ----
    tracked: Dict[int, TrackedObj] = {}
    per_obj_features: Dict[int, List[np.ndarray]] = {}
    next_id = 1
    frame_idx = 0

    trained = False
    centers: Optional[np.ndarray] = None
    train_start = time.time()

    print("[INFO] Phase 1: TRAINING — place your 3 majority colors in view, separated.")
    print(f"[INFO] Training time: {TRAIN_SECONDS:.1f}s (will auto-train).")
    print("[INFO] Phase 2: INFERENCE — add NEW parts; system classifies and moves RoboDK robot.")
    print("[INFO] HITL: unknown colors prompt you for action (assign/skip/retrain).")
    print("[INFO] Keys: q/ESC quit | r retrain bins")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        now = time.time()

        dets = detect_objects_edges(frame)  # (cx,cy,w,h,cnt)

        # --- Greedy association ---
        det_used = [False] * len(dets)
        ids = list(tracked.keys())
        pos = {oid: (tracked[oid].cx, tracked[oid].cy) for oid in ids}

        matched: Dict[int, int] = {}
        for di, (cx, cy, w, h, cnt) in enumerate(dets):
            best_id = None
            best_d = 1e9
            for oid in ids:
                d = ((cx - pos[oid][0])**2 + (cy - pos[oid][1])**2) ** 0.5
                if d < best_d:
                    best_d = d
                    best_id = oid
            if best_id is not None and best_d <= MATCH_DIST_PX and best_id not in matched:
                matched[best_id] = di
                det_used[di] = True

        # --- Update matched ---
        for oid, di in matched.items():
            cx, cy, w, h, cnt = dets[di]
            hsv_roi, mask = roi_hsv_and_mask_from_contour(frame, cnt)
            mean = mean_hsv_interior(hsv_roi, mask)
            if mean is None:
                # keep old color if this frame is bad
                mean = tracked[oid].mean_hsv
            feat = hsv_to_feature(mean)

            obj = tracked[oid]
            obj.cx, obj.cy, obj.w, obj.h = cx, cy, w, h
            obj.last_seen_frame = frame_idx
            obj.last_update_t = now
            obj.stable_frames += 1
            obj.mean_hsv = mean
            obj.feat = feat

            if not trained:
                per_obj_features.setdefault(oid, []).append(feat)

        # --- Create new objects ---
        for di, used in enumerate(det_used):
            if used:
                continue
            cx, cy, w, h, cnt = dets[di]
            hsv_roi, mask = roi_hsv_and_mask_from_contour(frame, cnt)
            mean = mean_hsv_interior(hsv_roi, mask)
            if mean is None:
                continue
            feat = hsv_to_feature(mean)

            oid = next_id
            next_id += 1

            tracked[oid] = TrackedObj(
                obj_id=oid, cx=cx, cy=cy, w=w, h=h,
                last_seen_frame=frame_idx,
                stable_frames=1,
                last_update_t=now,
                mean_hsv=mean,
                feat=feat,
                bin_name="UNASSIGNED",
                acted=False
            )

            if not trained:
                per_obj_features.setdefault(oid, []).append(feat)

        # --- Cleanup stale ---
        stale = []
        for oid, obj in tracked.items():
            if frame_idx - obj.last_seen_frame > MISSING_FRAMES_TO_REMOVE:
                stale.append(oid)
        for oid in stale:
            del tracked[oid]
            per_obj_features.pop(oid, None)

        # --- Training finish ---
        if not trained:
            if (now - train_start) >= TRAIN_SECONDS:
                reps = []
                for oid, feats in per_obj_features.items():
                    if len(feats) < TRAIN_MIN_SAMPLES_PER_OBJ:
                        continue
                    arr = np.vstack(feats)  # (T,4)
                    rep = np.median(arr, axis=0).astype(np.float32)
                    reps.append(rep)

                if len(reps) >= TRAIN_MIN_OBJECTS:
                    reps = np.vstack(reps)
                    centers = train_bins_kmeans(reps)

                    # Ensure centers are not collapsing
                    pairwise = []
                    for i in range(3):
                        for j in range(i + 1, 3):
                            pairwise.append(float(np.linalg.norm(centers[i] - centers[j])))
                    min_sep = min(pairwise) if pairwise else 0.0

                    if min_sep < MIN_CENTER_SEPARATION:
                        print(f"[WARN] Centers too close (min sep={min_sep:.3f}). Keep 3 different colors visible; retraining...")
                        train_start = time.time()
                        per_obj_features = {}
                    else:
                        trained = True
                        print("\n[INFO] MODEL TRAINED: learned 3 dominant color bins (BIN_1..BIN_3).")
                        print("[INFO] Existing objects will NOT trigger motion. Only NEW stable objects after this point will.\n")

                        # mark everything currently visible as already handled
                        for obj in tracked.values():
                            obj.acted = True
                            obj.bin_name = "TRAINING_SET"

                else:
                    print("[WARN] Not enough stable objects to train. Keep 3 blocks visible and separated; continuing training...")
                    train_start = time.time()

        # --- Inference / Motion (only for NEW stable objects) ---
        if trained and centers is not None:
            for oid, obj in list(tracked.items()):
                if obj.acted:
                    continue
                if obj.stable_frames < STABLE_FRAMES_TO_ACT:
                    continue

                # classify
                bin_guess, dmin = assign_bin(obj.feat, centers)
                sig = make_sig(obj.mean_hsv)
                forced = rules.get(sig)

                final_bin = None

                if (dmin > UNKNOWN_DIST_THRESH) and (forced is None):
                    action = ask_user_unknown_action(sig=sig, suggested=bin_guess)
                    if action == "RETRAIN":
                        trained = False
                        centers = None
                        train_start = time.time()
                        per_obj_features = {}
                        for o in tracked.values():
                            o.acted = False
                            o.bin_name = "UNASSIGNED"
                        print("\n[INFO] Retraining started. Show your 3 majority colors again.")
                        break  # exit this inference pass
                    elif action == "SKIP":
                        obj.bin_name = "UNKNOWN_SKIP"
                        obj.acted = True
                        continue
                    else:
                        # user assigned to a bin; remember for next time
                        rules[sig] = action
                        save_rules(rules)
                        final_bin = action
                else:
                    final_bin = forced if forced is not None else bin_guess

                obj.bin_name = final_bin
                obj.acted = True

                # Move RoboDK
                Xmm, Ymm = apply_homography(H, (float(obj.cx), float(obj.cy)))
                drop_app_t, drop_t = drop_targets.get(final_bin, (None, None))

                print(f"[ACTION] id={oid} -> {final_bin} (d={dmin:.3f}) at XY=({Xmm:.1f},{Ymm:.1f})")

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

        # --- Visualization ---
        vis = frame.copy()

        if not trained:
            remaining = max(0.0, TRAIN_SECONDS - (now - train_start))
            cv2.putText(vis, "TRAINING: show 3 majority colors separated",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(vis, f"training in {remaining:.1f}s",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(vis, "TRAINED: new stable object -> classify -> RoboDK move",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        if trained and centers is not None:
            y0 = 70
            for i in range(3):
                hsv_c = feature_to_hsv(centers[i])
                bgr = hsv_patch_to_bgr(hsv_c)
                cv2.rectangle(vis, (20, y0 + i*28), (60, y0 + i*28 + 18), bgr, -1)
                cv2.putText(vis, f"BIN_{i+1}", (70, y0 + i*28 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        for (cx, cy, w, h, cnt) in dets:
            x = int(cx - w/2)
            y = int(cy - h/2)
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 200, 0), 2)

        for oid, obj in tracked.items():
            x1 = int(obj.cx - obj.w/2)
            y1 = int(obj.cy - obj.h/2)
            cv2.putText(vis, f"id{oid} {obj.bin_name} st={obj.stable_frames} acted={obj.acted}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        cv2.imshow("RoboDK + Trained Bins + HITL", vis)
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break
        if key in (ord("r"), ord("R")):
            trained = False
            centers = None
            train_start = time.time()
            per_obj_features = {}
            for o in tracked.values():
                o.acted = False
                o.bin_name = "UNASSIGNED"
            print("\n[INFO] Manual retrain triggered. Show your 3 majority colors again.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()