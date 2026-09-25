"""
robodk_hitl_combined.py
=======================
Combines:
  1. RoboDK peg-targeting (homography + BoardFrame)
  2. HITL color-bin detection (train 3 bins, edge-based block detection)

Workflow
--------
TRAINING PHASE
  - Edge-based block detection accumulates HSV samples per tracked block.
  - After TRAIN_SECONDS with >= TRAIN_MIN_OBJECTS stable blocks, k-means builds
    3 bin centers.  RoboDK targets are NOT updated yet.

INFERENCE PHASE
  - Each NEW detected block is classified into BIN_1/2/3.
  - Unknown colors trigger a terminal HITL prompt.
  - After classification the robot moves:
      Approach target  →  Pick target  →  Retract (approach)
    where the target position comes from the block's pixel position mapped
    through the homography into the RoboDK BoardFrame.
  - Press M to move to the largest detected blue peg (legacy behaviour).
  - Press T to force-retrain at any time.
  - Press Q / ESC to quit.
"""

# ─────────────────────────── stdlib / third-party ────────────────────────────
import cv2
import json
import socket
import time
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# ─────────────────────────── RoboDK ──────────────────────────────────────────
from robodk import robolink, robomath

# ═════════════════════════════════════════════════════════════════════════════
# USER SETTINGS
# ═════════════════════════════════════════════════════════════════════════════

# ── RoboDK ────────────────────────────────────────────────────────────────────
BOARD_FRAME_NAME  = "BoardFrame"
Z_PICK_MM         = 50.0
Z_APPROACH_MM     = 100.0
H_FILE            = r"C:\Users\noah\Desktop\Python Reference\HITLDT\homography.npy"

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX = 1          # change to 0 if external cam is on a different index
FRAME_W      = 640
FRAME_H      = 480

# ── Block / peg detection geometry ────────────────────────────────────────────
MIN_AREA_PX    = 150
MAX_AREA_PX    = 150_000
ASPECT_MIN     = 0.70
ASPECT_MAX     = 1.35
SOLIDITY_MIN   = 0.9
APPROX_EPS_FRAC = 0.03

# ── Peg-circle detection (kept for legacy blue-peg manual mode) ───────────────
MIN_AREA_CIRCLE = 250
MIN_R  = 10
MAX_R  = 45
MIN_CIRC = 0.70
MIN_FILL = 0.55
RAD_TOL  = 0.25

COLOR_RANGES: Dict[str, List[Tuple[Tuple[int,int,int], Tuple[int,int,int]]]] = {
    "B": [((95,  80,  50), (135, 255, 255))],
    "Y": [((18,  50, 170), ( 40, 255, 255))],
    "O": [(( 3,  80,  50), ( 22, 255, 255))],
    "W": [(( 0,   0, 200), (179,  50, 255))],
}
DRAW_COLORS_BGR: Dict[str, Tuple[int,int,int]] = {
    "B": (255, 0,   0),
    "Y": (0,   255, 255),
    "O": (0,   165, 255),
    "W": (255, 255, 255),
}

# ── Tracking ──────────────────────────────────────────────────────────────────
MATCH_DIST_PX            = 90
MISSING_FRAMES_TO_REMOVE = 15

# ── Training ──────────────────────────────────────────────────────────────────
N_BINS             = 3
TRAIN_SECONDS      = 3.0
TRAIN_MIN_OBJECTS  = 3
TRAIN_MIN_SAMPLES  = 30   # per tracked block

# ── Color interior sampling ───────────────────────────────────────────────────
INTERIOR_ERODE_PX = 8
SAT_MIN           = 15
VAL_MIN           = 15

# ── Unknown / HITL ────────────────────────────────────────────────────────────
UNKNOWN_DIST_THRESH = 18.0
RULES_JSON          = "unknown_color_rules.json"

# ── Robot TCP (optional emulator / real controller) ───────────────────────────
ROBOT_IP        = "127.0.0.1"
ROBOT_PORT      = 1234
CMD_TERMINATOR  = "\n"
TCP_TIMEOUT_S   = 2.0
BIN_TO_CMD: Dict[str, str] = {
    "BIN_1": "moveP1",
    "BIN_2": "moveP2",
    "BIN_3": "moveP3",
}

# ═════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackedBlock:
    block_id:        int
    cx:              int
    cy:              int
    w:               int
    h:               int
    last_seen_frame: int
    mean_hsv:        np.ndarray
    last_update_t:   float
    bin_name:        str
    acted:           bool

# ═════════════════════════════════════════════════════════════════════════════
# TCP EMULATOR HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def send_to_emulator(cmd: str) -> bool:
    payload = (cmd + CMD_TERMINATOR).encode("utf-8", errors="ignore")
    try:
        with socket.create_connection((ROBOT_IP, ROBOT_PORT), timeout=TCP_TIMEOUT_S) as s:
            s.sendall(payload)
            try:
                s.shutdown(socket.SHUT_WR)
            except OSError:
                pass
        print(f"[ROBOT-TCP] SENT: {cmd!r}")
        return True
    except Exception as e:
        print(f"[ROBOT-TCP] SEND FAILED (non-fatal): {e}")
        return False

# ═════════════════════════════════════════════════════════════════════════════
# HITL / RULES HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def load_rules() -> Dict[str, str]:
    try:
        with open(RULES_JSON) as f:
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
    print(f"\n[HITL] Unknown colour (sig={sig}). Nearest bin guess: {suggested}")
    print("  1 → BIN_1   2 → BIN_2   3 → BIN_3   s → Skip   r → Retrain")
    while True:
        ans = input("Choice (1/2/3/s/r): ").strip().lower()
        if ans in ("1", "2", "3"):
            return f"BIN_{ans}"
        if ans == "s":
            return "SKIP"
        if ans == "r":
            return "RETRAIN"
        print("Invalid – try again.")

# ═════════════════════════════════════════════════════════════════════════════
# HOMOGRAPHY
# ═════════════════════════════════════════════════════════════════════════════

def load_homography() -> np.ndarray:
    try:
        H = np.load(H_FILE)
        if H.shape != (3, 3):
            raise ValueError("homography.npy is not 3×3")
        print(f"[INFO] Loaded homography: {H_FILE}")
        return H.astype(np.float64)
    except Exception as e:
        print(f"[WARN] Homography not loaded: {e}")
        print("[WARN] Targets will be WRONG until you calibrate homography.")
        return np.eye(3, dtype=np.float64)

def apply_homography(H: np.ndarray, uv: Tuple[float, float]) -> Tuple[float, float]:
    u, v = uv
    pt  = np.array([[[u, v]]], dtype=np.float32)
    xy  = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(xy[0]), float(xy[1])

# ═════════════════════════════════════════════════════════════════════════════
# ROBODK HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def require_frame(rdk: robolink.Robolink, name: str):
    f = rdk.Item(name, robolink.ITEM_TYPE_FRAME)
    if not f.Valid():
        raise RuntimeError(f"Frame '{name}' not found. Create it in RoboDK.")
    return f

def get_or_create_target(rdk: robolink.Robolink, name: str):
    t = rdk.Item(name, robolink.ITEM_TYPE_TARGET)
    if not t.Valid():
        t = rdk.AddTarget(name)
    return t

def pose_with_ref_rotation(ref_pose_abs, new_pose_abs):
    """Copy orientation from ref, position from new."""
    out = ref_pose_abs.copy()
    out.setPos(new_pose_abs.Pos())
    return out

def build_pick_app_poses(base_T_board, pick_ref_pose, Xmm: float, Ymm: float):
    board_T_pick = robomath.transl(Xmm, Ymm, Z_PICK_MM)
    board_T_app  = robomath.transl(Xmm, Ymm, Z_PICK_MM + Z_APPROACH_MM)
    pick_abs = pose_with_ref_rotation(pick_ref_pose, base_T_board * board_T_pick)
    app_abs  = pose_with_ref_rotation(pick_ref_pose, base_T_board * board_T_app)
    return app_abs, pick_abs

# ═════════════════════════════════════════════════════════════════════════════
# EDGE-BASED BLOCK DETECTION
# ═════════════════════════════════════════════════════════════════════════════

def detect_blocks_edges(frame_bgr) -> List[Tuple[int, int, int, int, np.ndarray]]:
    gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray  = cv2.GaussianBlur(gray, (5, 5), 0)
    v     = np.median(gray)
    lo    = int(max(0,   0.66 * v))
    hi    = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, lo, hi)
    k     = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, k, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []
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
        hull     = cv2.convexHull(cnt)
        hull_a   = max(1.0, cv2.contourArea(hull))
        solidity = area / hull_a
        if solidity < SOLIDITY_MIN:
            continue
        peri  = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, APPROX_EPS_FRAC * peri, True)
        if not (4 <= len(approx) <= 8):
            continue
        cx = x + w // 2
        cy = y + h // 2
        detections.append((cx, cy, w, h, cnt))
    return detections

def roi_hsv_mask(frame_bgr, contour: np.ndarray):
    H, W = frame_bgr.shape[:2]
    x, y, bw, bh = cv2.boundingRect(contour)
    pad = 2
    x1, y1 = max(x - pad, 0), max(y - pad, 0)
    x2, y2 = min(x + bw + pad, W - 1), min(y + bh + pad, H - 1)
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

def mean_hsv_interior(hsv_roi, mask) -> Optional[np.ndarray]:
    if hsv_roi is None or mask is None:
        return None
    if INTERIOR_ERODE_PX > 0:
        k      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                    (2*INTERIOR_ERODE_PX+1, 2*INTERIOR_ERODE_PX+1))
        mask_i = cv2.erode(mask, k, iterations=1)
    else:
        mask_i = mask
    pix = hsv_roi[mask_i > 0]
    if pix.size == 0:
        return None
    keep = (pix[:, 1] >= SAT_MIN) & (pix[:, 2] >= VAL_MIN)
    pix  = pix[keep]
    if pix.shape[0] < 50:
        return None
    return pix.mean(axis=0).astype(np.float32)

# ═════════════════════════════════════════════════════════════════════════════
# LEGACY: BLUE PEG (CIRCLE) DETECTION  (for manual M-key mode)
# ═════════════════════════════════════════════════════════════════════════════

def _circularity(contour) -> float:
    area = float(cv2.contourArea(contour))
    per  = float(cv2.arcLength(contour, True))
    return 0.0 if per <= 1e-6 else float(4.0 * np.pi * area / (per * per))

def detect_blue_pegs(frame_bgr):
    hsv  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lo, hi in COLOR_RANGES["B"]:
        mask = cv2.bitwise_or(mask,
                cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8)))
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    k    = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    circles = []
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for label in range(1, n):
        x, y, w, h, area = stats[label]
        if area < MIN_AREA_CIRCLE:
            continue
        blob = (labels == label).astype(np.uint8) * 255
        cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        if _circularity(c) < MIN_CIRC:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if r < MIN_R or r > MAX_R:
            continue
        fill = float(area) / max(1e-6, np.pi * r * r)
        if fill < MIN_FILL:
            continue
        circles.append((float(cx), float(cy), float(r)))
    if not circles:
        return None
    med = np.median([c[2] for c in circles])
    lo_r, hi_r = med * (1 - RAD_TOL), med * (1 + RAD_TOL)
    circles = [c for c in circles if lo_r <= c[2] <= hi_r]
    return max(circles, key=lambda c: c[2]) if circles else None  # largest r

# ═════════════════════════════════════════════════════════════════════════════
# K-MEANS TRAINING + ASSIGNMENT
# ═════════════════════════════════════════════════════════════════════════════

def dist_hsv(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(np.float32) - b.astype(np.float32)))

def train_bins(samples: np.ndarray) -> np.ndarray:
    samples   = samples.astype(np.float32)
    criteria  = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.2)
    _, _, centers = cv2.kmeans(samples, N_BINS, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    order = np.argsort(centers[:, 0])
    return centers[order].astype(np.float32)

def assign_bin(mean_hsv: np.ndarray, centers: np.ndarray) -> Tuple[str, float]:
    dists = [dist_hsv(mean_hsv, centers[i]) for i in range(len(centers))]
    idx   = int(np.argmin(dists))
    return f"BIN_{idx+1}", float(dists[idx])

def hsv_to_bgr(hsv: np.ndarray):
    px = np.uint8([[[int(hsv[0]), int(hsv[1]), int(hsv[2])]]])
    return cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0].tolist()

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def reset_training(blocks: dict, per_block_samples: dict):
    for b in blocks.values():
        b.acted    = False
        b.bin_name = "UNASSIGNED"
    per_block_samples.clear()

def main():
    # ── RoboDK setup ──────────────────────────────────────────────────────────
    rdk   = robolink.Robolink()
    robot = rdk.ItemUserPick("Select the robot", robolink.ITEM_TYPE_ROBOT)
    if not robot.Valid():
        raise RuntimeError("No robot selected.")

    board_frame  = require_frame(rdk, BOARD_FRAME_NAME)
    base_T_board = board_frame.PoseAbs()

    pick_ref = rdk.Item("PickRef", robolink.ITEM_TYPE_TARGET)
    if not pick_ref.Valid():
        raise RuntimeError("Create a RoboDK target named 'PickRef' with pick orientation.")
    pick_ref_pose = pick_ref.PoseAbs()

    # Motion targets (reused each detection)
    t_app  = get_or_create_target(rdk, "BIN_PICK_APP")
    t_pick = get_or_create_target(rdk, "BIN_PICK")
    # Legacy blue-peg manual targets
    t_blue_app  = get_or_create_target(rdk, "B_PICK_APP")
    t_blue_pick = get_or_create_target(rdk, "B_PICK")

    H = load_homography()

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    # ── State ─────────────────────────────────────────────────────────────────
    rules:             Dict[str, str]           = load_rules()
    blocks:            Dict[int, TrackedBlock]  = {}
    per_block_samples: Dict[int, List]          = {}
    next_id    = 1
    frame_idx  = 0
    trained    = False
    centers    = None
    training_start = time.time()
    last_blue_poses = None          # (app_abs, pick_abs) for M-key

    print("\n" + "═"*60)
    print("  TRAINING PHASE")
    print("  Place 3 distinctly-coloured blocks, clearly separated.")
    print(f"  Training time: {TRAIN_SECONDS:.1f}s.")
    print("─"*60)
    print("  KEY BINDINGS (focus the OpenCV window first)")
    print("  M  – move to best detected blue peg (manual)")
    print("  T  – force retrain")
    print("  Q / ESC – quit")
    print("═"*60 + "\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        now = time.time()

        vis = frame.copy()

        # ── BLOCK DETECTION ──────────────────────────────────────────────────
        detections = detect_blocks_edges(frame)

        # ── TRACK : associate detections → existing IDs ───────────────────────
        det_used = [False] * len(detections)
        matched: Dict[int, int] = {}

        for di, (cx, cy, w, h, cnt) in enumerate(detections):
            best_id, best_d = None, 1e9
            for bid, blk in blocks.items():
                d = ((cx - blk.cx)**2 + (cy - blk.cy)**2) ** 0.5
                if d < best_d:
                    best_d = d; best_id = bid
            if best_id is not None and best_d <= MATCH_DIST_PX and best_id not in matched:
                matched[best_id] = di
                det_used[di]     = True

        # Update matched blocks
        for bid, di in matched.items():
            cx, cy, w, h, cnt = detections[di]
            hsv_roi, mask      = roi_hsv_mask(frame, cnt)
            mean               = mean_hsv_interior(hsv_roi, mask)
            blk = blocks[bid]
            blk.cx, blk.cy, blk.w, blk.h = cx, cy, w, h
            blk.last_seen_frame = frame_idx
            blk.last_update_t   = now
            if mean is not None:
                blk.mean_hsv = mean
            if not trained and mean is not None:
                per_block_samples.setdefault(bid, []).append(mean)

        # New unmatched detections → new tracked blocks
        for di, used in enumerate(det_used):
            if used:
                continue
            cx, cy, w, h, cnt = detections[di]
            hsv_roi, mask      = roi_hsv_mask(frame, cnt)
            mean               = mean_hsv_interior(hsv_roi, mask)
            if mean is None:
                continue
            bid = next_id; next_id += 1
            blocks[bid] = TrackedBlock(
                block_id=bid, cx=cx, cy=cy, w=w, h=h,
                last_seen_frame=frame_idx, mean_hsv=mean,
                last_update_t=now, bin_name="UNASSIGNED", acted=False
            )
            if not trained:
                per_block_samples.setdefault(bid, []).append(mean)

            # ── INFERENCE: classify + move robot ─────────────────────────────
            if trained and centers is not None:
                bin_guess, dmin = assign_bin(mean, centers)
                sig    = make_sig(mean)
                forced = rules.get(sig)

                if dmin > UNKNOWN_DIST_THRESH and forced is None:
                    # HITL prompt (blocking – robot pauses)
                    action = ask_user_unknown_action(sig=sig, suggested=bin_guess)
                    if action == "RETRAIN":
                        trained = False; centers = None
                        training_start = time.time()
                        reset_training(blocks, per_block_samples)
                        print("\n[INFO] Retraining…")
                        continue
                    elif action == "SKIP":
                        blocks[bid].bin_name = "UNKNOWN_SKIP"
                        blocks[bid].acted    = True
                        continue
                    else:
                        rules[sig] = action; save_rules(rules)
                        final_bin = action
                else:
                    final_bin = forced if forced else bin_guess

                blocks[bid].bin_name = final_bin
                blocks[bid].acted    = True

                # ── Map pixel → board coordinates → RoboDK targets ───────────
                Xmm, Ymm = apply_homography(H, (float(cx), float(cy)))
                app_abs, pick_abs = build_pick_app_poses(
                    base_T_board, pick_ref_pose, Xmm, Ymm)
                t_app.setPoseAbs(app_abs)
                t_pick.setPoseAbs(pick_abs)

                print(f"[RoboDK] Block id={bid} → {final_bin} @ ({Xmm:.1f}, {Ymm:.1f}) mm")
                try:
                    robot.MoveJ(t_app)
                    robot.MoveL(t_pick)
                    robot.MoveL(t_app)
                    print(f"[RoboDK] Move complete for {final_bin}.")
                except Exception as e:
                    print(f"[RoboDK] Move error: {e}")

                # Also send to TCP emulator (non-blocking, best-effort)
                send_to_emulator(BIN_TO_CMD.get(final_bin, "moveUnknown"))

        # ── CLEANUP stale tracks ─────────────────────────────────────────────
        stale = [bid for bid, blk in blocks.items()
                 if frame_idx - blk.last_seen_frame > MISSING_FRAMES_TO_REMOVE]
        for bid in stale:
            del blocks[bid]; per_block_samples.pop(bid, None)

        # ── TRAINING: check if we can train ──────────────────────────────────
        if not trained:
            elapsed = now - training_start
            if elapsed >= TRAIN_SECONDS:
                reps = []
                for bid, samples in per_block_samples.items():
                    if len(samples) < TRAIN_MIN_SAMPLES:
                        continue
                    arr = np.vstack(samples)
                    reps.append(np.median(arr, axis=0).astype(np.float32))
                if len(reps) >= TRAIN_MIN_OBJECTS:
                    reps = np.vstack(reps)
                    cand = train_bins(reps)
                    pairwise = [dist_hsv(cand[i], cand[j])
                                for i in range(3) for j in range(i+1, 3)]
                    min_sep = min(pairwise) if pairwise else 0.0
                    if min_sep < 8.0:
                        print(f"[WARN] Centres too close (sep={min_sep:.1f}). "
                              "Ensure 3 clearly different colours. Restarting timer…")
                        training_start = time.time()
                        per_block_samples.clear()
                    else:
                        centers = cand; trained = True
                        print("\n[INFO] MODEL TRAINED — 3 colour bins learned.")
                        print("[INFO] Add new blocks to trigger robot moves.\n")
                else:
                    print(f"[WARN] Only {len(reps)} stable block(s) found "
                          f"(need {TRAIN_MIN_OBJECTS}). Keep blocks visible.")
                    training_start = time.time()

        # ── LEGACY: update blue-peg targets for M-key ────────────────────────
        best_blue = detect_blue_pegs(frame)
        if best_blue is not None:
            bu, bv, br = best_blue
            cv2.circle(vis, (int(bu), int(bv)), int(br), (0, 0, 255), 2)
            cv2.putText(vis, "BLUE PEG", (int(bu)+int(br)+4, int(bv)-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1)
            Xmm, Ymm = apply_homography(H, (bu, bv))
            app_abs, pick_abs = build_pick_app_poses(
                base_T_board, pick_ref_pose, Xmm, Ymm)
            t_blue_app.setPoseAbs(app_abs)
            t_blue_pick.setPoseAbs(pick_abs)
            last_blue_poses = (app_abs, pick_abs)

        # ── VISUALISATION ─────────────────────────────────────────────────────
        if not trained:
            rem = max(0.0, TRAIN_SECONDS - (now - training_start))
            cv2.putText(vis, f"TRAINING — show 3 colours | {rem:.1f}s left",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        else:
            cv2.putText(vis, "TRAINED | new block → classify → move robot",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,100), 2)
            if centers is not None:
                for i in range(3):
                    bgr = hsv_to_bgr(centers[i])
                    y0  = 50 + i * 30
                    cv2.rectangle(vis, (10, y0), (40, y0+20), bgr, -1)
                    cv2.putText(vis, f"BIN_{i+1}", (48, y0+15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)

        for (cx, cy, w, h, cnt) in detections:
            x = int(cx - w/2); y = int(cy - h/2)
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 200, 0), 2)

        for bid, blk in blocks.items():
            x1 = int(blk.cx - blk.w/2)
            y1 = int(blk.cy - blk.h/2)
            label = f"id{bid} {blk.bin_name}"
            cv2.putText(vis, label, (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        cv2.imshow("RoboDK HITL — Bin Sorter", vis)

        # ── KEY HANDLING ──────────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        if key in (ord("m"), ord("M")):
            if last_blue_poses is None:
                print("[INFO] No blue peg detected (or homography invalid).")
            else:
                print("[RoboDK] Moving to best blue peg…")
                try:
                    robot.MoveJ(t_blue_app)
                    robot.MoveL(t_blue_pick)
                    robot.MoveL(t_blue_app)
                    print("[RoboDK] Done.")
                except Exception as e:
                    print(f"[RoboDK] Move error: {e}")

        if key in (ord("t"), ord("T")):
            trained = False; centers = None
            training_start = time.time()
            reset_training(blocks, per_block_samples)
            print("\n[INFO] Manual retrain triggered.")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
