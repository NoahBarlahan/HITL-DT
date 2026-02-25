import cv2
import numpy as np
import time
import json
import socket
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# ============================================================
# CONFIG
# ============================================================

# ---- Robot / Emulator TCP ----
ROBOT_IP = "127.0.0.1"
ROBOT_PORT = 1234
CMD_TERMINATOR = "\n"    # try "\r\n" if needed
TCP_TIMEOUT_S = 2.0

BIN_TO_CMD = {
    "BIN_1": "moveP1",
    "BIN_2": "moveP2",
    "BIN_3": "moveP3",
}

# ---- Block geometry (tune these) ----
MIN_AREA_PX = 150
MAX_AREA_PX = 150000
ASPECT_MIN = 0.70
ASPECT_MAX = 1.35
SOLIDITY_MIN = 0.9        # how filled the contour is (filters noisy edges)
APPROX_EPS_FRAC = 0.03     # polygon approx tightness

# ---- Tracking ----
MATCH_DIST_PX = 90
MISSING_FRAMES_TO_REMOVE = 15

# ---- Training ----
N_BINS = 3
TRAIN_SECONDS = 3.0
TRAIN_MIN_OBJECTS = 3
TRAIN_MIN_SAMPLES = 30     # per block across frames

# ---- Color feature sampling ----
# We compute mean color on the *interior* only to avoid edge contamination.
INTERIOR_ERODE_PX = 8       # increase if you still see background mixing
SAT_MIN = 15                # reject near-gray pixels
VAL_MIN = 15

# ---- Unknown detection ----
UNKNOWN_DIST_THRESH = 18.0  # lower => stricter "unknown"
RULES_JSON = "unknown_color_rules.json"

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class TrackedBlock:
    block_id: int
    cx: int
    cy: int
    w: int
    h: int
    last_seen_frame: int
    mean_hsv: np.ndarray
    last_update_t: float
    bin_name: str
    acted: bool

# ============================================================
# ROBOT SEND
# ============================================================

def send_to_emulator(cmd: str) -> bool:
    payload = (cmd + CMD_TERMINATOR).encode("utf-8", errors="ignore")
    try:
        with socket.create_connection((ROBOT_IP, ROBOT_PORT), timeout=TCP_TIMEOUT_S) as s:
            s.sendall(payload)
            try:
                s.shutdown(socket.SHUT_WR)
            except OSError:
                pass
        print(f"[ROBOT] SENT: {cmd!r}")
        return True
    except Exception as e:
        print(f"[ROBOT] SEND FAILED: {e}")
        return False

def do_robot_action_for_bin(bin_name: str):
    cmd = BIN_TO_CMD.get(bin_name)
    if not cmd:
        print(f"[ROBOT] No command configured for {bin_name}")
        return
    print(f"[ROBOT] ACTION: {bin_name} -> {cmd}")
    send_to_emulator(cmd)

# ============================================================
# RULES FOR UNKNOWN COLORS
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
    # coarse signature so unknowns repeat consistently
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
    print("  s) Skip (no robot move)")
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
# EDGE-BASED BLOCK DETECTION (KEY FIX)
# ============================================================

def detect_blocks_edges(frame_bgr) -> List[Tuple[int, int, int, int, np.ndarray]]:
    """
    Edge-based detection: Canny -> contours -> square-ish filter.
    Returns (cx, cy, w, h, contour).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny (auto-ish thresholds using median)
    v = np.median(gray)
    lo = int(max(0, 0.66 * v))
    hi = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, lo, hi)

    # Close gaps so the block outline becomes one contour
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
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

        # Solidity: area / convex hull area (filters scribbly edges)
        hull = cv2.convexHull(cnt)
        hull_area = max(1.0, cv2.contourArea(hull))
        sol = area / hull_area
        if sol < SOLIDITY_MIN:
            continue

        # Optional: polygon approx should look like a quad-ish shape
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, APPROX_EPS_FRAC * peri, True)
        # allow 4..8 vertices for perspective/noise
        if not (4 <= len(approx) <= 8):
            continue

        cx = x + w // 2
        cy = y + h // 2
        detections.append((cx, cy, w, h, cnt))

    return detections

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
        return None, None, None

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    shifted = contour.copy()
    shifted[:, 0, 0] -= x1
    shifted[:, 0, 1] -= y1

    mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [shifted], -1, 255, thickness=-1)

    return hsv_roi, mask, (x1, y1, x2 - x1, y2 - y1)

# ============================================================
# COLOR FEATURE (INTERIOR-ONLY) (KEY FIX)
# ============================================================

def mean_hsv_interior(hsv_roi: np.ndarray, mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Compute mean HSV inside the *interior* of the block to avoid white background + edges.
    """
    if hsv_roi is None or mask is None:
        return None

    # Erode mask so we only sample block interior (avoid edge halo / background)
    if INTERIOR_ERODE_PX > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * INTERIOR_ERODE_PX + 1, 2 * INTERIOR_ERODE_PX + 1))
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

def hsv_patch_to_bgr(hsv_center: np.ndarray):
    hsv_px = np.uint8([[[int(hsv_center[0]), int(hsv_center[1]), int(hsv_center[2])]]])
    return cv2.cvtColor(hsv_px, cv2.COLOR_HSV2BGR)[0, 0].tolist()

def dist_hsv(a: np.ndarray, b: np.ndarray) -> float:
    # simple Euclidean in HSV; sufficient for solid-color blocks
    d = a.astype(np.float32) - b.astype(np.float32)
    return float(np.linalg.norm(d))

# ============================================================
# TRAIN + ASSIGN
# ============================================================

def train_3_bins_from_samples(samples: np.ndarray) -> np.ndarray:
    """
    samples: (N,3) mean HSV features
    returns centers: (3,3) sorted by Hue
    """
    samples = samples.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.2)
    _, labels, centers = cv2.kmeans(samples, N_BINS, None, criteria, 8, cv2.KMEANS_PP_CENTERS)
    order = np.argsort(centers[:, 0])
    return centers[order].astype(np.float32)

def assign_bin(mean_hsv: np.ndarray, centers: np.ndarray) -> Tuple[str, float]:
    dists = [dist_hsv(mean_hsv, centers[i]) for i in range(centers.shape[0])]
    idx = int(np.argmin(dists))
    return f"BIN_{idx+1}", float(dists[idx])

# ============================================================
# MAIN
# ============================================================

def main():
    rules = load_rules()

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    blocks: Dict[int, TrackedBlock] = {}
    per_block_samples: Dict[int, List[np.ndarray]] = {}

    next_id = 1
    frame_idx = 0

    trained = False
    centers = None
    training_start = time.time()

    print("[INFO] TRAINING PHASE: place 3 majority-color blocks clearly separated.")
    print(f"[INFO] Train time: {TRAIN_SECONDS:.1f}s. Then it will say MODEL TRAINED.")
    print("[INFO] INFERENCE PHASE: each NEW block triggers one robot move based on bin.")
    print(f"[INFO] Emulator: {ROBOT_IP}:{ROBOT_PORT} terminator={CMD_TERMINATOR!r}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        now = time.time()

        detections = detect_blocks_edges(frame)  # <-- EDGE BASED

        # ------- associate detections to tracked IDs -------
        det_used = [False] * len(detections)
        ids = list(blocks.keys())
        pos = {bid: (blocks[bid].cx, blocks[bid].cy) for bid in ids}

        matched: Dict[int, int] = {}
        for di, (cx, cy, w, h, cnt) in enumerate(detections):
            best_id = None
            best_d = 1e9
            for bid in ids:
                d = ((cx - pos[bid][0])**2 + (cy - pos[bid][1])**2) ** 0.5
                if d < best_d:
                    best_d = d
                    best_id = bid
            if best_id is not None and best_d <= MATCH_DIST_PX and best_id not in matched:
                matched[best_id] = di
                det_used[di] = True

        # ------- update matched -------
        for bid, di in matched.items():
            cx, cy, w, h, cnt = detections[di]
            hsv_roi, mask, _ = roi_hsv_and_mask_from_contour(frame, cnt)
            mean = mean_hsv_interior(hsv_roi, mask)

            blk = blocks[bid]
            blk.cx, blk.cy, blk.w, blk.h = cx, cy, w, h
            blk.last_seen_frame = frame_idx
            blk.last_update_t = now
            if mean is not None:
                blk.mean_hsv = mean

            if not trained and mean is not None:
                per_block_samples.setdefault(bid, []).append(mean)

        # ------- create new blocks -------
        for di, used in enumerate(det_used):
            if used:
                continue
            cx, cy, w, h, cnt = detections[di]
            hsv_roi, mask, _ = roi_hsv_and_mask_from_contour(frame, cnt)
            mean = mean_hsv_interior(hsv_roi, mask)
            if mean is None:
                continue

            bid = next_id
            next_id += 1
            blocks[bid] = TrackedBlock(
                block_id=bid,
                cx=cx, cy=cy, w=w, h=h,
                last_seen_frame=frame_idx,
                mean_hsv=mean,
                last_update_t=now,
                bin_name="UNASSIGNED",
                acted=False
            )

            if not trained:
                per_block_samples.setdefault(bid, []).append(mean)

            # If trained: NEW block triggers decision + one robot action
            if trained and centers is not None:
                bin_guess, dmin = assign_bin(mean, centers)
                sig = make_sig(mean)
                forced = rules.get(sig)

                if dmin > UNKNOWN_DIST_THRESH and forced is None:
                    action = ask_user_unknown_action(sig=sig, suggested=bin_guess)
                    if action == "RETRAIN":
                        trained = False
                        centers = None
                        training_start = time.time()
                        per_block_samples = {}
                        for b in blocks.values():
                            b.acted = False
                            b.bin_name = "UNASSIGNED"
                        print("\n[INFO] Retraining started. Place your 3 majority-color blocks again.")
                    elif action == "SKIP":
                        blocks[bid].bin_name = "UNKNOWN_SKIP"
                        blocks[bid].acted = True
                    else:
                        rules[sig] = action
                        save_rules(rules)
                        blocks[bid].bin_name = action
                        do_robot_action_for_bin(action)
                        blocks[bid].acted = True
                else:
                    final_bin = forced if forced is not None else bin_guess
                    blocks[bid].bin_name = final_bin
                    do_robot_action_for_bin(final_bin)
                    blocks[bid].acted = True

        # ------- cleanup stale tracked blocks -------
        remove_ids = []
        for bid, blk in blocks.items():
            if frame_idx - blk.last_seen_frame > MISSING_FRAMES_TO_REMOVE:
                remove_ids.append(bid)
        for bid in remove_ids:
            del blocks[bid]
            per_block_samples.pop(bid, None)

        # ------- training finish -------
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
                    centers = train_3_bins_from_samples(reps)

                    # Sanity: if centers are too close, you didn’t actually see 3 different colors
                    pairwise = []
                    for i in range(3):
                        for j in range(i+1, 3):
                            pairwise.append(dist_hsv(centers[i], centers[j]))
                    min_sep = min(pairwise) if pairwise else 0.0

                    if min_sep < 8.0:
                        # likely segmentation still mixing or only one color present
                        print("\n[WARN] Learned centers are very close (min separation ~"
                              f"{min_sep:.1f}). Ensure 3 different colors are visible and separated.")
                        training_start = time.time()
                        per_block_samples = {}
                    else:
                        trained = True
                        print("\n[INFO] MODEL TRAINED: learned 3 distinct color groups (BIN_1..BIN_3).")
                        print("[INFO] Now add NEW blocks one at a time. Each new block triggers one robot move.")

                else:
                    print("\n[WARN] Not enough stable blocks to train. Keep 3 blocks visible and separated.")
                    training_start = time.time()

        # ------- visualization -------
        vis = frame.copy()

        if not trained:
            remaining = max(0.0, TRAIN_SECONDS - (now - training_start))
            cv2.putText(vis, "TRAINING (edge-based): show 3 separated blocks",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(vis, f"train in {remaining:.1f}s",
                        (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(vis, "TRAINED: NEW block -> classify -> move",
                        (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        if trained and centers is not None:
            y0 = 70
            for i in range(3):
                bgr = hsv_patch_to_bgr(centers[i])
                cv2.rectangle(vis, (20, y0 + i*28), (60, y0 + i*28 + 18), bgr, -1)
                cv2.putText(vis, f"BIN_{i+1}", (70, y0 + i*28 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        # draw detected boxes
        for (cx, cy, w, h, cnt) in detections:
            x = int(cx - w/2)
            y = int(cy - h/2)
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 200, 0), 2)

        # draw tracked labels
        for bid, blk in blocks.items():
            x1 = int(blk.cx - blk.w/2)
            y1 = int(blk.cy - blk.h/2)
            cv2.putText(vis, f"id{bid} {blk.bin_name}", (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

        cv2.imshow("Edge-based blocks -> train 3 bins -> act on new blocks", vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()