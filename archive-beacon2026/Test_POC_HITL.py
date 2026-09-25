"""
hitl_edge_combined.py
=====================
Combines edge-based block detection (auto-training k-means) with the full
HITL confidence pipeline (human prompt + CSV logging + robot TCP commands).

WORKFLOW
--------
  PHASE 1 — TRAINING (first ~3 s):
    Place all 3 color blocks clearly visible and separated.
    The system detects them via Canny edges, samples interior HSV,
    and learns 3 color cluster centers with k-means. No operator
    input needed.

  PHASE 2 — INFERENCE:
    Every NEW block that enters frame is classified against the
    learned centers.  A confidence score is derived from the
    distance to the nearest cluster.

    Confidence >= CONFIDENCE_THRESH (60%):
      → Auto-assigned, robot command sent, row logged to CSV.

    Confidence <  CONFIDENCE_THRESH  OR  block is too far from
    all centers (unknown color):
      → HITL overlay appears on screen.
        Press 1 / 2 / 3 to assign to a bin.
        Press S to skip.
        Press R to restart training.

KEYS (during inference)
-----------------------
  1 / 2 / 3  — assign block to bin during HITL prompt
  S          — skip (no assignment)
  R          — retrain (restart training phase)
  C          — print live HSV of centre pixel to console
  Q / ESC    — quit and show session summary

CSV COLUMNS
-----------
  timestamp, trial, color_detected, top_prob, second_color,
  second_prob, hitl_triggered, operator_override, final_bin,
  response_time_ms, hsv_H, hsv_S, hsv_V
"""

import cv2
import numpy as np
import csv
import os
import time
import json
import socket
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

# ============================================================
# CONFIG — ROBOT / EMULATOR TCP
# ============================================================

ROBOT_IP       = "127.0.0.1"
ROBOT_PORT     = 1234
CMD_TERMINATOR = "\n"       # try "\r\n" if needed
TCP_TIMEOUT_S  = 2.0

BIN_TO_CMD = {
    "BIN_1": "moveP1",
    "BIN_2": "moveP2",
    "BIN_3": "moveP3",
}

# ============================================================
# CONFIG — BLOCK DETECTION (edge-based)
# ============================================================

CAMERA_INDEX = 1            # change to 0 if needed
FRAME_W      = 640
FRAME_H      = 480

MIN_AREA_PX      = 150
MAX_AREA_PX      = 150_000
ASPECT_MIN       = 0.70
ASPECT_MAX       = 1.35
SOLIDITY_MIN     = 0.90
APPROX_EPS_FRAC  = 0.03
INTERIOR_ERODE_PX = 8
SAT_MIN_DETECT   = 15       # for interior color sampling
VAL_MIN_DETECT   = 15

MATCH_DIST_PX           = 90
MISSING_FRAMES_TO_REMOVE = 15

# ============================================================
# CONFIG — TRAINING (k-means)
# ============================================================

N_BINS           = 3
TRAIN_SECONDS    = 3.0
TRAIN_MIN_OBJECTS = 3
TRAIN_MIN_SAMPLES = 30      # samples per block before training
MIN_CENTER_SEP   = 8.0      # if learned centers are closer, retrain

# ============================================================
# CONFIG — HITL PIPELINE
# ============================================================

CONFIDENCE_THRESH = 0.60    # below this → HITL triggered
UNKNOWN_DIST_THRESH = 18.0  # distance to nearest center → force HITL
STABLE_FRAMES     = 6       # consecutive frames block must be visible
                             # before acting (lower than fixed-box version
                             # because edge detection is already stable)
COOLDOWN_S        = 3.0     # seconds before same block can re-trigger

OPERATOR_ID = "pilot_001"
LOG_DIR     = "hitl_logs"
os.makedirs(LOG_DIR, exist_ok=True)

RULES_JSON = "unknown_color_rules.json"

# ============================================================
# CONFIG — DISPLAY
# ============================================================

BIN_NAMES = {"1": "BIN_1", "2": "BIN_2", "3": "BIN_3"}
BIN_COLORS_BGR = {
    "BIN_1": (220,  80,   0),
    "BIN_2": (  0, 200,   0),
    "BIN_3": (  0,   0, 220),
}
BAR_MAX_WIDTH    = 200
MIN_DISPLAY_PROB = 0.02

# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class TrackedBlock:
    block_id:       int
    cx:             int
    cy:             int
    w:              int
    h:              int
    contour:        np.ndarray
    last_seen_frame: int
    mean_hsv:       np.ndarray
    last_update_t:  float
    bin_name:       str
    acted:          bool
    stable_count:   int      # frames seen consecutively since last act
    last_acted_t:   float


# ============================================================
# ROBOT
# ============================================================

def send_to_robot(cmd: str) -> bool:
    payload = (cmd + CMD_TERMINATOR).encode("utf-8", errors="ignore")
    try:
        with socket.create_connection((ROBOT_IP, ROBOT_PORT),
                                      timeout=TCP_TIMEOUT_S) as s:
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


def robot_action(bin_name: str):
    cmd = BIN_TO_CMD.get(bin_name)
    if cmd:
        send_to_robot(cmd)
    else:
        print(f"[ROBOT] No command for {bin_name}")


# ============================================================
# UNKNOWN COLOR RULES  (persist decisions across sessions)
# ============================================================

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
    """Coarse HSV signature so unknown decisions repeat consistently."""
    h = int(mean_hsv[0] // 8)
    s = int(mean_hsv[1] // 16)
    v = int(mean_hsv[2] // 16)
    return f"h{h}_s{s}_v{v}"


# ============================================================
# EDGE-BASED BLOCK DETECTION
# ============================================================

def detect_blocks_edges(frame_bgr) -> List[Tuple[int, int, int, int, np.ndarray]]:
    """
    Canny → contours → square-ish + solidity filter.
    Returns list of (cx, cy, w, h, contour).
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    v  = np.median(gray)
    lo = int(max(0,   0.66 * v))
    hi = int(min(255, 1.33 * v))
    edges = cv2.Canny(gray, lo, hi)

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, k, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=2)

    contours, _ = cv2.findContours(edges,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    detections = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if not (MIN_AREA_PX <= area <= MAX_AREA_PX):
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 0 or h <= 0:
            continue
        aspect = w / float(h)
        if not (ASPECT_MIN <= aspect <= ASPECT_MAX):
            continue

        hull      = cv2.convexHull(cnt)
        hull_area = max(1.0, cv2.contourArea(hull))
        if (area / hull_area) < SOLIDITY_MIN:
            continue

        peri   = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, APPROX_EPS_FRAC * peri, True)
        if not (4 <= len(approx) <= 8):
            continue

        detections.append((x + w // 2, y + h // 2, w, h, cnt))

    return detections


# ============================================================
# COLOR FEATURE EXTRACTION  (interior-only to avoid edge halo)
# ============================================================

def extract_interior_hsv(frame_bgr,
                          contour: np.ndarray) -> Optional[np.ndarray]:
    H, W = frame_bgr.shape[:2]
    x, y, bw, bh = cv2.boundingRect(contour)
    pad = 2
    x1, y1 = max(x - pad, 0), max(y - pad, 0)
    x2, y2 = min(x + bw + pad, W - 1), min(y + bh + pad, H - 1)

    roi = frame_bgr[y1:y2, x1:x2]
    if roi.size == 0:
        return None

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    shifted        = contour.copy()
    shifted[:, 0, 0] -= x1
    shifted[:, 0, 1] -= y1

    mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [shifted], -1, 255, thickness=-1)

    # Erode to avoid edge / background contamination
    if INTERIOR_ERODE_PX > 0:
        k = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * INTERIOR_ERODE_PX + 1, 2 * INTERIOR_ERODE_PX + 1))
        mask = cv2.erode(mask, k, iterations=1)

    pix = hsv_roi[mask > 0]
    if pix.size == 0:
        return None

    keep = (pix[:, 1] >= SAT_MIN_DETECT) & (pix[:, 2] >= VAL_MIN_DETECT)
    pix  = pix[keep]
    if pix.shape[0] < 50:
        return None

    return pix.mean(axis=0).astype(np.float32)


# ============================================================
# K-MEANS TRAINING
# ============================================================

def train_centers(reps: np.ndarray) -> Optional[np.ndarray]:
    """
    reps: (N, 3) median HSV feature per tracked block.
    Returns (3, 3) cluster centers sorted by Hue, or None if too close.
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.2)
    _, _, centers = cv2.kmeans(reps.astype(np.float32), N_BINS, None,
                               criteria, 8, cv2.KMEANS_PP_CENTERS)
    order   = np.argsort(centers[:, 0])
    centers = centers[order].astype(np.float32)

    pairwise = [np.linalg.norm(centers[i] - centers[j])
                for i in range(3) for j in range(i + 1, 3)]
    if min(pairwise) < MIN_CENTER_SEP:
        return None
    return centers


# ============================================================
# CONFIDENCE SCORING  (distance → probability)
# ============================================================

def dist_hsv(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a.astype(np.float32) -
                                b.astype(np.float32)))


def compute_confidence(mean_hsv: np.ndarray,
                       centers: np.ndarray) -> Tuple[str, float, str, float, float]:
    """
    Returns (top_bin, top_prob, second_bin, second_prob, min_dist).

    Converts raw HSV distances to a softmax-style probability so the
    same 0–100% confidence scale used in hitl_pipeline still applies.
    """
    dists = np.array([dist_hsv(mean_hsv, centers[i])
                      for i in range(centers.shape[0])],
                     dtype=np.float32)
    # Convert distances to scores: closer = higher
    scores = 1.0 / (dists + 1e-6)
    total  = scores.sum()
    probs  = scores / total

    order      = np.argsort(-probs)
    top_idx    = int(order[0])
    second_idx = int(order[1]) if len(order) > 1 else top_idx

    top_bin    = f"BIN_{top_idx + 1}"
    second_bin = f"BIN_{second_idx + 1}"
    top_prob   = float(probs[top_idx])
    second_prob = float(probs[second_idx])
    min_dist   = float(dists[top_idx])

    return top_bin, top_prob, second_bin, second_prob, min_dist


def hsv_patch_to_bgr(hsv_center: np.ndarray) -> list:
    px = np.uint8([[[int(hsv_center[0]),
                     int(hsv_center[1]),
                     int(hsv_center[2])]]])
    return cv2.cvtColor(px, cv2.COLOR_HSV2BGR)[0, 0].tolist()


# ============================================================
# CSV LOGGING
# ============================================================

def init_log() -> str:
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(LOG_DIR, f"session_{OPERATOR_ID}_{ts}.csv")
    with open(path, "w", newline="") as f:
        csv.writer(f).writerow([
            "timestamp", "trial", "color_detected", "top_prob",
            "second_color", "second_prob",
            "hitl_triggered", "operator_override",
            "final_bin", "response_time_ms",
            "hsv_H", "hsv_S", "hsv_V",
        ])
    print(f"[LOG] {path}")
    return path


def log_trial(path: str, row: dict):
    with open(path, "a", newline="") as f:
        csv.writer(f).writerow([
            row["timestamp"],        row["trial"],
            row["color_detected"],   f"{row['top_prob']:.3f}",
            row["second_color"],     f"{row['second_prob']:.3f}",
            row["hitl_triggered"],   row["operator_override"],
            row["final_bin"],        f"{row['response_time_ms']:.1f}",
            row["hsv_H"],            row["hsv_S"],            row["hsv_V"],
        ])


# ============================================================
# DRAWING HELPERS
# ============================================================

def draw_prob_bars(vis, probs: Dict[str, float], origin: Tuple[int, int]):
    x0, y0  = origin
    bar_h   = 20
    gap     = 4
    COLORS  = [
        (220,  80,   0),
        (  0, 200,   0),
        (  0,   0, 220),
    ]
    for i, (name, prob) in enumerate(probs.items()):
        if prob < MIN_DISPLAY_PROB:
            continue
        bgr   = COLORS[i % len(COLORS)]
        bar_w = int(prob * BAR_MAX_WIDTH)
        y     = y0 + i * (bar_h + gap)
        cv2.rectangle(vis, (x0, y), (x0 + BAR_MAX_WIDTH, y + bar_h),
                      (40, 40, 40), -1)
        if bar_w > 0:
            cv2.rectangle(vis, (x0, y), (x0 + bar_w, y + bar_h), bgr, -1)
        cv2.putText(vis, f"{name}: {prob*100:.1f}%",
                    (x0 + BAR_MAX_WIDTH + 8, y + bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    bgr if prob > 0.1 else (150, 150, 150), 1)


def draw_hitl_overlay(vis, top_bin, top_prob, probs, mean_hsv):
    overlay = vis.copy()
    cv2.rectangle(overlay, (60, 80), (580, 430), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.82, vis, 0.18, 0, vis)

    cv2.putText(vis, "HUMAN INPUT NEEDED",
                (80, 125), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 2)
    cv2.putText(vis, f"System guess:  {top_bin}  ({top_prob*100:.0f}%)",
                (80, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    items  = list(probs.items())
    second = items[1] if len(items) > 1 else ("--", 0.0)
    H, S, V = int(mean_hsv[0]), int(mean_hsv[1]), int(mean_hsv[2])

    if top_prob < 0.40:
        conf_text = "Very uncertain — color does not match any cluster well"
    elif top_prob < CONFIDENCE_THRESH:
        conf_text = f"Low confidence — could be {top_bin} or {second[0]}"
    else:
        conf_text = f"Unknown color signature — operator decision required"

    for i, line in enumerate([
        f"Confidence: {top_prob*100:.0f}%  (threshold: {int(CONFIDENCE_THRESH*100)}%)",
        conf_text,
        f"Raw values: H={H}  S={S}  V={V}",
        f"Runner-up:  {second[0]} ({second[1]*100:.0f}%)",
    ]):
        cv2.putText(vis, line, (80, 200 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)

    btn_y = 330
    for key, bin_name in BIN_NAMES.items():
        bgr = BIN_COLORS_BGR[bin_name]
        bx  = 80 + (int(key) - 1) * 160
        cv2.rectangle(vis, (bx, btn_y), (bx + 130, btn_y + 50), bgr, -1)
        cv2.putText(vis, f"[{key}] {bin_name}",
                    (bx + 8, btn_y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.putText(vis, "[S] Skip    [R] Retrain",
                (80, btn_y + 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)


def draw_center_swatches(vis, centers: np.ndarray):
    y0 = 70
    for i in range(3):
        bgr = hsv_patch_to_bgr(centers[i])
        cv2.rectangle(vis, (20, y0 + i * 28),
                          (60, y0 + i * 28 + 18), bgr, -1)
        cv2.putText(vis, f"BIN_{i+1}",
                    (68, y0 + i * 28 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)


def draw_session_stats(vis, trial, auto_count, hitl_count, override_count):
    for i, s in enumerate([
        f"Trials:    {trial}",
        f"Auto:      {auto_count}",
        f"HITL:      {hitl_count}",
        f"Overrides: {override_count}",
    ]):
        cv2.putText(vis, s, (FRAME_W - 180, 20 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)


# ============================================================
# SESSION SUMMARY
# ============================================================

def print_summary(trial, auto_count, hitl_count, override_count,
                  rt_list, bin_counts):
    print("\n" + "=" * 50)
    print("  SESSION SUMMARY")
    print("=" * 50)
    print(f"  Total trials:       {trial}")
    print(f"  Auto decisions:     {auto_count}")
    print(f"  HITL triggered:     {hitl_count}  "
          f"({100*hitl_count/max(trial,1):.0f}%)")
    print(f"  Operator overrides: {override_count}  "
          f"({100*override_count/max(hitl_count,1):.0f}% of HITL)")
    if rt_list:
        print(f"  Avg HITL resp time: {np.mean(rt_list):.0f} ms")
    print("  Bin distribution:")
    for bin_name, count in sorted(bin_counts.items()):
        print(f"    {bin_name}: {count}")
    print("=" * 50)


# ============================================================
# MAIN
# ============================================================

def main():
    rules    = load_rules()
    log_file = init_log()

    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,    FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,   FRAME_H)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,     1)
    cap.set(cv2.CAP_PROP_AUTO_WB,        0)
    cap.set(cv2.CAP_PROP_WB_TEMPERATURE, 5500)
    for _ in range(30):
        cap.read()

    # ── Tracking state ────────────────────────────────────────
    blocks:            Dict[int, TrackedBlock] = {}
    per_block_samples: Dict[int, List[np.ndarray]] = {}
    next_id    = 1
    frame_idx  = 0

    # ── Training state ────────────────────────────────────────
    trained         = False
    centers: Optional[np.ndarray] = None
    training_start  = time.time()

    # ── Session stats ─────────────────────────────────────────
    trial          = 0
    auto_count     = 0
    hitl_count     = 0
    override_count = 0
    rt_list:       List[float] = []
    bin_counts     = {b: 0 for b in BIN_NAMES.values()}

    # ── HITL state ────────────────────────────────────────────
    hitl_active     = False
    hitl_block_id   = -1
    hitl_start_t    = 0.0
    hitl_top_bin    = ""
    hitl_top_prob   = 0.0
    hitl_probs:     Dict[str, float] = {}
    hitl_mean_hsv:  Optional[np.ndarray] = None
    hitl_min_dist   = 0.0

    print("\n" + "=" * 55)
    print("  HITL Edge-Based Color Sorting Pipeline")
    print("=" * 55)
    print(f"  Operator:       {OPERATOR_ID}")
    print(f"  HITL threshold: {int(CONFIDENCE_THRESH*100)}% confidence")
    print(f"  Robot emulator: {ROBOT_IP}:{ROBOT_PORT}")
    print()
    print("  PHASE 1 — TRAINING")
    print(f"  Place 3 color blocks in frame.  ({TRAIN_SECONDS:.0f}s auto-trains.)")
    print()
    print("  PHASE 2 — INFERENCE")
    print("  Each NEW block → classify → auto or HITL prompt.")
    print("  HITL keys: 1/2/3 = bin  S = skip  R = retrain")
    print("  C = HSV readout  |  Q / ESC = quit")
    print("=" * 55 + "\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            continue
        frame_idx += 1
        now = time.time()

        detections = detect_blocks_edges(frame)

        # ── Associate detections → tracked IDs ────────────────
        det_used = [False] * len(detections)

        for bid in list(blocks.keys()):
            blk      = blocks[bid]
            best_di  = -1
            best_d   = 1e9
            for di, (cx, cy, w, h, cnt) in enumerate(detections):
                d = ((cx - blk.cx) ** 2 + (cy - blk.cy) ** 2) ** 0.5
                if d < best_d:
                    best_d  = d
                    best_di = di

            if best_di >= 0 and best_d <= MATCH_DIST_PX \
                    and not det_used[best_di]:
                det_used[best_di] = True
                cx, cy, w, h, cnt = detections[best_di]
                mean = extract_interior_hsv(frame, cnt)

                blk.cx, blk.cy       = cx, cy
                blk.w,  blk.h        = w,  h
                blk.contour          = cnt
                blk.last_seen_frame  = frame_idx
                blk.last_update_t    = now
                if mean is not None:
                    blk.mean_hsv = mean

                if not trained and mean is not None:
                    per_block_samples.setdefault(bid, []).append(mean)

                # Increment stability counter for un-acted blocks
                if trained and not blk.acted:
                    blk.stable_count += 1

        # ── Create new tracked blocks ─────────────────────────
        for di, used in enumerate(det_used):
            if used:
                continue
            cx, cy, w, h, cnt = detections[di]
            mean = extract_interior_hsv(frame, cnt)
            if mean is None:
                continue

            bid = next_id
            next_id += 1
            blocks[bid] = TrackedBlock(
                block_id=bid, cx=cx, cy=cy, w=w, h=h,
                contour=cnt, last_seen_frame=frame_idx,
                mean_hsv=mean, last_update_t=now,
                bin_name="UNASSIGNED", acted=False,
                stable_count=1, last_acted_t=0.0,
            )
            if not trained:
                per_block_samples.setdefault(bid, []).append(mean)

        # ── Training finish check ──────────────────────────────
        if not trained:
            elapsed = now - training_start
            if elapsed >= TRAIN_SECONDS:
                reps = []
                for bid, samples in per_block_samples.items():
                    if len(samples) >= TRAIN_MIN_SAMPLES:
                        arr = np.vstack(samples)
                        reps.append(np.median(arr, axis=0))

                if len(reps) >= TRAIN_MIN_OBJECTS:
                    result = train_centers(np.vstack(reps))
                    if result is not None:
                        centers = result
                        trained = True
                        print("\n[INFO] MODEL TRAINED — 3 color clusters learned "
                              "(BIN_1 / BIN_2 / BIN_3).")
                        print("[INFO] Each NEW block will now trigger auto or HITL.\n")
                    else:
                        print("\n[WARN] Cluster centers too similar — ensure 3 "
                              "distinct colors are visible. Restarting timer…")
                        training_start  = time.time()
                        per_block_samples = {}
                else:
                    print("\n[WARN] Not enough stable blocks. Keep 3 blocks in "
                          "frame. Restarting timer…")
                    training_start  = time.time()

        # ── INFERENCE: decide on stable un-acted blocks ────────
        if trained and centers is not None and not hitl_active:
            for bid, blk in list(blocks.items()):
                if blk.acted:
                    continue
                if blk.stable_count < STABLE_FRAMES:
                    continue
                if (now - blk.last_acted_t) < COOLDOWN_S:
                    continue

                mean = blk.mean_hsv
                top_bin, top_prob, second_bin, second_prob, min_dist = \
                    compute_confidence(mean, centers)

                sig    = make_sig(mean)
                forced = rules.get(sig)

                # Build probs dict for display
                probs = {f"BIN_{i+1}": float(1.0 / (
                    dist_hsv(mean, centers[i]) + 1e-6))
                    for i in range(3)}
                total_ = sum(probs.values())
                probs  = {k: v / total_ for k, v in probs.items()}

                is_unknown = (min_dist > UNKNOWN_DIST_THRESH) and (forced is None)
                is_low_conf = top_prob < CONFIDENCE_THRESH

                if is_unknown or is_low_conf:
                    # HITL triggered
                    trial      += 1
                    hitl_count += 1
                    hitl_active    = True
                    hitl_block_id  = bid
                    hitl_start_t   = now
                    hitl_top_bin   = top_bin
                    hitl_top_prob  = top_prob
                    hitl_probs     = probs
                    hitl_mean_hsv  = mean.copy()
                    hitl_min_dist  = min_dist
                    print(f"\n[HITL]  Trial {trial:03d}  {top_bin}  "
                          f"conf={top_prob*100:.0f}%  dist={min_dist:.1f}  "
                          f"→ awaiting operator…")
                    break   # one HITL at a time

                else:
                    # Auto classify
                    final_bin = forced if forced is not None else top_bin
                    trial      += 1
                    auto_count += 1
                    bin_counts[final_bin] = bin_counts.get(final_bin, 0) + 1

                    blk.bin_name    = final_bin
                    blk.acted       = True
                    blk.last_acted_t = now
                    robot_action(final_bin)

                    log_trial(log_file, {
                        "timestamp":         datetime.now().isoformat(),
                        "trial":             trial,
                        "color_detected":    top_bin,
                        "top_prob":          top_prob,
                        "second_color":      second_bin,
                        "second_prob":       second_prob,
                        "hitl_triggered":    False,
                        "operator_override": False,
                        "final_bin":         final_bin,
                        "response_time_ms":  0.0,
                        "hsv_H": int(mean[0]),
                        "hsv_S": int(mean[1]),
                        "hsv_V": int(mean[2]),
                    })
                    print(f"[AUTO]  Trial {trial:03d}  {top_bin}  "
                          f"conf={top_prob*100:.0f}%  → {final_bin}")

        # ── Remove stale blocks ────────────────────────────────
        for bid in [b for b, blk in blocks.items()
                    if frame_idx - blk.last_seen_frame > MISSING_FRAMES_TO_REMOVE]:
            del blocks[bid]
            per_block_samples.pop(bid, None)

        # ══════════════════════════════════════════════════════
        #  VISUALISATION
        # ══════════════════════════════════════════════════════
        vis = frame.copy()

        # Phase banner
        if not trained:
            remaining = max(0.0, TRAIN_SECONDS - (now - training_start))
            cv2.putText(vis, f"TRAINING — show 3 separated blocks ({remaining:.1f}s)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.70, (255, 255, 255), 2)
        else:
            cv2.putText(vis, "TRAINED — new block triggers auto / HITL",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.70, (255, 255, 255), 2)
            draw_center_swatches(vis, centers)

        # Draw all detected contour boxes
        for (cx, cy, w, h, cnt) in detections:
            x = cx - w // 2
            y = cy - h // 2
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 0), 1)

        # Draw tracked block labels + stability bars
        for bid, blk in blocks.items():
            x1 = blk.cx - blk.w // 2
            y1 = blk.cy - blk.h // 2

            # Label
            label = f"id{bid}"
            if blk.bin_name != "UNASSIGNED":
                bgr   = BIN_COLORS_BGR.get(blk.bin_name, (200, 200, 200))
                label = f"id{bid} {blk.bin_name}"
                cv2.rectangle(vis, (x1, y1),
                              (x1 + blk.w, y1 + blk.h), bgr, 2)
            cv2.putText(vis, label,
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        (255, 255, 255), 1)

            # Stability bar (only for un-acted blocks during inference)
            if trained and not blk.acted:
                bar_w = int(BAR_MAX_WIDTH *
                            min(blk.stable_count / STABLE_FRAMES, 1.0))
                bx = x1
                by = y1 + blk.h + 4
                cv2.rectangle(vis, (bx, by),
                              (bx + BAR_MAX_WIDTH, by + 8),
                              (50, 50, 50), -1)
                cv2.rectangle(vis, (bx, by),
                              (bx + bar_w, by + 8),
                              (0, 200, 200), -1)

        # Probability bars (for the HITL block or most recently classified)
        if trained and hitl_active and hitl_probs:
            draw_prob_bars(vis, hitl_probs, (10, 70))
        elif trained and not hitl_active:
            # Show bars for the most recently seen un-acted block, if any
            candidate = next(
                (blk for blk in blocks.values()
                 if not blk.acted and blk.stable_count > 0),
                None
            )
            if candidate is not None and centers is not None:
                _, tp, _, _, _ = compute_confidence(candidate.mean_hsv, centers)
                p = {f"BIN_{i+1}": float(1.0 / (
                        dist_hsv(candidate.mean_hsv, centers[i]) + 1e-6))
                     for i in range(3)}
                tot = sum(p.values())
                p   = {k: v / tot for k, v in p.items()}
                draw_prob_bars(vis, p, (10, 70))

        # HITL overlay
        if hitl_active and hitl_mean_hsv is not None:
            draw_hitl_overlay(vis, hitl_top_bin, hitl_top_prob,
                              hitl_probs, hitl_mean_hsv)
            elapsed = now - hitl_start_t
            cv2.putText(vis, f"Waiting {elapsed:.1f}s",
                        (80, 415), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (150, 150, 150), 1)

        # Stats HUD
        draw_session_stats(vis, trial, auto_count, hitl_count, override_count)

        # Live centre-pixel HSV
        h_, s_, v_ = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[
            FRAME_H // 2, FRAME_W // 2]
        cv2.putText(vis, f"H:{h_}  S:{s_}  V:{v_}",
                    (10, FRAME_H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 100), 1)

        cv2.imshow("HITL Edge Pipeline", vis)

        # ══════════════════════════════════════════════════════
        #  KEY HANDLING
        # ══════════════════════════════════════════════════════
        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), 27):
            break

        if key in (ord("c"), ord("C")):
            h_, s_, v_ = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[
                FRAME_H // 2, FRAME_W // 2]
            print(f"\n[CAL] Centre pixel — H:{h_}  S:{s_}  V:{v_}")

        if key in (ord("r"), ord("R")):
            # Manual retrain
            trained         = False
            centers         = None
            training_start  = time.time()
            per_block_samples = {}
            for blk in blocks.values():
                blk.acted     = False
                blk.bin_name  = "UNASSIGNED"
                blk.stable_count = 0
            hitl_active = False
            print("\n[INFO] Retraining started — show 3 blocks again.\n")

        # HITL response
        if hitl_active and key in (ord("1"), ord("2"), ord("3"),
                                   ord("s"), ord("S")):
            rt_ms = (now - hitl_start_t) * 1000

            if key in (ord("s"), ord("S")):
                final_bin        = "SKIPPED"
                operator_override = True
            else:
                final_bin        = BIN_NAMES[chr(key)]
                operator_override = (final_bin != hitl_top_bin)

            if operator_override and final_bin != "SKIPPED":
                override_count += 1

            if final_bin != "SKIPPED":
                bin_counts[final_bin] = bin_counts.get(final_bin, 0) + 1
                sig = make_sig(hitl_mean_hsv)
                # Persist rule if operator consistently assigns same bin
                if not operator_override:
                    rules[sig] = final_bin
                    save_rules(rules)
                robot_action(final_bin)

            rt_list.append(rt_ms)

            # Update the tracked block that triggered HITL
            if hitl_block_id in blocks:
                blk = blocks[hitl_block_id]
                blk.bin_name     = final_bin
                blk.acted        = True
                blk.last_acted_t = time.time()

            log_trial(log_file, {
                "timestamp":         datetime.now().isoformat(),
                "trial":             trial,
                "color_detected":    hitl_top_bin,
                "top_prob":          hitl_top_prob,
                "second_color":      list(hitl_probs.keys())[1]
                                     if len(hitl_probs) > 1 else "--",
                "second_prob":       list(hitl_probs.values())[1]
                                     if len(hitl_probs) > 1 else 0.0,
                "hitl_triggered":    True,
                "operator_override": operator_override,
                "final_bin":         final_bin,
                "response_time_ms":  rt_ms,
                "hsv_H": int(hitl_mean_hsv[0]),
                "hsv_S": int(hitl_mean_hsv[1]),
                "hsv_V": int(hitl_mean_hsv[2]),
            })

            flag = " [OVERRIDE]" if operator_override else ""
            print(f"[HITL]  → {final_bin}  rt={rt_ms:.0f}ms{flag}")
            hitl_active = False

    # ── Cleanup ───────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()
    print_summary(trial, auto_count, hitl_count, override_count,
                  rt_list, bin_counts)
    print(f"\n[LOG] Session saved: {log_file}")


if __name__ == "__main__":
    main()