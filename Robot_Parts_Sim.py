"""
interactive_object_sorter_train_and_pick_robodk.py
==================================================
FLIR / PySpin object sorter with:
- User-defined number of bins
- User-defined object type for each bin
- User-defined RoboDK target for each bin
- Training phase with 5 captures per object type
- On-screen CAPTURE button to record samples
- On-screen NEXT button to move to next object/bin
- Click an object to select it in sorting mode
- Press SPACE to send robot to the selected object's bin
- If classification is uncertain, ask user for input and learn from correction

WORKFLOW
--------
1. Program starts and asks:
      - how many bins?
      - what object type goes in each bin?
      - what RoboDK target name should each bin use?
2. Training phase:
      - place one example object in view
      - click CAPTURE 5 times for that object/bin
      - click NEXT to go to the next object/bin
3. Sorting phase:
      - place objects in view
      - click one object to select it
      - press SPACE to send robot to predicted bin
      - if confidence is low, user is asked which bin it belongs to
      - correction is added to training set

KEYS
----
SPACE      move selected object to predicted/corrected bin
q / ESC    quit

MOUSE
-----
Left click:
- Training mode: CAPTURE or NEXT button
- Sorting mode: select object
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import PySpin

from robodk.robolink import Robolink, ITEM_TYPE_ROBOT, ITEM_TYPE_TARGET


# =========================================================
# SETTINGS — CAMERA / DETECTION
# =========================================================
MIN_OBJECT_AREA = 1200
SET_EXPOSURE_AUTO_OFF = False
EXPOSURE_US = 10_000.0
SET_GAIN_AUTO_OFF = False
GAIN_DB = 5.0

# =========================================================
# SETTINGS — SOFTWARE WHITE BALANCE
# =========================================================
ENABLE_WHITE_BALANCE = True
WB_BLUE_GAIN = 1.00
WB_GREEN_GAIN = 1.00
WB_RED_GAIN = 1.00
WB_BRIGHTNESS_GAIN = 1.00

# =========================================================
# SETTINGS — SEGMENTATION
# =========================================================
SAT_THRESHOLD = 25
DARK_THRESHOLD = 45
BRIGHT_THRESHOLD = 250
MORPH_KERNEL_SIZE = 5
MIN_MASK_PIXELS = 150

# =========================================================
# SETTINGS — TRACKING
# =========================================================
TRACKING_MAX_DISTANCE = 100
TRACKING_MAX_MISSES = 20
TRACKING_MIN_HITS_TO_CONFIRM = 2
TRACKING_COLOR_WEIGHT = 0.35
TRACKING_POSITION_WEIGHT = 1.00
TRACKING_SIZE_WEIGHT = 0.20
TRACK_DELETE_MISSES = 50

# =========================================================
# SETTINGS — CLASSIFICATION
# =========================================================
TRAINING_IMAGES_PER_BIN = 5
UNKNOWN_DISTANCE_THRESHOLD = 32.0
CONFIDENT_DISTANCE_THRESHOLD = 18.0
MAX_FEATURE_HISTORY = 20

# =========================================================
# SETTINGS — DISPLAY
# =========================================================
DISPLAY_SCALE = 0.60
PANEL_PADDING = 12
HEADER_H = 30

# =========================================================
# SETTINGS — ROBODK
# =========================================================
ROBOT_NAME = "Huayan E10"
MIDDLE_TARGET_NAME = "BoardFrame"
ROBOT_ENABLE = True


# =========================================================
# DATA CLASSES
# =========================================================
@dataclass
class BinDefinition:
    index: int
    label: str
    target_name: str


@dataclass
class FeatureSample:
    bin_index: int
    feature: np.ndarray


@dataclass
class DetectedObject:
    cx: int
    cy: int
    w: int
    h: int
    contour: np.ndarray
    area: float
    mean_bgr: Tuple[int, int, int]
    mean_hsv: Tuple[float, float, float]
    mean_lab: Tuple[float, float, float]
    feature: Optional[np.ndarray] = None
    object_id: int = 0
    pred_bin_index: Optional[int] = None
    pred_label: str = "unknown"
    confidence: float = 0.0
    distance: float = 1e9


@dataclass
class Track:
    track_id: int
    cx: int
    cy: int
    w: int
    h: int
    area: float
    mean_bgr: Tuple[int, int, int]
    mean_hsv: Tuple[float, float, float]
    mean_lab: Tuple[float, float, float]
    contour: Optional[np.ndarray] = None
    feature: Optional[np.ndarray] = None

    hits: int = 1
    misses: int = 0
    confirmed: bool = False

    pred_bin_index: Optional[int] = None
    pred_label: str = "unknown"
    confidence: float = 0.0
    distance: float = 1e9


@dataclass
class TrainingState:
    bins: List[BinDefinition]
    current_bin_idx: int = 0
    captured_for_current: int = 0
    done: bool = False

    def current_bin(self) -> BinDefinition:
        return self.bins[self.current_bin_idx]


@dataclass
class UIState:
    mode: str = "training"   # "training" or "sorting"
    selected_track_id: Optional[int] = None
    message: str = ""
    capture_button: Tuple[int, int, int, int] = (0, 0, 0, 0)
    next_button: Tuple[int, int, int, int] = (0, 0, 0, 0)


# =========================================================
# CAMERA HELPERS
# =========================================================
def _set_enumeration_node(nodemap, node_name: str, entry_name: str) -> bool:
    node = PySpin.CEnumerationPtr(nodemap.GetNode(node_name))
    if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
        return False
    entry = node.GetEntryByName(entry_name)
    if not PySpin.IsAvailable(entry) or not PySpin.IsReadable(entry):
        return False
    node.SetIntValue(entry.GetValue())
    return True


def _set_float_node(nodemap, node_name: str, value: float) -> bool:
    node = PySpin.CFloatPtr(nodemap.GetNode(node_name))
    if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
        return False
    value = max(node.GetMin(), min(node.GetMax(), value))
    node.SetValue(value)
    return True


def _try_set_first_pixel_format(nodemap, candidates: List[str]) -> Optional[str]:
    for fmt in candidates:
        if _set_enumeration_node(nodemap, "PixelFormat", fmt):
            return fmt
    return None


def _get_current_pixel_format(nodemap) -> str:
    try:
        node = PySpin.CEnumerationPtr(nodemap.GetNode("PixelFormat"))
        if not PySpin.IsAvailable(node) or not PySpin.IsReadable(node):
            return "Unknown"
        entry = PySpin.CEnumEntryPtr(node.GetCurrentEntry())
        if not PySpin.IsAvailable(entry) or not PySpin.IsReadable(entry):
            return "Unknown"
        return entry.GetSymbolic()
    except Exception:
        return "Unknown"


def init_flir_camera():
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    if cam_list.GetSize() == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        raise RuntimeError("No FLIR cameras detected.")

    cam = cam_list.GetByIndex(0)
    cam.Init()
    nodemap = cam.GetNodeMap()

    _set_enumeration_node(nodemap, "AcquisitionMode", "Continuous")

    chosen_pf = _try_set_first_pixel_format(
        nodemap,
        ["BayerRG8", "BayerBG8", "BayerGR8", "BayerGB8", "RGB8", "BGR8", "Mono8"],
    )
    print(f"[INFO] Requested PixelFormat -> {chosen_pf if chosen_pf else 'unchanged'}")
    print(f"[INFO] Active PixelFormat    -> {_get_current_pixel_format(nodemap)}")

    if SET_EXPOSURE_AUTO_OFF:
        if _set_enumeration_node(nodemap, "ExposureAuto", "Off"):
            _set_float_node(nodemap, "ExposureTime", EXPOSURE_US)

    if SET_GAIN_AUTO_OFF:
        if _set_enumeration_node(nodemap, "GainAuto", "Off"):
            _set_float_node(nodemap, "Gain", GAIN_DB)

    cam.BeginAcquisition()

    processor = PySpin.ImageProcessor()
    try:
        processor.SetColorProcessing(PySpin.HQ_LINEAR)
    except Exception:
        pass

    return system, cam_list, cam, processor


def release_flir_camera(system, cam_list, cam):
    try:
        if cam:
            try:
                cam.EndAcquisition()
            except Exception:
                pass
            try:
                cam.DeInit()
            except Exception:
                pass
    finally:
        try:
            if cam_list:
                cam_list.Clear()
        except Exception:
            pass
        try:
            if system:
                system.ReleaseInstance()
        except Exception:
            pass


def get_bgr_frame(cam, processor):
    image_result = cam.GetNextImage(1000)
    try:
        if image_result.IsIncomplete():
            return None

        converted = processor.Convert(image_result, PySpin.PixelFormat_BGR8)
        arr = converted.GetNDArray()
        if arr is None:
            return None

        if arr.ndim == 2:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        if arr.ndim == 3:
            if arr.shape[2] == 1:
                return cv2.cvtColor(arr[:, :, 0], cv2.COLOR_GRAY2BGR)
            if arr.shape[2] == 3:
                return arr.copy()
            if arr.shape[2] == 4:
                return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

        return None
    finally:
        image_result.Release()


# =========================================================
# IMAGE HELPERS
# =========================================================
def apply_white_balance(frame_bgr: np.ndarray) -> np.ndarray:
    if not ENABLE_WHITE_BALANCE:
        return frame_bgr
    f = frame_bgr.astype(np.float32)
    b, g, r = cv2.split(f)
    b *= WB_BLUE_GAIN
    g *= WB_GREEN_GAIN
    r *= WB_RED_GAIN
    balanced = cv2.merge((b, g, r)) * WB_BRIGHTNESS_GAIN
    return np.clip(balanced, 0, 255).astype(np.uint8)


def build_object_mask_from_hsv(frame_hsv: np.ndarray) -> np.ndarray:
    _, s, v = cv2.split(frame_hsv)
    sat_mask = cv2.inRange(s, SAT_THRESHOLD, 255)
    dark_mask = cv2.inRange(v, 0, DARK_THRESHOLD)
    not_glare = cv2.inRange(v, 0, BRIGHT_THRESHOLD)

    obj_mask = cv2.bitwise_or(sat_mask, dark_mask)
    obj_mask = cv2.bitwise_and(obj_mask, not_glare)

    kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    obj_mask = cv2.morphologyEx(obj_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    obj_mask = cv2.dilate(obj_mask, kernel, iterations=1)
    return obj_mask


def compute_mean_colors_cached(frame_bgr, frame_hsv, frame_lab, obj_mask):
    valid = obj_mask > 0
    if not np.any(valid):
        return (0, 0, 0), (0.0, 0.0, 0.0), (0.0, 128.0, 128.0)

    px_bgr = frame_bgr[valid].astype(np.float32)
    px_hsv = frame_hsv[valid].astype(np.float32)
    px_lab = frame_lab[valid].astype(np.float32)

    m_bgr = tuple(int(v) for v in np.mean(px_bgr, axis=0))
    m_hsv = tuple(float(v) for v in np.mean(px_hsv, axis=0))
    m_lab = tuple(float(v) for v in np.mean(px_lab, axis=0))
    return m_bgr, m_hsv, m_lab


def contour_shape_features(cnt: np.ndarray, area: float, w: int, h: int) -> np.ndarray:
    perimeter = cv2.arcLength(cnt, True)
    hull = cv2.convexHull(cnt)
    hull_area = max(cv2.contourArea(hull), 1.0)

    circularity = (4.0 * np.pi * area / (perimeter * perimeter)) if perimeter > 1e-6 else 0.0
    solidity = area / hull_area
    aspect = float(w) / max(h, 1)
    extent = area / max(float(w * h), 1.0)

    moments = cv2.moments(cnt)
    hu = cv2.HuMoments(moments).flatten()
    hu = np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
    vertex_count = float(len(approx))

    return np.array(
        [
            area,
            perimeter,
            circularity,
            solidity,
            aspect,
            extent,
            vertex_count,
            hu[0],
            hu[1],
            hu[2],
        ],
        dtype=np.float32,
    )


def build_feature_vector(obj: "DetectedObject") -> np.ndarray:
    shape = contour_shape_features(obj.contour, obj.area, obj.w, obj.h)
    color = np.array(
        [
            obj.mean_hsv[0],
            obj.mean_hsv[1],
            obj.mean_hsv[2],
            obj.mean_lab[0],
            obj.mean_lab[1],
            obj.mean_lab[2],
        ],
        dtype=np.float32,
    )
    feat = np.concatenate([shape, color], axis=0)
    return feat.astype(np.float32)


def detect_objects(frame_bgr: np.ndarray) -> List[DetectedObject]:
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    frame_lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)

    mask = build_object_mask_from_hsv(frame_hsv)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    erode_kernel = np.ones((3, 3), np.uint8)
    objects: List[DetectedObject] = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_OBJECT_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        obj_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.drawContours(obj_mask, [cnt], -1, 255, thickness=-1)
        obj_mask = cv2.erode(obj_mask, erode_kernel, iterations=1)

        if np.count_nonzero(obj_mask) < MIN_MASK_PIXELS:
            continue

        m_bgr, m_hsv, m_lab = compute_mean_colors_cached(frame_bgr, frame_hsv, frame_lab, obj_mask)

        obj = DetectedObject(
            cx=x + w // 2,
            cy=y + h // 2,
            w=w,
            h=h,
            contour=cnt,
            area=area,
            mean_bgr=m_bgr,
            mean_hsv=m_hsv,
            mean_lab=m_lab,
        )
        obj.feature = build_feature_vector(obj)
        objects.append(obj)

    return objects


# =========================================================
# TRACKING
# =========================================================
def _match_cost(track: Track, det: DetectedObject) -> float:
    pos_dist = float(np.hypot(det.cx - track.cx, det.cy - track.cy))
    if pos_dist > TRACKING_MAX_DISTANCE:
        return 1e9

    lab_track = np.array(track.mean_lab, dtype=np.float32)
    lab_det = np.array(det.mean_lab, dtype=np.float32)
    color_dist = float(np.linalg.norm(lab_det - lab_track))

    size_dist = abs(float(det.area - track.area)) / max(track.area, 1.0)

    return (
        TRACKING_POSITION_WEIGHT * pos_dist
        + TRACKING_COLOR_WEIGHT * color_dist
        + TRACKING_SIZE_WEIGHT * 100.0 * size_dist
    )


class StableObjectTracker:
    def __init__(self):
        self.next_track_id = 1
        self.tracks: Dict[int, Track] = {}

    def _create_track(self, det: DetectedObject):
        tid = self.next_track_id
        self.next_track_id += 1
        self.tracks[tid] = Track(
            track_id=tid,
            cx=det.cx,
            cy=det.cy,
            w=det.w,
            h=det.h,
            area=det.area,
            mean_bgr=det.mean_bgr,
            mean_hsv=det.mean_hsv,
            mean_lab=det.mean_lab,
            contour=det.contour,
            feature=det.feature.copy() if det.feature is not None else None,
            hits=1,
            misses=0,
            confirmed=False,
        )

    def update(self, detections: List[DetectedObject]):
        track_ids = list(self.tracks.keys())
        unmatched_dets = set(range(len(detections)))
        assigned_tracks = set()

        candidate_pairs = []
        for tid in track_ids:
            tr = self.tracks[tid]
            for i, det in enumerate(detections):
                cost = _match_cost(tr, det)
                if cost < 1e9:
                    candidate_pairs.append((cost, tid, i))
        candidate_pairs.sort(key=lambda x: x[0])

        used_tracks = set()
        used_dets = set()
        matched_pairs = []

        for cost, tid, i in candidate_pairs:
            if tid in used_tracks or i in used_dets:
                continue
            used_tracks.add(tid)
            used_dets.add(i)
            matched_pairs.append((tid, i))

        for tid, i in matched_pairs:
            det = detections[i]
            tr = self.tracks[tid]
            tr.cx = det.cx
            tr.cy = det.cy
            tr.w = det.w
            tr.h = det.h
            tr.area = det.area
            tr.mean_bgr = det.mean_bgr
            tr.mean_hsv = det.mean_hsv
            tr.mean_lab = det.mean_lab
            tr.contour = det.contour
            tr.feature = det.feature.copy() if det.feature is not None else None
            tr.hits += 1
            tr.misses = 0
            if tr.hits >= TRACKING_MIN_HITS_TO_CONFIRM:
                tr.confirmed = True
            unmatched_dets.discard(i)
            assigned_tracks.add(tid)

        for tid in track_ids:
            if tid not in assigned_tracks:
                self.tracks[tid].misses += 1

        for i in sorted(unmatched_dets):
            self._create_track(detections[i])

    def cleanup_stale_tracks(self):
        stale_ids = [tid for tid, tr in self.tracks.items() if tr.misses > TRACK_DELETE_MISSES]
        for tid in stale_ids:
            del self.tracks[tid]

    def get_visible_tracks(self) -> List[Track]:
        return [tr for tr in self.tracks.values() if tr.confirmed and tr.misses == 0 and tr.contour is not None]

    def get_visible_objects(self) -> List[DetectedObject]:
        out: List[DetectedObject] = []
        for tr in self.get_visible_tracks():
            out.append(
                DetectedObject(
                    cx=tr.cx,
                    cy=tr.cy,
                    w=tr.w,
                    h=tr.h,
                    contour=tr.contour,
                    area=tr.area,
                    mean_bgr=tr.mean_bgr,
                    mean_hsv=tr.mean_hsv,
                    mean_lab=tr.mean_lab,
                    feature=tr.feature.copy() if tr.feature is not None else None,
                    object_id=tr.track_id,
                    pred_bin_index=tr.pred_bin_index,
                    pred_label=tr.pred_label,
                    confidence=tr.confidence,
                    distance=tr.distance,
                )
            )
        out.sort(key=lambda o: o.object_id)
        return out

    def update_prediction(self, track_id: int, pred_bin_index: Optional[int], pred_label: str, confidence: float, distance: float):
        tr = self.tracks.get(track_id)
        if tr is None:
            return
        tr.pred_bin_index = pred_bin_index
        tr.pred_label = pred_label
        tr.confidence = confidence
        tr.distance = distance


# =========================================================
# SIMPLE CLASSIFIER
# =========================================================
class FeatureClassifier:
    def __init__(self):
        self.samples: List[FeatureSample] = []
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def add_sample(self, bin_index: int, feature: np.ndarray):
        self.samples.append(FeatureSample(bin_index=bin_index, feature=feature.astype(np.float32)))
        self._recompute_norm()

    def has_data(self) -> bool:
        return len(self.samples) > 0

    def _recompute_norm(self):
        if not self.samples:
            self.mean = None
            self.std = None
            return
        X = np.stack([s.feature for s in self.samples], axis=0).astype(np.float32)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std = np.where(self.std < 1e-3, 1.0, self.std).astype(np.float32)

    def predict(self, feature: np.ndarray, bins: List[BinDefinition]) -> Tuple[Optional[int], str, float, float]:
        if not self.samples or self.mean is None or self.std is None:
            return None, "unknown", 0.0, 1e9

        fz = (feature - self.mean) / self.std

        best_bin = None
        best_dist = 1e9

        for b in bins:
            feats = [s.feature for s in self.samples if s.bin_index == b.index]
            if not feats:
                continue
            X = np.stack(feats, axis=0).astype(np.float32)
            Xz = (X - self.mean) / self.std
            dists = np.linalg.norm(Xz - fz[None, :], axis=1)
            d = float(np.min(dists))
            if d < best_dist:
                best_dist = d
                best_bin = b.index

        if best_bin is None:
            return None, "unknown", 0.0, 1e9

        label = bins[best_bin].label
        if best_dist <= CONFIDENT_DISTANCE_THRESHOLD:
            conf = 1.0
        elif best_dist >= UNKNOWN_DISTANCE_THRESHOLD:
            conf = 0.0
        else:
            conf = 1.0 - ((best_dist - CONFIDENT_DISTANCE_THRESHOLD) / max(UNKNOWN_DISTANCE_THRESHOLD - CONFIDENT_DISTANCE_THRESHOLD, 1e-6))

        return best_bin, label, float(conf), float(best_dist)


# =========================================================
# ROBODK
# =========================================================
class RoboDKController:
    def __init__(self, bins: List[BinDefinition], enabled: bool = True):
        self.enabled = enabled
        self.rdk = None
        self.robot = None
        self.middle_target = None
        self.middle_available = False
        self.bin_targets: Dict[int, object] = {}

        if not self.enabled:
            print("[INFO] RoboDK disabled.")
            return

        try:
            self.rdk = Robolink()
            self.robot = self.rdk.Item(ROBOT_NAME, ITEM_TYPE_ROBOT)
            if not self.robot.Valid():
                raise RuntimeError(f"Robot '{ROBOT_NAME}' not found in RoboDK.")

            self.middle_target = self.rdk.Item(MIDDLE_TARGET_NAME, ITEM_TYPE_TARGET)
            self.middle_available = self.middle_target.Valid()

            for b in bins:
                tgt = self.rdk.Item(b.target_name, ITEM_TYPE_TARGET)
                if not tgt.Valid():
                    raise RuntimeError(f"Target '{b.target_name}' not found in RoboDK.")
                self.bin_targets[b.index] = tgt

            print("[INFO] RoboDK connected.")
        except Exception as e:
            self.enabled = False
            self.rdk = None
            self.robot = None
            self.middle_target = None
            self.middle_available = False
            self.bin_targets = {}
            print(f"[WARN] RoboDK disabled: {e}")

    def move_to_bin_then_middle(self, bin_def: BinDefinition, object_id: int) -> bool:
        if not self.enabled:
            print(f"[SIM-OFF] Would move object #{object_id} to {bin_def.label} ({bin_def.target_name})")
            return True

        if self.robot is None or not self.robot.Valid():
            print("[WARN] Robot not valid.")
            return False

        target = self.bin_targets.get(bin_def.index)
        if target is None or not target.Valid():
            print(f"[WARN] Target invalid for bin {bin_def.label}")
            return False

        try:
            print(f"[ROBOT] Moving object #{object_id} -> {bin_def.label} ({bin_def.target_name})")
            self.robot.MoveJ(target)
            if self.middle_available and self.middle_target is not None and self.middle_target.Valid():
                self.robot.MoveJ(self.middle_target)
            return True
        except Exception as e:
            print(f"[ERROR] RoboDK move failed: {e}")
            return False


# =========================================================
# UI
# =========================================================
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

_BG = (20, 20, 20)
_WHITE = (255, 255, 255)
_GRAY = (170, 170, 170)
_GREEN = (80, 210, 120)
_ORANGE = (80, 180, 255)
_RED = (80, 80, 255)
_BLUE = (255, 180, 80)


def draw_header(img, text: str):
    cv2.rectangle(img, (0, 0), (img.shape[1], HEADER_H), (45, 45, 45), -1)
    cv2.putText(img, text, (12, HEADER_H - 8), FONT, 0.55, _WHITE, 1, cv2.LINE_AA)


def draw_button(img, rect: Tuple[int, int, int, int], label: str, color: Tuple[int, int, int]):
    x1, y1, x2, y2 = rect
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (20, 20, 20), 2)
    (tw, th), _ = cv2.getTextSize(label, FONT_BOLD, 0.7, 2)
    tx = x1 + (x2 - x1 - tw) // 2
    ty = y1 + (y2 - y1 + th) // 2
    cv2.putText(img, label, (tx, ty), FONT_BOLD, 0.7, (0, 0, 0), 2, cv2.LINE_AA)


def point_in_rect(x: int, y: int, rect: Tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2


def render_training_panel(frame: np.ndarray, ui: UIState, training: TrainingState, visible_objects: List[DetectedObject], classifier: FeatureClassifier):
    img = frame.copy()
    H, W = img.shape[:2]
    draw_header(img, "TRAINING MODE")

    current_bin = training.current_bin()
    status_lines = [
        f"Train bin {current_bin.index + 1}/{len(training.bins)}",
        f"Object type: {current_bin.label}",
        f"RoboDK target: {current_bin.target_name}",
        f"Captured: {training.captured_for_current}/{TRAINING_IMAGES_PER_BIN}",
        f"Samples total: {len(classifier.samples)}",
        "Place one object in view, then click CAPTURE.",
    ]

    y = HEADER_H + 28
    for line in status_lines:
        cv2.putText(img, line, (15, y), FONT, 0.65, _WHITE, 2, cv2.LINE_AA)
        y += 28

    for obj in visible_objects:
        x = obj.cx - obj.w // 2
        y0 = obj.cy - obj.h // 2
        cv2.rectangle(img, (x, y0), (x + obj.w, y0 + obj.h), _BLUE, 2)
        cv2.putText(img, f"#{obj.object_id}", (x, max(18, y0 - 8)), FONT_BOLD, 0.6, _BLUE, 2, cv2.LINE_AA)

    ui.capture_button = (W - 260, H - 90, W - 140, H - 30)
    ui.next_button = (W - 130, H - 90, W - 20, H - 30)

    draw_button(img, ui.capture_button, "CAPTURE", _GREEN)
    draw_button(img, ui.next_button, "NEXT", _ORANGE)

    if ui.message:
        cv2.putText(img, ui.message, (15, H - 25), FONT, 0.6, _ORANGE, 2, cv2.LINE_AA)

    return img


def render_sorting_panel(frame: np.ndarray, ui: UIState, objects: List[DetectedObject], bins: List[BinDefinition]):
    img = frame.copy()
    H, W = img.shape[:2]
    draw_header(img, "SORTING MODE  |  Click object to select, SPACE to move robot")

    for obj in objects:
        x = obj.cx - obj.w // 2
        y = obj.cy - obj.h // 2

        selected = (ui.selected_track_id == obj.object_id)
        color = _GREEN if selected else _BLUE
        thickness = 3 if selected else 2
        cv2.rectangle(img, (x, y), (x + obj.w, y + obj.h), color, thickness)

        label = obj.pred_label
        bin_txt = "?"
        if obj.pred_bin_index is not None and 0 <= obj.pred_bin_index < len(bins):
            bin_txt = f"bin {obj.pred_bin_index + 1}"

        cv2.putText(
            img,
            f"#{obj.object_id} {label} {obj.confidence:.2f} {bin_txt}",
            (x, max(18, y - 8)),
            FONT,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    panel_y = H - 110
    cv2.rectangle(img, (0, panel_y), (W, H), (30, 30, 30), -1)

    lines = []
    if ui.selected_track_id is None:
        lines.append("No object selected.")
    else:
        sel = next((o for o in objects if o.object_id == ui.selected_track_id), None)
        if sel is None:
            lines.append("Selected object no longer visible.")
        else:
            lines.append(f"Selected object #{sel.object_id}")
            lines.append(f"Prediction: {sel.pred_label}")
            lines.append(f"Confidence: {sel.confidence:.2f}, distance: {sel.distance:.2f}")
            lines.append("Press SPACE to move robot.")

    if ui.message:
        lines.append(ui.message)

    yy = panel_y + 25
    for line in lines[:4]:
        cv2.putText(img, line, (15, yy), FONT, 0.6, _WHITE, 1, cv2.LINE_AA)
        yy += 24

    return img


# =========================================================
# USER INPUT
# =========================================================
def ask_user_configuration() -> List[BinDefinition]:
    while True:
        try:
            n = int(input("How many bins do you want? ").strip())
            if n <= 0:
                raise ValueError
            break
        except Exception:
            print("Please enter a positive integer.")

    bins: List[BinDefinition] = []
    for i in range(n):
        label = input(f"What object type goes in bin {i + 1}? ").strip()
        if not label:
            label = f"object_type_{i+1}"
        target_name = input(f"What RoboDK target name should bin {i + 1} use? ").strip()
        if not target_name:
            target_name = f"BIN_{i+1}_DROP"
        bins.append(BinDefinition(index=i, label=label, target_name=target_name))
    return bins


def ask_user_for_uncertain_bin(obj: DetectedObject, bins: List[BinDefinition]) -> int:
    print("\n[QUESTION] I am not confident about this object.")
    print(f"Object #{obj.object_id}: predicted='{obj.pred_label}', confidence={obj.confidence:.2f}, distance={obj.distance:.2f}")
    for b in bins:
        print(f"  {b.index + 1}. {b.label} -> {b.target_name}")

    while True:
        try:
            raw = input("Which bin should this object go to? Enter number: ").strip()
            idx = int(raw) - 1
            if 0 <= idx < len(bins):
                return idx
        except Exception:
            pass
        print("Invalid input. Try again.")


# =========================================================
# GLOBAL MOUSE STATE
# =========================================================
MOUSE_CLICK = {"x": -1, "y": -1, "clicked": False}


def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        MOUSE_CLICK["x"] = x
        MOUSE_CLICK["y"] = y
        MOUSE_CLICK["clicked"] = True


# =========================================================
# MAIN
# =========================================================
def main():
    bins = ask_user_configuration()
    training = TrainingState(bins=bins)
    ui = UIState(mode="training", message="Click CAPTURE when one object is visible.")
    classifier = FeatureClassifier()
    tracker = StableObjectTracker()
    robot = RoboDKController(bins=bins, enabled=ROBOT_ENABLE)

    system = cam_list = cam = processor = None

    try:
        system, cam_list, cam, processor = init_flir_camera()

        cv2.namedWindow("Interactive Object Sorter")
        cv2.setMouseCallback("Interactive Object Sorter", mouse_callback)

        print("\nTraining will start now.")
        print("For each object type/bin:")
        print(f"  - place one object in view")
        print(f"  - click CAPTURE {TRAINING_IMAGES_PER_BIN} times")
        print("  - click NEXT to continue\n")

        while True:
            frame = get_bgr_frame(cam, processor)
            if frame is None:
                continue

            frame = apply_white_balance(frame)
            detections = detect_objects(frame)
            tracker.update(detections)

            visible_objects = tracker.get_visible_objects()

            for obj in visible_objects:
                if obj.feature is None:
                    continue
                pred_bin, pred_label, conf, dist = classifier.predict(obj.feature, bins)
                tracker.update_prediction(obj.object_id, pred_bin, pred_label, conf, dist)

            tracker.cleanup_stale_tracks()
            visible_objects = tracker.get_visible_objects()

            # -------------------------
            # Training Mode
            # -------------------------
            if ui.mode == "training":
                display = render_training_panel(frame, ui, training, visible_objects, classifier)

                if MOUSE_CLICK["clicked"]:
                    mx = int(MOUSE_CLICK["x"] / DISPLAY_SCALE) if DISPLAY_SCALE != 1.0 else MOUSE_CLICK["x"]
                    my = int(MOUSE_CLICK["y"] / DISPLAY_SCALE) if DISPLAY_SCALE != 1.0 else MOUSE_CLICK["y"]
                    MOUSE_CLICK["clicked"] = False

                    if point_in_rect(mx, my, ui.capture_button):
                        if len(visible_objects) != 1:
                            ui.message = "Exactly one object must be visible for training."
                        else:
                            obj = visible_objects[0]
                            if obj.feature is None:
                                ui.message = "Could not extract features."
                            else:
                                classifier.add_sample(training.current_bin_idx, obj.feature)
                                training.captured_for_current += 1
                                ui.message = (
                                    f"Captured {training.captured_for_current}/{TRAINING_IMAGES_PER_BIN} "
                                    f"for '{training.current_bin().label}'."
                                )

                    elif point_in_rect(mx, my, ui.next_button):
                        if training.captured_for_current < TRAINING_IMAGES_PER_BIN:
                            ui.message = f"Need {TRAINING_IMAGES_PER_BIN} captures before NEXT."
                        else:
                            training.current_bin_idx += 1
                            training.captured_for_current = 0
                            if training.current_bin_idx >= len(training.bins):
                                training.done = True
                                ui.mode = "sorting"
                                ui.message = "Training complete. Click an object, then press SPACE."
                                print("[INFO] Training complete. Sorting mode started.")
                            else:
                                ui.message = f"Now train '{training.current_bin().label}'."

            # -------------------------
            # Sorting Mode
            # -------------------------
            else:
                display = render_sorting_panel(frame, ui, visible_objects, bins)

                if MOUSE_CLICK["clicked"]:
                    mx = int(MOUSE_CLICK["x"] / DISPLAY_SCALE) if DISPLAY_SCALE != 1.0 else MOUSE_CLICK["x"]
                    my = int(MOUSE_CLICK["y"] / DISPLAY_SCALE) if DISPLAY_SCALE != 1.0 else MOUSE_CLICK["y"]
                    MOUSE_CLICK["clicked"] = False

                    picked_id = None
                    for obj in visible_objects:
                        x1 = obj.cx - obj.w // 2
                        y1 = obj.cy - obj.h // 2
                        x2 = x1 + obj.w
                        y2 = y1 + obj.h
                        if x1 <= mx <= x2 and y1 <= my <= y2:
                            picked_id = obj.object_id
                            break

                    if picked_id is not None:
                        ui.selected_track_id = picked_id
                        sel = next((o for o in visible_objects if o.object_id == picked_id), None)
                        if sel is not None:
                            ui.message = f"Selected object #{sel.object_id}: {sel.pred_label} ({sel.confidence:.2f})"

            if DISPLAY_SCALE != 1.0:
                display_show = cv2.resize(display, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE, interpolation=cv2.INTER_AREA)
            else:
                display_show = display

            cv2.imshow("Interactive Object Sorter", display_show)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                if ui.mode != "sorting":
                    ui.message = "Finish training first."
                elif ui.selected_track_id is None:
                    ui.message = "Select an object first."
                else:
                    sel = next((o for o in visible_objects if o.object_id == ui.selected_track_id), None)
                    if sel is None:
                        ui.message = "Selected object is not visible."
                    else:
                        chosen_bin_index = sel.pred_bin_index

                        if chosen_bin_index is None or sel.confidence < 0.50 or sel.distance > UNKNOWN_DISTANCE_THRESHOLD:
                            chosen_bin_index = ask_user_for_uncertain_bin(sel, bins)
                            if sel.feature is not None:
                                classifier.add_sample(chosen_bin_index, sel.feature)
                                ui.message = (
                                    f"User corrected object #{sel.object_id} to '{bins[chosen_bin_index].label}'. "
                                    "Learned from correction."
                                )
                        else:
                            ui.message = f"Moving object #{sel.object_id} to '{bins[chosen_bin_index].label}'."

                        ok = robot.move_to_bin_then_middle(bins[chosen_bin_index], sel.object_id)
                        if ok:
                            print(f"[DONE] Object #{sel.object_id} -> {bins[chosen_bin_index].label}")
                        else:
                            print(f"[FAIL] Robot move failed for object #{sel.object_id}")

            elif key in (ord("q"), 27):
                break

    except Exception as e:
        print(f"[ERROR] {e}")
        raise
    finally:
        cv2.destroyAllWindows()
        release_flir_camera(system, cam_list, cam)


if __name__ == "__main__":
    main()