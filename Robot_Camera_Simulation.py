"""
camera_sorter_4panel_spacebar_queue_robodk.py
=============================================
FLIR / PySpin color sorter with:
- FAST processing
- STABLE object IDs
- EXACTLY 3 bin assignments
- Manual robot trigger on SPACEBAR
- Queue multiple removed objects, then service them sequentially
- Robot goes to bin, then returns to MIDDLE after each queued object
- Panel D shows a job/log table:
    object number | assigned bin | robot put to

WORKFLOW
--------
1. Let objects stabilize in view.
2. Physically pick one or more objects with tweezers.
3. Press SPACEBAR.
4. The script finds all removed objects not yet processed and:
      bin -> middle -> next bin -> middle -> ...
5. Panel D updates "robot put to" for each processed object.

KEYS
----
SPACE      process removed-object queue
q / ESC    quit
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
ACTION_COOLDOWN_S = 2.5

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
SAT_THRESHOLD = 30
DARK_THRESHOLD = 40
BRIGHT_THRESHOLD = 250
MORPH_KERNEL_SIZE = 5
MIN_MASK_PIXELS = 150

# =========================================================
# SETTINGS — CLUSTERING / BINS
# =========================================================
ENABLE_DYNAMIC_BINS = True
FREEZE_CLUSTERS_AFTER_INIT = True
CLUSTER_UPDATE_EVERY_N_FRAMES = 20
MIN_COLOR_CLUSTERS = 3
MAX_COLOR_CLUSTERS = 3
MIN_OBJECTS_FOR_CLUSTERING = 3

MAX_LAB_DISTANCE_FOR_CONFIDENCE = 45.0

BIN_SLOTS = [
    {"id": "BIN_1_DROP", "cost": 1.0},
    {"id": "BIN_2_DROP", "cost": 1.5},
    {"id": "BIN_3_DROP", "cost": 2.0},
]

CONF_HIGH = 0.70
CONF_MEDIUM = 0.40

# =========================================================
# SETTINGS — TRACKING / STABILITY
# =========================================================
TRACKING_MAX_DISTANCE = 90
TRACKING_MAX_MISSES = 15
TRACKING_MIN_HITS_TO_CONFIRM = 2
TRACKING_COLOR_WEIGHT = 0.30
TRACKING_POSITION_WEIGHT = 1.00
TRACKING_SIZE_WEIGHT = 0.15

# Removed object must be absent this many frames before SPACEBAR can queue it
REMOVAL_TRIGGER_MISSES = 8
TRACK_DELETE_MISSES = 50

# =========================================================
# SETTINGS — DISPLAY
# =========================================================
DISPLAY_SCALE = 0.55
PANEL_PADDING = 12
SWATCH_H = 40
CLUSTER_ROW_H = 28
TABLE_ROW_H = 26

# =========================================================
# SETTINGS — ROBODK
# =========================================================
ROBOT_NAME = "Huayan E10"
MIDDLE_TARGET_NAME = "MIDDLE"
ROBOT_ENABLE = True
ROBOT_MOVE_BLOCKING = True

BIN_TARGET_NAMES = {
    "BIN_1_DROP": "BIN_1_DROP",
    "BIN_2_DROP": "BIN_2_DROP",
    "BIN_3_DROP": "BIN_3_DROP",
}


# =========================================================
# DATA
# =========================================================
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
    object_id: int = 0
    color_name: str = "unknown"
    confidence: float = 0.0
    cluster_id: int = -1
    target_bin: str = "N/A"
    cluster_distance: float = 0.0


@dataclass
class ClusterState:
    k: int
    centers_lab: np.ndarray
    label_to_bin: Dict[int, str]
    cluster_display_names: Dict[int, str]
    cluster_counts: Dict[int, int]
    cluster_bgr: Dict[int, Tuple[int, int, int]] = field(default_factory=dict)


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

    hits: int = 1
    misses: int = 0
    confirmed: bool = False

    cluster_id: int = -1
    target_bin: str = "N/A"
    color_name: str = "unknown"
    confidence: float = 0.0
    cluster_distance: float = 0.0

    queued: bool = False
    processed: bool = False


@dataclass
class JobRecord:
    object_id: int
    assigned_bin: str = "N/A"
    robot_put_to: str = "-"
    status: str = "VISIBLE"


CLUSTER_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def cluster_letter(cluster_id: int) -> str:
    if 0 <= cluster_id < len(CLUSTER_LETTERS):
        return CLUSTER_LETTERS[cluster_id]
    return f"C{cluster_id}"


# =========================================================
# PYSPIN CAMERA HELPERS
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

    if _set_enumeration_node(nodemap, "AcquisitionMode", "Continuous"):
        print("[INFO] AcquisitionMode -> Continuous")

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

    try:
        serial = cam.TLDevice.DeviceSerialNumber.GetValue()
    except Exception:
        serial = "Unknown"

    print(f"[INFO] FLIR camera ready. Serial: {serial}")
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
# WHITE BALANCE
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


# =========================================================
# COLOR HELPERS
# =========================================================
NAMED_COLORS: Dict[str, Tuple[int, int, int]] = {
    "red":     (0, 0, 220),
    "orange":  (0, 140, 255),
    "yellow":  (0, 220, 220),
    "green":   (0, 200, 0),
    "blue":    (220, 50, 0),
    "purple":  (200, 0, 200),
    "pink":    (180, 150, 255),
    "white":   (230, 230, 230),
    "gray":    (150, 150, 150),
    "black":   (60, 60, 60),
    "unknown": (200, 200, 200),
}


def named_bgr(name: str) -> Tuple[int, int, int]:
    return NAMED_COLORS.get(name, (200, 200, 200))


def bgr_to_color_name(mean_bgr: Tuple[int, int, int]) -> str:
    patch = np.uint8([[list(mean_bgr)]])
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])

    if v < 45:
        return "black"
    if v > 210 and s < 25:
        return "white"
    if s < 30:
        return "gray"
    if h < 8 or h >= 172:
        return "red"
    if h < 20:
        return "orange"
    if h < 35:
        return "yellow"
    if h < 85:
        return "green"
    if h < 130:
        return "blue"
    if h < 160:
        return "purple"
    return "pink"


def lab_to_display_name(center_lab: np.ndarray) -> str:
    bgr = cv2.cvtColor(np.uint8([[center_lab]]), cv2.COLOR_LAB2BGR)[0, 0]
    return bgr_to_color_name((int(bgr[0]), int(bgr[1]), int(bgr[2])))


def lab_to_bgr(center_lab: np.ndarray) -> Tuple[int, int, int]:
    bgr = cv2.cvtColor(np.uint8([[center_lab]]), cv2.COLOR_LAB2BGR)[0, 0]
    return (int(bgr[0]), int(bgr[1]), int(bgr[2]))


# =========================================================
# FAST SEGMENTATION / COLOR MEASUREMENT
# =========================================================
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


def detect_objects(frame_bgr: np.ndarray):
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

        objects.append(
            DetectedObject(
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
        )

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

    def get_visible_objects(self) -> List[DetectedObject]:
        objects: List[DetectedObject] = []
        for tid, tr in self.tracks.items():
            if not tr.confirmed or tr.misses != 0 or tr.contour is None:
                continue
            objects.append(
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
                    object_id=tr.track_id,
                    color_name=tr.color_name,
                    confidence=tr.confidence,
                    cluster_id=tr.cluster_id,
                    target_bin=tr.target_bin,
                    cluster_distance=tr.cluster_distance,
                )
            )
        objects.sort(key=lambda o: o.object_id)
        return objects

    def get_visible_unassigned_tracks(self) -> List[Track]:
        return [tr for tr in self.tracks.values() if tr.confirmed and tr.misses == 0 and tr.cluster_id < 0]

    def get_visible_tracks(self) -> List[Track]:
        return [tr for tr in self.tracks.values() if tr.confirmed and tr.misses == 0]

    def get_removed_tracks_waiting_space(self) -> List[Track]:
        ready = []
        for tr in self.tracks.values():
            if not tr.confirmed:
                continue
            if tr.processed:
                continue
            if tr.target_bin not in BIN_TARGET_NAMES:
                continue
            if tr.misses >= REMOVAL_TRIGGER_MISSES:
                ready.append(tr)
        ready.sort(key=lambda t: t.track_id)
        return ready

    def mark_queued(self, track_id: int):
        tr = self.tracks.get(track_id)
        if tr is not None:
            tr.queued = True

    def mark_processed(self, track_id: int):
        tr = self.tracks.get(track_id)
        if tr is not None:
            tr.processed = True
            tr.queued = False

    def update_track_assignment(
        self,
        track_id: int,
        cluster_id: int,
        target_bin: str,
        color_name: str,
        confidence: float,
        cluster_distance: float,
    ):
        tr = self.tracks.get(track_id)
        if tr is None:
            return
        tr.cluster_id = cluster_id
        tr.target_bin = target_bin
        tr.color_name = color_name
        tr.confidence = confidence
        tr.cluster_distance = cluster_distance


# =========================================================
# CLUSTERING — ALWAYS MAP TO 3 BINS
# =========================================================
def _kmeans(data: np.ndarray, k: int):
    if k <= 1:
        centers = np.mean(data, axis=0, keepdims=True).astype(np.float32)
        labels = np.zeros(len(data), dtype=np.int32)
        return labels, centers

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.2)
    _, labels, centers = cv2.kmeans(
        data.astype(np.float32),
        k,
        None,
        criteria,
        5,
        cv2.KMEANS_PP_CENTERS,
    )
    return labels.reshape(-1), centers.astype(np.float32)


def build_cluster_state_from_lab_points(lab_points: List[Tuple[float, float, float]]) -> Optional[ClusterState]:
    if not ENABLE_DYNAMIC_BINS or len(lab_points) < 1:
        return None

    data = np.array(lab_points, dtype=np.float32)
    k = min(3, len(data))

    labels, centers = _kmeans(data, k)
    if labels is None or centers is None or k <= 0:
        return None

    label_to_bin = {}
    cluster_counts = {}
    cluster_names = {}
    cluster_bgr = {}

    for cid in range(k):
        label_to_bin[cid] = BIN_SLOTS[cid]["id"]
        cluster_counts[cid] = int(np.sum(labels == cid))
        cluster_names[cid] = lab_to_display_name(centers[cid])
        cluster_bgr[cid] = lab_to_bgr(centers[cid])

    print(f"[INFO] Clusters: k={k}")
    for cid in range(k):
        print(
            f"       {cluster_letter(cid)}: {cluster_names[cid]}  "
            f"n={cluster_counts.get(cid, 0)}  bin={label_to_bin.get(cid, 'N/A')}"
        )

    return ClusterState(
        k=k,
        centers_lab=centers,
        label_to_bin=label_to_bin,
        cluster_display_names=cluster_names,
        cluster_counts=cluster_counts,
        cluster_bgr=cluster_bgr,
    )


def assign_single_object_to_cluster(obj: DetectedObject, cluster_state: Optional[ClusterState]):
    if cluster_state is None or cluster_state.k <= 0:
        obj.cluster_id = 0
        obj.target_bin = BIN_SLOTS[0]["id"]
        obj.color_name = bgr_to_color_name(obj.mean_bgr)
        obj.confidence = 0.5
        obj.cluster_distance = 0.0
        return

    centers = cluster_state.centers_lab
    p = np.array(obj.mean_lab, dtype=np.float32)
    dists = np.linalg.norm(centers - p[None, :], axis=1)

    cid = int(np.argmin(dists))
    dist = float(dists[cid])

    obj.cluster_id = cid
    obj.cluster_distance = dist
    obj.target_bin = cluster_state.label_to_bin.get(cid, BIN_SLOTS[min(cid, 2)]["id"])
    obj.color_name = cluster_state.cluster_display_names.get(cid, "unknown")
    obj.confidence = float(max(0.0, min(1.0, 1.0 - dist / MAX_LAB_DISTANCE_FOR_CONFIDENCE)))


# =========================================================
# ROBODK HELPERS
# =========================================================
class RoboDKController:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.rdk = None
        self.robot = None
        self.bin_targets: Dict[str, object] = {}
        self.middle_target = None
        self.middle_available = False

        if not self.enabled:
            print("[INFO] RoboDK disabled.")
            return

        self._connect()

    def _connect(self):
        try:
            self.rdk = Robolink()

            self.robot = self.rdk.Item(ROBOT_NAME, ITEM_TYPE_ROBOT)
            if not self.robot.Valid():
                raise RuntimeError(f"Robot '{ROBOT_NAME}' not found in RoboDK.")

            for bin_id, target_name in BIN_TARGET_NAMES.items():
                tgt = self.rdk.Item(target_name, ITEM_TYPE_TARGET)
                if not tgt.Valid():
                    raise RuntimeError(f"Target '{target_name}' not found in RoboDK.")
                self.bin_targets[bin_id] = tgt

            self.middle_target = self.rdk.Item(MIDDLE_TARGET_NAME, ITEM_TYPE_TARGET)
            self.middle_available = self.middle_target.Valid()

            print("[INFO] RoboDK connected.")
            print(f"[INFO] Robot: {ROBOT_NAME}")
            print(f"[INFO] Bin targets: {', '.join(self.bin_targets.keys())}")
            if self.middle_available:
                print(f"[INFO] Middle target: {MIDDLE_TARGET_NAME}")
            else:
                print(f"[WARN] Middle target '{MIDDLE_TARGET_NAME}' not found. Bin motion will still work.")

        except Exception as e:
            self.enabled = False
            self.rdk = None
            self.robot = None
            self.bin_targets = {}
            self.middle_target = None
            self.middle_available = False
            print(f"[WARN] RoboDK connection disabled: {e}")

    def move_to_bin_then_middle(self, bin_id: str, object_id: int) -> bool:
        if not self.enabled:
            print(f"[SIM-OFF] Would move object #{object_id} to {bin_id}")
            if self.middle_available:
                print(f"[SIM-OFF] Would return to {MIDDLE_TARGET_NAME}")
            return True

        if self.robot is None or not self.robot.Valid():
            print("[WARN] Robot not valid. Skipping move.")
            return False

        target = self.bin_targets.get(bin_id)
        if target is None or not target.Valid():
            print(f"[WARN] Target for {bin_id} not valid. Skipping move.")
            return False

        try:
            print(f"[ROBOT] Moving for object #{object_id} -> {bin_id}")
            self.robot.MoveJ(target)

            if self.middle_available and self.middle_target is not None and self.middle_target.Valid():
                print(f"[ROBOT] Returning to {MIDDLE_TARGET_NAME}")
                self.robot.MoveJ(self.middle_target)
            else:
                print(f"[WARN] Middle target unavailable. Staying at {bin_id}")

            return True
        except Exception as e:
            print(f"[ERROR] RoboDK move failed for object #{object_id}: {e}")
            return False


# =========================================================
# JOB LOG HELPERS
# =========================================================
def ensure_job_record(job_log: Dict[int, JobRecord], object_id: int):
    if object_id not in job_log:
        job_log[object_id] = JobRecord(object_id=object_id)


def sync_job_log_from_tracks(job_log: Dict[int, JobRecord], tracker: StableObjectTracker):
    visible_ids = {tr.track_id for tr in tracker.get_visible_tracks()}

    for tr in tracker.tracks.values():
        if not tr.confirmed:
            continue

        ensure_job_record(job_log, tr.track_id)
        rec = job_log[tr.track_id]

        if tr.target_bin != "N/A":
            rec.assigned_bin = tr.target_bin

        if tr.processed:
            rec.status = "DONE"
            if rec.robot_put_to == "-":
                rec.robot_put_to = tr.target_bin
        elif tr.misses >= REMOVAL_TRIGGER_MISSES:
            rec.status = "WAIT_SPACE"
        elif tr.track_id in visible_ids:
            rec.status = "VISIBLE"
        else:
            rec.status = "TRACKING"


def process_spacebar_queue(
    tracker: StableObjectTracker,
    robot: RoboDKController,
    job_log: Dict[int, JobRecord],
):
    candidates = tracker.get_removed_tracks_waiting_space()

    if not candidates:
        print("[QUEUE] No removed objects waiting for SPACE.")
        return

    print(f"[QUEUE] Processing {len(candidates)} removed object(s).")

    for tr in candidates:
        tracker.mark_queued(tr.track_id)
        ensure_job_record(job_log, tr.track_id)
        job_log[tr.track_id].assigned_bin = tr.target_bin
        job_log[tr.track_id].status = "QUEUED"

    for tr in candidates:
        ensure_job_record(job_log, tr.track_id)
        job_log[tr.track_id].status = "MOVING"

        ok = robot.move_to_bin_then_middle(tr.target_bin, tr.track_id)

        if ok:
            tracker.mark_processed(tr.track_id)
            job_log[tr.track_id].robot_put_to = tr.target_bin
            job_log[tr.track_id].status = "DONE"
        else:
            job_log[tr.track_id].status = "MOVE_FAIL"

    print("[QUEUE] Done.")


# =========================================================
# PANEL RENDERING
# =========================================================
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX

_HEADER_H = 30
_BG_PANEL = (20, 20, 20)
_WHITE = (255, 255, 255)
_GRAY = (160, 160, 160)
_DIM = (100, 100, 100)


def _header(img, text: str, color=(200, 200, 200)):
    cv2.rectangle(img, (0, 0), (img.shape[1], _HEADER_H), (50, 50, 50), -1)
    cv2.putText(img, text, (PANEL_PADDING, _HEADER_H - 8), FONT, 0.55, color, 1, cv2.LINE_AA)


def render_panel_a(frame_bgr, objects, cluster_state):
    view = frame_bgr.copy()
    H, W = view.shape[:2]

    for obj in objects:
        cid = obj.cluster_id
        if cluster_state and 0 <= cid < cluster_state.k:
            c = cluster_state.cluster_bgr.get(cid, (200, 200, 200))
        else:
            c = named_bgr(obj.color_name)

        x = obj.cx - obj.w // 2
        y = obj.cy - obj.h // 2

        cv2.rectangle(view, (x, y), (x + obj.w, y + obj.h), c, 2)

        num_str = str(obj.object_id)
        (tw, th), _ = cv2.getTextSize(num_str, FONT_BOLD, 0.60, 2)
        bx1, by1 = obj.cx - tw // 2 - 5, obj.cy - th // 2 - 5
        bx2, by2 = obj.cx + tw // 2 + 5, obj.cy + th // 2 + 5
        cv2.rectangle(view, (bx1, by1), (bx2, by2), c, -1)
        cv2.putText(view, num_str, (bx1 + 4, by2 - 4), FONT_BOLD, 0.60, (0, 0, 0), 2, cv2.LINE_AA)

        if cid >= 0:
            cv2.putText(view, cluster_letter(cid), (x, max(18, y - 5)), FONT_BOLD, 0.65, c, 2, cv2.LINE_AA)

        cv2.putText(view, f"{obj.confidence:.2f}", (x, min(H - 5, y + obj.h + 16)), FONT, 0.42, _GRAY, 1, cv2.LINE_AA)

    _header(view, "A  Live Camera — stable IDs", color=(180, 220, 255))
    cv2.putText(view, f"Objects: {len(objects)}", (W - 110, _HEADER_H - 8), FONT, 0.45, _GRAY, 1)
    return view


def render_panel_b(panel_size, objects, cluster_state):
    ph, pw = panel_size
    img = np.full((ph, pw, 3), _BG_PANEL, dtype=np.uint8)
    _header(img, "B  Object color swatches")

    y = _HEADER_H + PANEL_PADDING
    for obj in objects:
        if y + SWATCH_H > ph - 4:
            break

        cid = obj.cluster_id
        if cluster_state and 0 <= cid < cluster_state.k:
            swatch_bgr = cluster_state.cluster_bgr.get(cid, (200, 200, 200))
        else:
            swatch_bgr = named_bgr(obj.color_name)

        sw = 36
        cv2.rectangle(img, (PANEL_PADDING, y), (PANEL_PADDING + sw, y + SWATCH_H - 4), swatch_bgr, -1)
        cv2.rectangle(img, (PANEL_PADDING, y), (PANEL_PADDING + sw, y + SWATCH_H - 4), _GRAY, 1)

        cv2.putText(img, f"#{obj.object_id}", (PANEL_PADDING + sw + 8, y + 14), FONT_BOLD, 0.55, _WHITE, 1, cv2.LINE_AA)
        cv2.putText(img, f"Cluster {cluster_letter(cid) if cid >= 0 else '?'}", (PANEL_PADDING + sw + 8, y + 28), FONT, 0.42, swatch_bgr, 1, cv2.LINE_AA)

        h, s, v = obj.mean_hsv
        cv2.putText(img, f"HSV ({h:.0f},{s:.0f},{v:.0f}) conf:{obj.confidence:.2f}", (PANEL_PADDING + sw + 90, y + 14), FONT, 0.38, _GRAY, 1, cv2.LINE_AA)
        cv2.putText(img, f"{obj.color_name} -> {obj.target_bin}", (PANEL_PADDING + sw + 90, y + 28), FONT, 0.38, _DIM, 1, cv2.LINE_AA)

        y += SWATCH_H + 2

    return img


def render_panel_c(panel_size, objects, cluster_state):
    ph, pw = panel_size
    img = np.full((ph, pw, 3), _BG_PANEL, dtype=np.uint8)
    _header(img, "C  Cluster membership", color=(200, 255, 200))

    if not cluster_state or cluster_state.k <= 0:
        cv2.putText(img, "No clusters yet", (PANEL_PADDING, _HEADER_H + 30), FONT, 0.55, _DIM, 1)
        return img

    cluster_members: Dict[int, List[int]] = {cid: [] for cid in range(cluster_state.k)}
    for obj in objects:
        if obj.cluster_id >= 0:
            cluster_members[obj.cluster_id].append(obj.object_id)

    y = _HEADER_H + PANEL_PADDING + 10

    for cid in range(cluster_state.k):
        if y + CLUSTER_ROW_H > ph - 4:
            break

        c_bgr = cluster_state.cluster_bgr.get(cid, (200, 200, 200))
        c_name = cluster_state.cluster_display_names.get(cid, "?")
        c_bin = cluster_state.label_to_bin.get(cid, "N/A")
        members = sorted(cluster_members.get(cid, []))
        member_str = ", ".join(f"#{n}" for n in members) if members else "(none)"

        cv2.circle(img, (PANEL_PADDING + 14, y + CLUSTER_ROW_H // 2), 14, c_bgr, -1)
        cv2.putText(img, cluster_letter(cid), (PANEL_PADDING + 8, y + CLUSTER_ROW_H // 2 + 5), FONT_BOLD, 0.60, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(img, f"Cluster {cluster_letter(cid)} — {c_name} [{c_bin}]", (PANEL_PADDING + 36, y + 14), FONT_BOLD, 0.50, _WHITE, 1, cv2.LINE_AA)
        cv2.putText(img, f"  Objects: {member_str}", (PANEL_PADDING + 36, y + 28), FONT, 0.43, c_bgr, 1, cv2.LINE_AA)

        y += CLUSTER_ROW_H + 10

    return img


def render_panel_d(panel_size, job_log: Dict[int, JobRecord]):
    ph, pw = panel_size
    img = np.full((ph, pw, 3), _BG_PANEL, dtype=np.uint8)
    _header(img, "D  Pick / robot log  (SPACE = process queue)", color=(255, 220, 180))

    x1 = PANEL_PADDING
    x2 = 82
    x3 = 235
    x4 = 390

    y = _HEADER_H + PANEL_PADDING + 4

    cv2.putText(img, "Obj", (x1, y + 12), FONT_BOLD, 0.45, _WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, "Assigned Bin", (x2, y + 12), FONT_BOLD, 0.45, _WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, "Robot Put To", (x3, y + 12), FONT_BOLD, 0.45, _WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, "Status", (x4, y + 12), FONT_BOLD, 0.45, _WHITE, 1, cv2.LINE_AA)

    y += 18
    cv2.line(img, (PANEL_PADDING, y), (pw - PANEL_PADDING, y), (80, 80, 80), 1)
    y += 8

    for object_id in sorted(job_log.keys()):
        if y + TABLE_ROW_H > ph - 8:
            break

        rec = job_log[object_id]

        status_color = _GRAY
        if rec.status == "DONE":
            status_color = (120, 220, 120)
        elif rec.status == "WAIT_SPACE":
            status_color = (120, 220, 220)
        elif rec.status == "MOVING":
            status_color = (120, 180, 255)
        elif rec.status == "MOVE_FAIL":
            status_color = (100, 100, 255)

        cv2.putText(img, f"#{rec.object_id}", (x1, y + 14), FONT, 0.43, _WHITE, 1, cv2.LINE_AA)
        cv2.putText(img, rec.assigned_bin, (x2, y + 14), FONT, 0.43, _GRAY, 1, cv2.LINE_AA)
        cv2.putText(img, rec.robot_put_to, (x3, y + 14), FONT, 0.43, _WHITE, 1, cv2.LINE_AA)
        cv2.putText(img, rec.status, (x4, y + 14), FONT, 0.43, status_color, 1, cv2.LINE_AA)

        y += TABLE_ROW_H

    return img


def compose_4panel(frame_bgr, objects, cluster_state, job_log):
    fH, fW = frame_bgr.shape[:2]
    panel_a = render_panel_a(frame_bgr, objects, cluster_state)
    panel_b = render_panel_b((fH, fW), objects, cluster_state)
    panel_c = render_panel_c((fH, fW), objects, cluster_state)
    panel_d = render_panel_d((fH, fW), job_log)

    top = np.hstack([panel_a, panel_b])
    bottom = np.hstack([panel_c, panel_d])
    grid = np.vstack([top, bottom])

    cv2.line(grid, (fW, 0), (fW, grid.shape[0]), (80, 80, 80), 1)
    cv2.line(grid, (0, fH), (grid.shape[1], fH), (80, 80, 80), 1)
    return grid


# =========================================================
# MAIN
# =========================================================
def main():
    system = cam_list = cam = processor = None
    cluster_state: Optional[ClusterState] = None
    frame_idx = 0
    last_log_t = 0.0

    tracker = StableObjectTracker()
    clusters_locked = False
    robot = RoboDKController(enabled=ROBOT_ENABLE)
    job_log: Dict[int, JobRecord] = {}

    try:
        system, cam_list, cam, processor = init_flir_camera()
        print("\nControls:")
        print("  SPACE = process removed objects queue")
        print("  q / ESC = quit\n")

        while True:
            frame = get_bgr_frame(cam, processor)
            if frame is None:
                continue

            frame_idx += 1
            frame = apply_white_balance(frame)

            detections = detect_objects(frame)
            tracker.update(detections)

            if ENABLE_DYNAMIC_BINS:
                if cluster_state is None:
                    seed_tracks = tracker.get_visible_tracks()
                    if len(seed_tracks) >= MIN_OBJECTS_FOR_CLUSTERING:
                        lab_points = [tr.mean_lab for tr in seed_tracks]
                        cluster_state = build_cluster_state_from_lab_points(lab_points)
                        if FREEZE_CLUSTERS_AFTER_INIT and cluster_state is not None:
                            clusters_locked = True

                elif not clusters_locked and frame_idx % CLUSTER_UPDATE_EVERY_N_FRAMES == 0:
                    seed_tracks = tracker.get_visible_tracks()
                    if len(seed_tracks) >= MIN_OBJECTS_FOR_CLUSTERING:
                        lab_points = [tr.mean_lab for tr in seed_tracks]
                        cluster_state = build_cluster_state_from_lab_points(lab_points)

                for tr in tracker.get_visible_unassigned_tracks():
                    temp_obj = DetectedObject(
                        cx=tr.cx,
                        cy=tr.cy,
                        w=tr.w,
                        h=tr.h,
                        contour=tr.contour if tr.contour is not None else np.empty((0, 1, 2), dtype=np.int32),
                        area=tr.area,
                        mean_bgr=tr.mean_bgr,
                        mean_hsv=tr.mean_hsv,
                        mean_lab=tr.mean_lab,
                        object_id=tr.track_id,
                    )
                    assign_single_object_to_cluster(temp_obj, cluster_state)
                    tracker.update_track_assignment(
                        track_id=tr.track_id,
                        cluster_id=temp_obj.cluster_id,
                        target_bin=temp_obj.target_bin,
                        color_name=temp_obj.color_name,
                        confidence=temp_obj.confidence,
                        cluster_distance=temp_obj.cluster_distance,
                    )

            sync_job_log_from_tracks(job_log, tracker)

            tracker.cleanup_stale_tracks()
            objects = tracker.get_visible_objects()

            now = time.time()
            if now - last_log_t >= ACTION_COOLDOWN_S:
                for obj in objects:
                    cl = cluster_letter(obj.cluster_id) if obj.cluster_id >= 0 else "?"
                    h, s, v = obj.mean_hsv
                    L, A, B = obj.mean_lab
                    print(
                        f"[OBJ #{obj.object_id}] cluster={cl}  {obj.color_name}  "
                        f"hsv=({h:.0f},{s:.0f},{v:.0f})  "
                        f"lab=({L:.0f},{A:.0f},{B:.0f})  "
                        f"conf={obj.confidence:.2f}  bin={obj.target_bin}"
                    )
                waiting = [r.object_id for r in job_log.values() if r.status == "WAIT_SPACE"]
                if waiting:
                    print(f"[WAIT_SPACE] Objects waiting for SPACE: {waiting}")
                if objects or waiting:
                    last_log_t = now

            display = compose_4panel(frame, objects, cluster_state, job_log)

            if DISPLAY_SCALE != 1.0:
                display = cv2.resize(
                    display,
                    None,
                    fx=DISPLAY_SCALE,
                    fy=DISPLAY_SCALE,
                    interpolation=cv2.INTER_AREA,
                )

            cv2.imshow("FLIR Color Sorter — SPACEBAR Queue + RoboDK", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                process_spacebar_queue(tracker, robot, job_log)

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