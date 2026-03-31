"""
combined_color_robot_robodk_simple_v2.py
========================================
Simplified combined script:
- FLIR Blackfly S camera via PySpin
- Edge-based object detection
- Color classification from object mask
- RoboDK pick and place
- Low-confidence / unknown color human-in-the-loop selection

This version avoids the earlier tracking crash and also avoids the
pose-copy/setPos logic that can trigger RoboDK "list assignment index out of range".

KEYS
----
q / ESC   quit
1 / 2 / 3 select BIN_1 / BIN_2 / BIN_3 during HITL prompt
s         skip current pending object

Camera tuning keys (only when HITL is not active)
a         auto white balance ON
l         lock white balance
r / f     red ratio + / -
b / v     blue ratio + / -
z / x / c / m   Bayer RG / BG / GR / GB
d / k / t / g   daylight / cloudy / tungsten / general
"""

import gc
import cv2
import time
import numpy as np
import PySpin
from dataclasses import dataclass
from typing import Optional, Tuple

from robodk import robolink, robomath


# =========================================================
# SETTINGS — CAMERA / DETECTION
# =========================================================
MIN_OBJECT_AREA = 1200
EDGE_LOW = 60
EDGE_HIGH = 150
BAYER_MODE = "RG"   # RG, BG, GR, GB

RED_RATIO = 1.10
BLUE_RATIO = 1.80
LIGHT_SOURCE = "Daylight5000K"

# =========================================================
# SETTINGS — COLOR TO BIN
# =========================================================
COLOR_TO_BIN = {
    "red": "BIN_1",
    "orange": "BIN_1",
    "yellow": "BIN_1",
    "green": "BIN_2",
    "blue": "BIN_3",
    "purple": "BIN_3",
    "pink": "BIN_3",
    "white": "BIN_2",
    "gray": "BIN_2",
    "black": "BIN_2",
}

CONF_HIGH = 0.55
CONF_MEDIUM = 0.25

# =========================================================
# SETTINGS — ROBODK
# =========================================================
BOARD_FRAME_NAME = "BoardFrame"
PICK_REF_NAME = "PickRef"
HOME_TARGET_NAME = "Home"

H_FILE = (
    r"C:\Users\achivate\OneDrive - Cal Poly\DT\ADEPT\HITL-Digital-Twin"
    r"\DT_Test_Files\homography.npy"
)

BIN_DROP_TARGETS = {
    "BIN_1": "BIN_1_DROP",
    "BIN_2": "BIN_2_DROP",
    "BIN_3": "BIN_3_DROP",
}
DROP_TARGET_FALLBACK = "Place"

Z_PICK_MM = 50.0
Z_APPROACH_MM = 120.0

LINEAR_SPEED_HIGH = 200.0
LINEAR_SPEED_MEDIUM = 100.0
LINEAR_SPEED_LOW = 50.0

JOINT_SPEED_HIGH = 80.0
JOINT_SPEED_MEDIUM = 40.0
JOINT_SPEED_LOW = 20.0

ACTION_COOLDOWN_S = 2.5

BIN_NAMES = {"1": "BIN_1", "2": "BIN_2", "3": "BIN_3"}

BIN_COLORS = {
    "BIN_1": (220, 80, 0),
    "BIN_2": (0, 200, 0),
    "BIN_3": (0, 0, 220),
}

TIER_COLORS = {
    "HIGH": (0, 220, 0),
    "MEDIUM": (0, 200, 255),
    "LOW": (0, 130, 255),
    "UNKNOWN": (0, 0, 220),
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
    color_name: str
    confidence: float
    area: float


# =========================================================
# PYSPIN NODE HELPERS
# =========================================================
def set_enum_node(nodemap, node_name, entry_name):
    try:
        node = PySpin.CEnumerationPtr(nodemap.GetNode(node_name))
        if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
            return False
        entry = node.GetEntryByName(entry_name)
        if not PySpin.IsAvailable(entry) or not PySpin.IsReadable(entry):
            return False
        node.SetIntValue(entry.GetValue())
        return True
    except Exception:
        return False


def get_enum_symbolic(nodemap, node_name):
    try:
        node = PySpin.CEnumerationPtr(nodemap.GetNode(node_name))
        if not PySpin.IsAvailable(node) or not PySpin.IsReadable(node):
            return "Unavailable"
        entry = node.GetCurrentEntry()
        if not PySpin.IsAvailable(entry) or not PySpin.IsReadable(entry):
            return "Unknown"
        return entry.GetSymbolic()
    except Exception:
        return "Unknown"


def set_bool_node(nodemap, node_name, value):
    try:
        node = PySpin.CBooleanPtr(nodemap.GetNode(node_name))
        if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
            return False
        node.SetValue(bool(value))
        return True
    except Exception:
        return False


def set_balance_ratio(nodemap, selector_name, ratio):
    try:
        if not set_enum_node(nodemap, "BalanceRatioSelector", selector_name):
            return False
        node = PySpin.CFloatPtr(nodemap.GetNode("BalanceRatio"))
        if not PySpin.IsAvailable(node) or not PySpin.IsWritable(node):
            return False
        ratio = max(node.GetMin(), min(node.GetMax(), ratio))
        node.SetValue(ratio)
        return True
    except Exception:
        return False


def get_balance_ratio(nodemap, selector_name):
    try:
        if not set_enum_node(nodemap, "BalanceRatioSelector", selector_name):
            return None
        node = PySpin.CFloatPtr(nodemap.GetNode("BalanceRatio"))
        if not PySpin.IsAvailable(node) or not PySpin.IsReadable(node):
            return None
        return float(node.GetValue())
    except Exception:
        return None


# =========================================================
# CAMERA SETUP
# =========================================================
def try_set_pixel_format(nodemap):
    for fmt in ["BGR8", "RGB8", "BayerRG8", "BayerBG8", "BayerGR8", "BayerGB8", "Mono8"]:
        if set_enum_node(nodemap, "PixelFormat", fmt):
            return fmt
    return get_enum_symbolic(nodemap, "PixelFormat")


def configure_camera(nodemap):
    if not set_enum_node(nodemap, "AcquisitionMode", "Continuous"):
        raise RuntimeError("Could not set AcquisitionMode to Continuous.")

    set_enum_node(nodemap, "ExposureAuto", "Off")
    set_enum_node(nodemap, "GainAuto", "Off")
    set_enum_node(nodemap, "BalanceWhiteAuto", "Off")

    set_bool_node(nodemap, "IspEnable", True)
    set_enum_node(nodemap, "ColorTransformationSelector", "RGBtoRGB")
    set_bool_node(nodemap, "ColorTransformationEnable", True)
    set_enum_node(nodemap, "RgbTransformLightSource", LIGHT_SOURCE)

    requested_pf = try_set_pixel_format(nodemap)
    active_pf = get_enum_symbolic(nodemap, "PixelFormat")
    return requested_pf, active_pf


def init_camera():
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()

    if cam_list.GetSize() == 0:
        cam_list.Clear()
        system.ReleaseInstance()
        raise RuntimeError("No camera detected.")

    cam = cam_list[0]
    cam.Init()
    nodemap = cam.GetNodeMap()

    requested_pf, active_pf = configure_camera(nodemap)
    cam.BeginAcquisition()

    print(f"[INFO] Requested PixelFormat: {requested_pf}")
    print(f"[INFO] Active PixelFormat:    {active_pf}")
    print(f"[INFO] WhiteBalanceAuto:      {get_enum_symbolic(nodemap, 'BalanceWhiteAuto')}")
    print(f"[INFO] Light Source:          {get_enum_symbolic(nodemap, 'RgbTransformLightSource')}")

    return system, cam_list, cam, nodemap, active_pf


def release_camera(system, cam_list, cam):
    try:
        if cam is not None:
            try:
                cam.EndAcquisition()
            except Exception:
                pass
            try:
                cam.DeInit()
            except Exception:
                pass
    finally:
        gc.collect()
        try:
            if cam_list is not None:
                cam_list.Clear()
        except Exception:
            pass
        gc.collect()
        try:
            if system is not None:
                system.ReleaseInstance()
        except Exception as e:
            print(f"[WARN] ReleaseInstance error: {e}")


# =========================================================
# IMAGE CONVERSION
# =========================================================
def bayer_to_bgr(frame_2d, mode):
    if mode == "RG":
        return cv2.cvtColor(frame_2d, cv2.COLOR_BAYER_RG2BGR)
    if mode == "BG":
        return cv2.cvtColor(frame_2d, cv2.COLOR_BAYER_BG2BGR)
    if mode == "GR":
        return cv2.cvtColor(frame_2d, cv2.COLOR_BAYER_GR2BGR)
    if mode == "GB":
        return cv2.cvtColor(frame_2d, cv2.COLOR_BAYER_GB2BGR)
    return cv2.cvtColor(frame_2d, cv2.COLOR_BAYER_RG2BGR)


def get_bgr_frame(cam, active_pixel_format):
    image_result = None
    try:
        image_result = cam.GetNextImage(1000)

        if image_result.IsIncomplete():
            print(f"[WARN] Incomplete image: {image_result.GetImageStatus()}")
            return None

        frame = image_result.GetNDArray()
        if frame is None:
            return None

        if len(frame.shape) == 3:
            if frame.shape[2] == 3:
                if active_pixel_format == "RGB8":
                    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                return frame.copy()

            if frame.shape[2] == 4:
                return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

        if len(frame.shape) == 2:
            if active_pixel_format == "Mono8":
                return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            return bayer_to_bgr(frame, BAYER_MODE)

        return None

    finally:
        if image_result is not None:
            try:
                image_result.Release()
            except Exception:
                pass


# =========================================================
# COLOR CLASSIFICATION
# =========================================================
def color_to_bgr(name):
    table = {
        "red": (0, 0, 255),
        "orange": (0, 165, 255),
        "yellow": (0, 255, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "purple": (255, 0, 255),
        "pink": (203, 192, 255),
        "white": (255, 255, 255),
        "gray": (170, 170, 170),
        "black": (80, 80, 80),
        "unknown": (255, 255, 255),
    }
    return table.get(name, (255, 255, 255))


def classify_color_from_mask(hsv, obj_mask):
    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    neutral_mask = obj_mask > 0
    if np.count_nonzero(neutral_mask) == 0:
        return "unknown", 0.0, {}

    s_vals = s[neutral_mask]
    v_vals = v[neutral_mask]

    s_mean = float(np.mean(s_vals))
    v_mean = float(np.mean(v_vals))

    total_pixels = max(1, int(np.count_nonzero(obj_mask)))

    if v_mean < 50:
        return "black", 1.0, {"black": total_pixels}
    if s_mean < 35 and v_mean > 185:
        return "white", 1.0, {"white": total_pixels}
    if s_mean < 35:
        return "gray", 1.0, {"gray": total_pixels}

    red1 = cv2.inRange(hsv, np.array([0, 70, 50]), np.array([10, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 70, 50]), np.array([179, 255, 255]))
    red = cv2.bitwise_or(red1, red2)

    orange = cv2.inRange(hsv, np.array([11, 70, 50]), np.array([20, 255, 255]))
    yellow = cv2.inRange(hsv, np.array([21, 70, 50]), np.array([35, 255, 255]))
    green = cv2.inRange(hsv, np.array([36, 60, 40]), np.array([85, 255, 255]))
    blue = cv2.inRange(hsv, np.array([86, 60, 40]), np.array([130, 255, 255]))
    purple = cv2.inRange(hsv, np.array([131, 60, 40]), np.array([160, 255, 255]))
    pink = cv2.inRange(hsv, np.array([161, 50, 80]), np.array([169, 255, 255]))

    candidates = {
        "red": red,
        "orange": orange,
        "yellow": yellow,
        "green": green,
        "blue": blue,
        "purple": purple,
        "pink": pink,
    }

    scores = {}
    for name, color_mask in candidates.items():
        overlap = cv2.bitwise_and(color_mask, color_mask, mask=obj_mask)
        scores[name] = int(np.count_nonzero(overlap))

    best_name = max(scores, key=scores.get)
    best_score = scores[best_name]
    best_fraction = best_score / total_pixels

    if best_fraction < 0.08:
        return "unknown", best_fraction, scores

    return best_name, best_fraction, scores


def confidence_tier(color_name: str, confidence: float) -> str:
    if color_name == "unknown":
        return "UNKNOWN"
    if confidence >= CONF_HIGH:
        return "HIGH"
    if confidence >= CONF_MEDIUM:
        return "MEDIUM"
    return "LOW"


# =========================================================
# DETECTION
# =========================================================
def detect_objects_and_colors(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, EDGE_LOW, EDGE_HIGH)

    kernel = np.ones((3, 3), np.uint8)
    edge_map = cv2.dilate(edges, kernel, iterations=1)
    edge_map = cv2.morphologyEx(edge_map, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(edge_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    objects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_OBJECT_AREA:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        cx = x + w // 2
        cy = y + h // 2

        obj_mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(obj_mask, [cnt], -1, 255, thickness=-1)
        obj_mask = cv2.erode(obj_mask, kernel, iterations=2)

        if np.count_nonzero(obj_mask) < 100:
            continue

        color_name, conf, _ = classify_color_from_mask(hsv, obj_mask)

        objects.append(
            DetectedObject(
                cx=cx,
                cy=cy,
                w=w,
                h=h,
                contour=cnt,
                color_name=color_name,
                confidence=conf,
                area=area,
            )
        )

    return edges, objects


# =========================================================
# ROBODK HELPERS
# =========================================================
def require_item(rdk, name: str, item_type: int):
    item = rdk.Item(name, item_type)
    if not item.Valid():
        raise RuntimeError(f"RoboDK item '{name}' not found.")
    return item


def get_or_create_target(rdk, name: str):
    t = rdk.Item(name, robolink.ITEM_TYPE_TARGET)
    if not t.Valid():
        t = rdk.AddTarget(name)
    return t


def load_homography() -> np.ndarray:
    try:
        H = np.load(H_FILE)
        if H.shape != (3, 3):
            raise ValueError("Homography is not 3x3")
        print(f"[INFO] Homography loaded: {H_FILE}")
        return H.astype(np.float64)
    except Exception as e:
        print(f"[WARN] Homography not loaded ({e}) — using identity.")
        return np.eye(3, dtype=np.float64)


def px_to_board(H: np.ndarray, cx: float, cy: float) -> Tuple[float, float]:
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    xy = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(xy[0]), float(xy[1])


def make_pose_from_xyzrxryrz(x, y, z, rx, ry, rz):
    return robomath.TxyzRxyz_2_Pose([x, y, z, rx, ry, rz])


def pose_to_xyzrxryrz(pose):
    vals = robomath.Pose_2_TxyzRxyz(pose)
    return [float(v) for v in vals]


def build_pick_poses_from_pickref(board_frame, pick_ref, x_mm: float, y_mm: float):
    """
    Create absolute approach/pick poses by:
    1) reading PickRef absolute pose
    2) extracting its orientation as xyzrxryrz
    3) replacing only xyz using BoardFrame-relative board coords

    This avoids using clone/setPos, which caused the RoboDK list error.
    """
    base_T_board = board_frame.PoseAbs()
    pick_ref_abs = pick_ref.PoseAbs()

    ref_vals = pose_to_xyzrxryrz(pick_ref_abs)
    rx, ry, rz = ref_vals[3], ref_vals[4], ref_vals[5]

    pick_board_abs = base_T_board * robomath.transl(x_mm, y_mm, Z_PICK_MM)
    app_board_abs = base_T_board * robomath.transl(x_mm, y_mm, Z_PICK_MM + Z_APPROACH_MM)

    pick_xyz = pick_board_abs.Pos()
    app_xyz = app_board_abs.Pos()

    pick_pose = make_pose_from_xyzrxryrz(pick_xyz[0], pick_xyz[1], pick_xyz[2], rx, ry, rz)
    app_pose = make_pose_from_xyzrxryrz(app_xyz[0], app_xyz[1], app_xyz[2], rx, ry, rz)

    return app_pose, pick_pose


def set_robot_speed(robot, tier: str):
    if tier == "HIGH":
        robot.setSpeed(LINEAR_SPEED_HIGH, JOINT_SPEED_HIGH)
    elif tier == "MEDIUM":
        robot.setSpeed(LINEAR_SPEED_MEDIUM, JOINT_SPEED_MEDIUM)
    else:
        robot.setSpeed(LINEAR_SPEED_LOW, JOINT_SPEED_LOW)


def move_robot_home(robot, home_target, log_prefix: str = "") -> str:
    try:
        robot.MoveJ(home_target)
        if log_prefix:
            print(f"{log_prefix} Robot moved home.")
        return "MoveJ_home"
    except Exception as e:
        print(f"[RoboDK] Home move error: {e}")
        return "home_error"


def execute_motion(robot, t_app, t_pick, drop_target, tier: str, log_prefix: str = "") -> str:
    try:
        if tier == "HIGH":
            motion_type = "MoveJ_fast"
            set_robot_speed(robot, "HIGH")
            robot.MoveJ(t_app)
            robot.MoveL(t_pick)
            robot.MoveL(t_app)
            robot.MoveJ(drop_target)
            print(f"{log_prefix} [HIGH] Fast move -> drop.")

        elif tier == "MEDIUM":
            motion_type = "MoveL_medium"
            set_robot_speed(robot, "MEDIUM")
            robot.MoveL(t_app)
            robot.MoveL(t_pick)
            robot.MoveL(t_app)
            robot.MoveL(drop_target)
            print(f"{log_prefix} [MEDIUM] Medium move -> drop.")

        else:
            motion_type = "MoveL_slow"
            set_robot_speed(robot, "LOW")
            robot.MoveL(t_app)
            time.sleep(0.5)
            robot.MoveL(t_pick)
            robot.MoveL(t_app)
            robot.MoveL(drop_target)
            print(f"{log_prefix} [LOW] Slow move -> drop.")

        return motion_type

    except Exception as e:
        print(f"[RoboDK] Motion error: {e}")
        return "error"


# =========================================================
# DISPLAY
# =========================================================
def draw_hitl_overlay(vis, obj: DetectedObject):
    overlay = vis.copy()
    h, w = vis.shape[:2]
    cv2.rectangle(overlay, (50, 70), (w - 50, h - 40), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.83, vis, 0.17, 0, vis)

    tier = confidence_tier(obj.color_name, obj.confidence)
    tier_color = TIER_COLORS.get(tier, (255, 255, 255))

    cv2.putText(vis, "HUMAN INPUT NEEDED", (80, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 100, 255), 2)

    cv2.putText(vis, f"Detected color: {obj.color_name}", (80, 170),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.putText(vis, f"Confidence: {obj.confidence*100:.1f}%", (80, 205),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1)

    cv2.putText(vis, f"Tier: {tier}", (80, 240),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, tier_color, 2)

    cv2.putText(vis, "Choose destination bin:", (80, 280),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)

    btn_y = 330
    for key, bin_name in BIN_NAMES.items():
        bgr = BIN_COLORS[bin_name]
        bx = 80 + (int(key) - 1) * 160
        cv2.rectangle(vis, (bx, btn_y), (bx + 140, btn_y + 50), bgr, -1)
        cv2.putText(vis, f"[{key}] {bin_name}", (bx + 8, btn_y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2)

    cv2.putText(vis, "[S] Skip", (80, btn_y + 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 150, 150), 1)


def make_display(frame_bgr, edges, current_obj: Optional[DetectedObject], nodemap, hitl_obj=None):
    left = frame_bgr.copy()
    right = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    if current_obj is not None:
        x = current_obj.cx - current_obj.w // 2
        y = current_obj.cy - current_obj.h // 2
        c = color_to_bgr(current_obj.color_name)
        tier = confidence_tier(current_obj.color_name, current_obj.confidence)

        label = f"{current_obj.color_name} ({current_obj.confidence:.2f}) [{tier}]"
        cv2.rectangle(left, (x, y), (x + current_obj.w, y + current_obj.h), c, 2)
        cv2.putText(left, label, (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, c, 2)

        cv2.rectangle(right, (x, y), (x + current_obj.w, y + current_obj.h), c, 2)
        cv2.putText(right, label, (x, max(20, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, c, 2)

    cv2.putText(left, "Camera", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    cv2.putText(right, "Edges", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    red_ratio = get_balance_ratio(nodemap, "Red")
    blue_ratio = get_balance_ratio(nodemap, "Blue")
    wb_mode = get_enum_symbolic(nodemap, "BalanceWhiteAuto")

    combined = np.hstack((left, right))
    info = (
        f"WB={wb_mode}  Red={red_ratio if red_ratio is not None else 'N/A'}  "
        f"Blue={blue_ratio if blue_ratio is not None else 'N/A'}  "
        f"Bayer={BAYER_MODE}"
    )
    cv2.putText(combined, info, (10, combined.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    if hitl_obj is not None:
        draw_hitl_overlay(combined, hitl_obj)

    return combined


# =========================================================
# MAIN
# =========================================================
def main():
    global RED_RATIO, BLUE_RATIO, BAYER_MODE

    system = None
    cam_list = None
    cam = None
    nodemap = None
    active_pixel_format = "Unknown"

    robot = None
    home_target = None

    try:
        rdk = robolink.Robolink()

        robot = rdk.ItemUserPick("Select the robot", robolink.ITEM_TYPE_ROBOT)
        if not robot.Valid():
            raise RuntimeError("No robot selected in RoboDK.")

        board_frame = require_item(rdk, BOARD_FRAME_NAME, robolink.ITEM_TYPE_FRAME)
        pick_ref = require_item(rdk, PICK_REF_NAME, robolink.ITEM_TYPE_TARGET)
        home_target = require_item(rdk, HOME_TARGET_NAME, robolink.ITEM_TYPE_TARGET)

        t_app = get_or_create_target(rdk, "_PICK_APP")
        t_pick = get_or_create_target(rdk, "_PICK")

        drop_targets = {}
        for bin_name, target_name in BIN_DROP_TARGETS.items():
            item = rdk.Item(target_name, robolink.ITEM_TYPE_TARGET)

            if not item.Valid():
                print(
                    f"[WARN] Drop target '{target_name}' not found for {bin_name}. "
                    f"Trying fallback '{DROP_TARGET_FALLBACK}'."
                )
                item = rdk.Item(DROP_TARGET_FALLBACK, robolink.ITEM_TYPE_TARGET)

            if not item.Valid():
                raise RuntimeError(
                    f"No valid drop target found for {bin_name}. "
                    f"Create '{target_name}' or '{DROP_TARGET_FALLBACK}' in RoboDK."
                )

            drop_targets[bin_name] = item
            print(f"[RoboDK] {bin_name} -> drop target '{item.Name()}'")

        H = load_homography()
        move_robot_home(robot, home_target, "[INIT]")

        system, cam_list, cam, nodemap, active_pixel_format = init_camera()
        set_enum_node(nodemap, "BalanceWhiteAuto", "Off")
        set_balance_ratio(nodemap, "Red", RED_RATIO)
        set_balance_ratio(nodemap, "Blue", BLUE_RATIO)

        print("\nControls:")
        print("  q / ESC = quit")
        print("  1 / 2 / 3 = select bin during HITL prompt")
        print("  s = skip pending object")
        print("  a = auto white balance ON")
        print("  l = lock white balance")
        print("  r/f = red ratio +/-")
        print("  b/v = blue ratio +/-")
        print("  z/x/c/m = Bayer RG/BG/GR/GB")
        print("  d/k/t/g = daylight/cloudy/tungsten/general")
        print()

        last_action_t = 0.0
        hitl_active = False
        pending_obj: Optional[DetectedObject] = None

        while True:
            frame = get_bgr_frame(cam, active_pixel_format)
            if frame is None:
                continue

            edges, detections = detect_objects_and_colors(frame)

            current_obj = None
            if detections:
                detections.sort(key=lambda d: d.area, reverse=True)
                current_obj = detections[0]

            now = time.time()

            if current_obj is not None:
                tier = confidence_tier(current_obj.color_name, current_obj.confidence)
                target_bin = COLOR_TO_BIN.get(current_obj.color_name)

                cv2.putText(
                    frame,
                    f"Detected: {current_obj.color_name}  conf={current_obj.confidence:.2f}  tier={tier}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                )

                if (not hitl_active) and (now - last_action_t >= ACTION_COOLDOWN_S):
                    x_mm, y_mm = px_to_board(H, float(current_obj.cx), float(current_obj.cy))

                    print(
                        f"[OBJ] color={current_obj.color_name} "
                        f"conf={current_obj.confidence:.2f} "
                        f"tier={tier} "
                        f"bin={target_bin if target_bin else 'N/A'} "
                        f"px=({current_obj.cx},{current_obj.cy}) "
                        f"mm=({x_mm:.1f},{y_mm:.1f})"
                    )

                    if tier in ("HIGH", "MEDIUM") and target_bin is not None:
                        try:
                            print("[DBG] building poses...")
                            app_abs, pick_abs = build_pick_poses_from_pickref(
                                board_frame, pick_ref, x_mm, y_mm
                            )

                            print("[DBG] setting target poses...")
                            t_app.setPoseAbs(app_abs)
                            t_pick.setPoseAbs(pick_abs)

                            print("[DBG] executing motion...")
                            motion_type = execute_motion(
                                robot,
                                t_app,
                                t_pick,
                                drop_targets[target_bin],
                                tier,
                                log_prefix="[AUTO]",
                            )
                            print(f"[AUTO] motion={motion_type}")
                            last_action_t = time.time()

                        except Exception as motion_setup_error:
                            print(f"[ERROR] Motion setup failed: {motion_setup_error}")
                            move_robot_home(robot, home_target, "[SAFE]")
                            last_action_t = time.time()

                    else:
                        hitl_active = True
                        pending_obj = current_obj
                        move_robot_home(robot, home_target, "[HITL]")
                        print("[HITL] Waiting for operator selection...")

            display = make_display(frame, edges, current_obj, nodemap, pending_obj if hitl_active else None)
            cv2.imshow("Combined Color Detection + RoboDK", display)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            if hitl_active and key in (ord("1"), ord("2"), ord("3"), ord("s"), ord("S")):
                if pending_obj is None:
                    hitl_active = False
                    continue

                if key in (ord("s"), ord("S")):
                    print("[HITL] Skipped object.")
                    last_action_t = time.time()
                else:
                    final_bin = BIN_NAMES[chr(key)]

                    try:
                        x_mm, y_mm = px_to_board(H, float(pending_obj.cx), float(pending_obj.cy))
                        print("[DBG] building HITL poses...")
                        app_abs, pick_abs = build_pick_poses_from_pickref(
                            board_frame, pick_ref, x_mm, y_mm
                        )

                        print("[DBG] setting HITL target poses...")
                        t_app.setPoseAbs(app_abs)
                        t_pick.setPoseAbs(pick_abs)

                        print("[DBG] executing HITL motion...")
                        motion_type = execute_motion(
                            robot,
                            t_app,
                            t_pick,
                            drop_targets[final_bin],
                            "LOW",
                            log_prefix="[HITL]",
                        )
                        print(f"[HITL] final_bin={final_bin}, motion={motion_type}")
                        last_action_t = time.time()

                    except Exception as hitl_motion_error:
                        print(f"[ERROR] HITL motion setup failed: {hitl_motion_error}")
                        move_robot_home(robot, home_target, "[SAFE]")
                        last_action_t = time.time()

                pending_obj = None
                hitl_active = False
                continue

            if not hitl_active:
                if key == ord("a"):
                    set_enum_node(nodemap, "BalanceWhiteAuto", "Continuous")
                    print("[INFO] Auto white balance ON")

                elif key == ord("l"):
                    set_enum_node(nodemap, "BalanceWhiteAuto", "Off")
                    rr = get_balance_ratio(nodemap, "Red")
                    br = get_balance_ratio(nodemap, "Blue")
                    if rr is not None:
                        RED_RATIO = rr
                    if br is not None:
                        BLUE_RATIO = br
                    print(f"[INFO] White balance locked. Red={RED_RATIO:.3f}, Blue={BLUE_RATIO:.3f}")

                elif key == ord("r"):
                    RED_RATIO += 0.05
                    set_enum_node(nodemap, "BalanceWhiteAuto", "Off")
                    set_balance_ratio(nodemap, "Red", RED_RATIO)
                    print(f"[INFO] Red ratio = {RED_RATIO:.3f}")

                elif key == ord("f"):
                    RED_RATIO -= 0.05
                    set_enum_node(nodemap, "BalanceWhiteAuto", "Off")
                    set_balance_ratio(nodemap, "Red", RED_RATIO)
                    print(f"[INFO] Red ratio = {RED_RATIO:.3f}")

                elif key == ord("b"):
                    BLUE_RATIO += 0.05
                    set_enum_node(nodemap, "BalanceWhiteAuto", "Off")
                    set_balance_ratio(nodemap, "Blue", BLUE_RATIO)
                    print(f"[INFO] Blue ratio = {BLUE_RATIO:.3f}")

                elif key == ord("v"):
                    BLUE_RATIO -= 0.05
                    set_enum_node(nodemap, "BalanceWhiteAuto", "Off")
                    set_balance_ratio(nodemap, "Blue", BLUE_RATIO)
                    print(f"[INFO] Blue ratio = {BLUE_RATIO:.3f}")

                elif key == ord("z"):
                    BAYER_MODE = "RG"
                    print("[INFO] Bayer mode = RG")

                elif key == ord("x"):
                    BAYER_MODE = "BG"
                    print("[INFO] Bayer mode = BG")

                elif key == ord("c"):
                    BAYER_MODE = "GR"
                    print("[INFO] Bayer mode = GR")

                elif key == ord("m"):
                    BAYER_MODE = "GB"
                    print("[INFO] Bayer mode = GB")

                elif key == ord("d"):
                    set_enum_node(nodemap, "RgbTransformLightSource", "Daylight5000K")
                    print("[INFO] Light source = Daylight5000K")

                elif key == ord("k"):
                    set_enum_node(nodemap, "RgbTransformLightSource", "Cloudy6500K")
                    print("[INFO] Light source = Cloudy6500K")

                elif key == ord("t"):
                    set_enum_node(nodemap, "RgbTransformLightSource", "Tungsten2800K")
                    print("[INFO] Light source = Tungsten2800K")

                elif key == ord("g"):
                    set_enum_node(nodemap, "RgbTransformLightSource", "General")
                    print("[INFO] Light source = General")

    except Exception as e:
        print(f"[ERROR] {e}")

    finally:
        try:
            if robot is not None and home_target is not None and robot.Valid():
                move_robot_home(robot, home_target, "[EXIT]")
        except Exception:
            pass

        cv2.destroyAllWindows()
        release_camera(system, cam_list, cam)


if __name__ == "__main__":
    main()