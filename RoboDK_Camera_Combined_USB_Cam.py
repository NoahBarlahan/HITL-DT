"""
manual_click_calibration_robodk_usb.py
=====================================

Manual test/calibration script for RoboDK + USB camera.

What it does
------------
- Shows live camera feed
- Lets you LEFT-CLICK a point in the image
- Converts that pixel to board X/Y using homography
- Lets you send the robot to:
    - Home
    - Bin 1 / Bin 2 / Bin 3
    - Clicked point at approach height
    - Clicked point at pick height
    - Full approach -> pick -> approach test motion

Keys
----
q / ESC   Quit
h         Move robot Home
1 / 2 / 3 Move robot to BIN_1 / BIN_2 / BIN_3 drop target
g         Move robot to clicked point at APPROACH height only
p         Move robot to clicked point at PICK height
m         Do full approach -> pick -> approach sequence
c         Clear clicked point

Mouse
-----
Left click in the camera window to choose a target point.
"""

import cv2
import numpy as np
from robodk import robolink, robomath


# =========================================================
# SETTINGS
# =========================================================
CAMERA_INDEX = 0
FRAME_W = 1280
FRAME_H = 720

BOARD_FRAME_NAME = "BoardFrame"
PICK_REF_NAME = "PickRef"
HOME_TARGET_NAME = "Home"

H_FILE = r"C:\Users\noah\Desktop\HITL-DT\homography.npy"

BIN_DROP_TARGETS = {
    "BIN_1": "BIN_1_DROP",
    "BIN_2": "BIN_2_DROP",
    "BIN_3": "BIN_3_DROP",
}

Z_PICK_MM = 50.0
Z_APPROACH_MM = 120.0

LINEAR_SPEED = 50.0
JOINT_SPEED = 20.0

WINDOW_NAME = "Manual Click Calibration"


# =========================================================
# CAMERA
# =========================================================
def init_camera(index=CAMERA_INDEX, width=FRAME_W, height=FRAME_H):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam/USB camera at index {index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Camera opened at index {index}")
    print(f"[INFO] Resolution: {actual_w} x {actual_h}")

    for _ in range(10):
        cap.read()

    return cap


def release_camera(cap):
    try:
        if cap is not None:
            cap.release()
    except Exception:
        pass


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


def load_homography():
    try:
        H = np.load(H_FILE)
        if H.shape != (3, 3):
            raise ValueError("Homography is not 3x3")
        print(f"[INFO] Homography loaded: {H_FILE}")
        return H.astype(np.float64)
    except Exception as e:
        raise RuntimeError(f"Could not load homography: {e}")


def px_to_board(H: np.ndarray, cx: float, cy: float):
    pt = np.array([[[cx, cy]]], dtype=np.float32)
    xy = cv2.perspectiveTransform(pt, H)[0, 0]
    return float(xy[0]), float(xy[1])


def pose_to_xyzrxryrz(pose):
    vals = robomath.Pose_2_TxyzRxyz(pose)
    return [float(v) for v in vals]


def make_pose_from_xyzrxryrz(x, y, z, rx, ry, rz):
    return robomath.TxyzRxyz_2_Pose([x, y, z, rx, ry, rz])


def build_click_poses(board_frame, pick_ref, x_mm, y_mm):
    base_T_board = board_frame.PoseAbs()
    pick_ref_abs = pick_ref.PoseAbs()

    ref_vals = pose_to_xyzrxryrz(pick_ref_abs)
    rx, ry, rz = ref_vals[3], ref_vals[4], ref_vals[5]

    pick_abs_xyz_pose = base_T_board * robomath.transl(x_mm, y_mm, Z_PICK_MM)
    app_abs_xyz_pose = base_T_board * robomath.transl(x_mm, y_mm, Z_PICK_MM + Z_APPROACH_MM)

    pick_xyz = pick_abs_xyz_pose.Pos()
    app_xyz = app_abs_xyz_pose.Pos()

    pick_pose = make_pose_from_xyzrxryrz(
        pick_xyz[0], pick_xyz[1], pick_xyz[2], rx, ry, rz
    )
    app_pose = make_pose_from_xyzrxryrz(
        app_xyz[0], app_xyz[1], app_xyz[2], rx, ry, rz
    )

    return app_pose, pick_pose


# =========================================================
# MOUSE CALLBACK
# =========================================================
def on_mouse(event, x, y, flags, state):
    if event == cv2.EVENT_LBUTTONDOWN:
        x_mm, y_mm = px_to_board(state["H"], float(x), float(y))
        state["click_px"] = (x, y)
        state["click_mm"] = (x_mm, y_mm)
        print(f"[CLICK] px=({x}, {y})  ->  board=({x_mm:.1f}, {y_mm:.1f}) mm")


# =========================================================
# DRAWING
# =========================================================
def draw_overlay(frame, state):
    vis = frame.copy()

    # clicked point
    if state["click_px"] is not None:
        x, y = state["click_px"]
        x_mm, y_mm = state["click_mm"]

        cv2.circle(vis, (x, y), 8, (0, 255, 255), -1)
        cv2.circle(vis, (x, y), 20, (0, 255, 255), 2)

        txt1 = f"Clicked px: ({x}, {y})"
        txt2 = f"Board mm: ({x_mm:.1f}, {y_mm:.1f})"
        cv2.putText(vis, txt1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(vis, txt2, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        cv2.putText(
            vis,
            "Left click to select a point",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

    lines = [
        "[H] Home",
        "[1] BIN_1",
        "[2] BIN_2",
        "[3] BIN_3",
        "[G] Go to clicked point (approach height)",
        "[P] Go to clicked point (pick height)",
        "[M] Full approach -> pick -> approach",
        "[C] Clear click",
        "[Q] Quit",
    ]

    y0 = 100
    for i, line in enumerate(lines):
        cv2.putText(
            vis,
            line,
            (20, y0 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

    return vis


# =========================================================
# MAIN
# =========================================================
def main():
    cap = None
    robot = None
    home_target = None

    try:
        rdk = robolink.Robolink()

        robot = rdk.ItemUserPick("Select the robot", robolink.ITEM_TYPE_ROBOT)
        if not robot.Valid():
            raise RuntimeError("No robot selected in RoboDK.")

        robot.setSpeed(LINEAR_SPEED, JOINT_SPEED)

        board_frame = require_item(rdk, BOARD_FRAME_NAME, robolink.ITEM_TYPE_FRAME)
        pick_ref = require_item(rdk, PICK_REF_NAME, robolink.ITEM_TYPE_TARGET)
        home_target = require_item(rdk, HOME_TARGET_NAME, robolink.ITEM_TYPE_TARGET)

        drop_targets = {}
        for bin_name, target_name in BIN_DROP_TARGETS.items():
            item = require_item(rdk, target_name, robolink.ITEM_TYPE_TARGET)
            drop_targets[bin_name] = item
            print(f"[RoboDK] {bin_name} -> {target_name}")

        t_app = get_or_create_target(rdk, "_CLICK_APP")
        t_pick = get_or_create_target(rdk, "_CLICK_PICK")

        H = load_homography()
        cap = init_camera()

        state = {
            "H": H,
            "click_px": None,
            "click_mm": None,
        }

        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, on_mouse, state)

        print("\nControls:")
        print("  Left click = choose point")
        print("  h = home")
        print("  1/2/3 = move to bin targets")
        print("  g = go to clicked point at approach height")
        print("  p = go to clicked point at pick height")
        print("  m = full approach -> pick -> approach")
        print("  c = clear click")
        print("  q / ESC = quit\n")

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                continue

            display = draw_overlay(frame, state)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):
                break

            elif key == ord("h"):
                print("[CMD] Move Home")
                robot.MoveJ(home_target)

            elif key == ord("1"):
                print("[CMD] Move BIN_1")
                robot.MoveJ(drop_targets["BIN_1"])

            elif key == ord("2"):
                print("[CMD] Move BIN_2")
                robot.MoveJ(drop_targets["BIN_2"])

            elif key == ord("3"):
                print("[CMD] Move BIN_3")
                robot.MoveJ(drop_targets["BIN_3"])

            elif key == ord("c"):
                state["click_px"] = None
                state["click_mm"] = None
                print("[CMD] Cleared clicked point")

            elif key in (ord("g"), ord("p"), ord("m")):
                if state["click_mm"] is None:
                    print("[WARN] No clicked point selected yet.")
                    continue

                x_mm, y_mm = state["click_mm"]
                app_abs, pick_abs = build_click_poses(board_frame, pick_ref, x_mm, y_mm)

                t_app.setPoseAbs(app_abs)
                t_pick.setPoseAbs(pick_abs)

                if key == ord("g"):
                    print(f"[CMD] Move to clicked point APPROACH: ({x_mm:.1f}, {y_mm:.1f}) mm")
                    robot.MoveJ(t_app)

                elif key == ord("p"):
                    print(f"[CMD] Move to clicked point PICK: ({x_mm:.1f}, {y_mm:.1f}) mm")
                    robot.MoveJ(t_app)
                    robot.MoveL(t_pick)

                elif key == ord("m"):
                    print(f"[CMD] Full motion to clicked point: ({x_mm:.1f}, {y_mm:.1f}) mm")
                    robot.MoveJ(t_app)
                    robot.MoveL(t_pick)
                    robot.MoveL(t_app)

    except Exception as e:
        print(f"[ERROR] {e}")

    finally:
        try:
            if robot is not None and home_target is not None and robot.Valid():
                robot.MoveJ(home_target)
        except Exception:
            pass

        cv2.destroyAllWindows()
        release_camera(cap)


if __name__ == "__main__":
    main()