"""Return the Lite6 to the saved AprilTag scan pose.

Paste into UFACTORY Studio's Python IDE, save, then Run.
Running this file enables motion and commands one joint move.
The path from the current pose must be clear; this is not obstacle planning.
Standalone use: py -3.12 -m tools.robot.return_to_scan_position
"""

import time
from xarm.wrapper import XArmAPI

ROBOT_IP = "192.168.1.160"
SCAN_JOINTS_DEG = [0.0, -6.4, 85.8, 0.0, 92.2, 0.0]
JOINT_SPEED_DEG_S = 5.0
JOINT_ACCEL_DEG_S2 = 20.0


def check(code, operation):
    if code != 0:
        raise RuntimeError("{} failed (SDK code {}).".format(operation, code))


def main():
    arm = XArmAPI(ROBOT_IP, is_radian=False)
    move_started = False
    try:
        code, faults = arm.get_err_warn_code()
        check(code, "Read robot faults")
        if any(faults):
            raise RuntimeError("Resolve robot error/warning {} in Studio first.".format(faults))
        check(arm.motion_enable(enable=True), "Enable motors")
        check(arm.set_mode(0), "Select position mode")
        check(arm.set_state(0), "Enable motion")
        time.sleep(1.0)

        print("Moving to scan joints (degrees):", SCAN_JOINTS_DEG)
        move_started = True
        check(arm.set_servo_angle(
            angle=SCAN_JOINTS_DEG,
            speed=JOINT_SPEED_DEG_S,
            mvacc=JOINT_ACCEL_DEG_S2,
            is_radian=False,
            relative=False,
            radius=-1,
            wait=True,
            timeout=120,
        ), "Return to scan pose")
        time.sleep(0.5)
        code, actual = arm.get_servo_angle(is_radian=False)
        check(code, "Read final joints")
        if len(actual) < 6 or any(abs(a-b) > 0.25 for a, b in zip(actual[:6], SCAN_JOINTS_DEG)):
            raise RuntimeError("Final joints differ from scan target: {}".format(actual))
        print("Scan pose reached. In AprilTag app press P, then S.")
    except BaseException:
        if move_started:
            arm.set_state(4)  # Stop an interrupted or failed move.
        raise
    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()
