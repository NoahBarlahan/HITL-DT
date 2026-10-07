"""Check a fixture AprilTag against a saved datum from a known camera pose.

First run: return to known camera pose [P], then scan/save correct fixture [S].
Restarts: [K] keep saved datum or [R] redo it. Confirm pose [P], then scan [S].
After a successful scan, errors update live while the camera stays at that pose.
[C] starts another check; [B] cancels a scan; [Q] quits.
The robot must return to the SAME pose, including orientation, for every scan.
Camera return is manually confirmed; robot motion is not controlled here.

Setup and calibration steps:
1. Install: py -3.12 -m pip install -r requirements.txt
2. Set the AprilTag family, ID, and *measured printed* edge size in
   vision/vision_config.py. The edge size is measured between detection corners, not
   the paper edge or white margin.
3. For your Calib.io checkerboard, measure square size (15 mm) and count squares.
4. Run: py -3.12 -m tools.calibration.calibrate_checkerboard --squares ACROSS DOWN
   Use actual counts. Capture 25-40 varied views; Enter solves, S saves.
   ChArUco remains an alternative via tools/calibration/calibrate_charuco.py.
5. Run this script. Confirm reported tag X/Y/Z against a ruler and confirm
   roll/pitch/yaw signs using deliberate, known rotations.
6. Only after that validation, mount the camera on the robot and perform a
   separate hand-eye calibration for TCP-to-camera pose.

Without calibration/camera_calibration.npz this script still detects tag IDs
but cannot save or compare a metric fixture datum.
"""

import cv2
import numpy as np
from vision.camera_source import open_camera
from vision.fixture_datum import FixtureDatumSession
from vision.apriltag_tracking import load_calibration, estimate_tag_pose, tag_center_pixel

from vision.vision_config import (
    APRILTAG_DICTIONARY,
    TARGET_TAG_ID,
)


# Set True to mirror only the displayed image. Detection and pose use the
# original image so the AprilTag's encoded pattern and coordinate system remain
# correct.
MIRROR_IMAGE = False


def mirrored_corners(corners, frame_width):
    """Convert detected corners to their positions in a mirrored display."""
    mirrored = corners.copy()
    mirrored[:, :, 0] = frame_width - 1 - mirrored[:, :, 0]
    return mirrored


def draw_tag_measurements(frame, tag_corners, tag_id, pose):
    """Mark the observed tag; the dashboard reports only errors from the datum."""
    center = tuple(np.rint(tag_center_pixel(tag_corners)).astype(int))
    cv2.circle(frame, center, 5, (0, 255, 0), -1)


def main():
    if not hasattr(cv2, "aruco"):
        print("AprilTag detection requires opencv-contrib-python. Reinstall requirements.txt.")
        return

    calibration = load_calibration()
    if calibration is None:
        print("No calibration file found: datum capture and comparison unavailable.")
    else:
        print(
            "Loaded calibration with RMS error "
            f"{calibration['reprojection_error']:.4f} pixels."
        )

    capture = open_camera()
    if capture is None:
        print("No camera found. Check CAMERA_INDICES and camera permissions.")
        return

    if calibration is not None and calibration["camera_identity"] != capture.camera_identity:
        print("Calibration is for a different/unidentified camera. Metric pose disabled; recalibrate.")
        calibration = None

    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_DICTIONARY)
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    warned_size_mismatch = False
    session = FixtureDatumSession(capture.camera_identity, calibration)

    try:
        while True:
            ok, raw_frame = capture.read()
            if not ok or raw_frame is None:
                print("Could not read a frame from the camera.")
                break

            active_calibration = calibration
            raw_size = (raw_frame.shape[1], raw_frame.shape[0])
            if calibration is not None and calibration["image_size"] != raw_size:
                active_calibration = None
                if not warned_size_mismatch:
                    print(
                        "Calibration image size does not match current stream; "
                        "metric pose disabled. Recalibrate at this resolution."
                    )
                    warned_size_mismatch = True

            corners, tag_ids, _ = detector.detectMarkers(
                cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
            )
            frame = cv2.flip(raw_frame, 1) if MIRROR_IMAGE else raw_frame.copy()
            observations = []

            if tag_ids is None:
                cv2.putText(
                    frame,
                    "No AprilTag detected",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            else:
                display_corners = (
                    [mirrored_corners(tag, frame.shape[1]) for tag in corners]
                    if MIRROR_IMAGE
                    else corners
                )
                cv2.aruco.drawDetectedMarkers(frame, display_corners, tag_ids)
                for raw_corners, shown_corners, tag_id in zip(
                    corners, display_corners, tag_ids.flatten()
                ):
                    tag_id = int(tag_id)
                    if TARGET_TAG_ID is None or tag_id == TARGET_TAG_ID:
                        pose = estimate_tag_pose(raw_corners, active_calibration)
                        observations.append((tag_id, pose))
                        if session.expected_tag_id is None or tag_id == session.expected_tag_id:
                            # Draw only one eligible tag's text to avoid overlapping labels.
                            if len([t for t, _ in observations if session.expected_tag_id is None or t == session.expected_tag_id]) == 1:
                                draw_tag_measurements(frame, shown_corners, tag_id, pose)

            session.observe(observations, active_calibration is not None)
            lines = session.dashboard_lines()
            height, width = frame.shape[:2]
            panel = np.full((height + 30 + 25*len(lines), width, 3), (23, 26, 31), dtype=np.uint8)
            panel[:height] = frame
            colors = dict(heading=(130, 215, 255), normal=(220, 225, 230),
                          warning=(0, 185, 255), good=(100, 235, 125))
            for index, (line, style) in enumerate(lines):
                scale = 0.47
                text_width = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
                scale = min(scale, scale*(width-24)/max(1, text_width))
                cv2.putText(panel, line, (12, height+25+25*index),
                            cv2.FONT_HERSHEY_SIMPLEX, scale, colors[style], 1, cv2.LINE_AA)
            cv2.imshow("AprilTag fixture datum", panel)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            session.handle_key(key)
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
