"""Calibrate camera intrinsics from varied ChArUco-board views.

Calibration steps:
1. Install packages: py -3.12 -m pip install -r requirements.txt
2. Create the board: py -3.12 -m tools.calibration.generate_charuco_board
3. Print it at 100% scale, mount it flat, and verify the physical square size.
4. Set the measured square and marker sizes in vision/vision_config.py.
5. Use the camera's final resolution, focus, and lens configuration.
6. Run this script and press C to capture 25-40 GOOD, varied views. Include
   center/edges/corners of the image, near/far views, and multiple tilts.
   Do not collect a burst of nearly identical images.
7. Press Enter to calibrate after at least 15 captures. At 30 captures this
   script automatically calibrates. Press R to discard captured views; Q quits.
8. The result is saved to calibration/camera_calibration.npz. Keep it only for
   this camera, resolution, focus, and lens setup. Recalibrate after changes.
9. Run CV-AprilTagModel.py to verify AprilTag X/Y/Z and orientation against a
   ruler and known rotations before relying on the result.
"""

from datetime import datetime

import cv2
import numpy as np
from vision.camera_source import open_camera

from vision.vision_config import (
    CALIBRATION_FILE,
    CAMERA_INDICES,
    CHARUCO_DICTIONARY,
    CHARUCO_MARKER_LENGTH_MM,
    CHARUCO_SQUARE_LENGTH_MM,
    CHARUCO_SQUARES_X,
    CHARUCO_SQUARES_Y,
)


MIN_CHARUCO_CORNERS = 12
MIN_CAPTURE_COUNT = 15
TARGET_CAPTURE_COUNT = 30


def make_board_and_detector():
    dictionary = cv2.aruco.getPredefinedDictionary(CHARUCO_DICTIONARY)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        CHARUCO_SQUARE_LENGTH_MM,
        CHARUCO_MARKER_LENGTH_MM,
        dictionary,
    )
    return board, cv2.aruco.CharucoDetector(board)


def save_calibration(camera_matrix, distortion, image_size, reprojection_error, count, camera_identity):
    """Save calibration data and its capture context for later validation."""
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        CALIBRATION_FILE,
        camera_matrix=camera_matrix,
        distortion_coefficients=distortion,
        image_width=image_size[0],
        image_height=image_size[1],
        reprojection_error=reprojection_error,
        capture_count=count,
        camera_identity=camera_identity,
        charuco_square_length_mm=CHARUCO_SQUARE_LENGTH_MM,
        charuco_marker_length_mm=CHARUCO_MARKER_LENGTH_MM,
        created_at=datetime.now().astimezone().isoformat(),
    )
    print(f"Saved calibration to {CALIBRATION_FILE}")


def calibrate(captured_object_points, captured_image_points, image_size):
    """Solve intrinsics from ChArUco point correspondences."""
    error, camera_matrix, distortion, _, _ = cv2.calibrateCamera(
        captured_object_points,
        captured_image_points,
        image_size,
        None,
        None,
    )
    return error, camera_matrix, distortion


def main():
    if not hasattr(cv2, "aruco"):
        print("ChArUco calibration requires opencv-contrib-python. Reinstall requirements.txt.")
        return

    capture = open_camera()
    if capture is None:
        print("No camera found. Check CAMERA_INDICES and camera permissions.")
        return

    board, detector = make_board_and_detector()
    board_points = board.getChessboardCorners()
    captured_object_points = []
    captured_image_points = []
    image_size = None

    print("Press C to capture a varied board view; Enter to calibrate; R to reset; Q to quit.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("Could not read a frame from the camera.")
                break

            image_size = (frame.shape[1], frame.shape[0])
            charuco_corners, charuco_ids, _, _ = detector.detectBoard(frame)
            display = frame.copy()
            corner_count = 0 if charuco_ids is None else len(charuco_ids)
            if charuco_ids is not None:
                cv2.aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)

            cv2.putText(
                display,
                f"Captured: {len(captured_object_points)}/{TARGET_CAPTURE_COUNT} | corners: {corner_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0) if corner_count >= MIN_CHARUCO_CORNERS else (0, 0, 255),
                2,
            )
            cv2.putText(
                display,
                "C capture | Enter calibrate | R reset | Q quit",
                (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
            )
            cv2.imshow("ChArUco calibration", display)

            key = cv2.waitKey(1) & 0xFF
            wants_capture = key == ord("c")
            wants_calibration = key in (13, 10)

            if wants_capture:
                if charuco_ids is None or corner_count < MIN_CHARUCO_CORNERS:
                    print(f"Need at least {MIN_CHARUCO_CORNERS} ChArUco corners; found {corner_count}.")
                else:
                    ids = charuco_ids.flatten()
                    captured_object_points.append(board_points[ids].astype(np.float32))
                    captured_image_points.append(charuco_corners.reshape(-1, 2).astype(np.float32))
                    print(f"Captured view {len(captured_object_points)} with {corner_count} corners.")

            if key == ord("r"):
                captured_object_points.clear()
                captured_image_points.clear()
                print("Discarded captured views.")

            if len(captured_object_points) >= TARGET_CAPTURE_COUNT:
                wants_calibration = True

            if wants_calibration:
                if len(captured_object_points) < MIN_CAPTURE_COUNT:
                    print(f"Capture at least {MIN_CAPTURE_COUNT} varied views first.")
                else:
                    error, camera_matrix, distortion = calibrate(
                        captured_object_points, captured_image_points, image_size
                    )
                    save_calibration(
                        camera_matrix,
                        distortion,
                        image_size,
                        error,
                        len(captured_object_points),
                        capture.camera_identity,
                    )
                    print(f"RMS reprojection error: {error:.4f} pixels")
                    print("Validate with a ruler before using this for robot motion.")
                    break

            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
