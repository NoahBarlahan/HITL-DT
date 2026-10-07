"""Shared, editable settings for AprilTag detection and ChArUco calibration."""

from pathlib import Path

import cv2


PROJECT_DIRECTORY = Path(__file__).resolve().parents[1]
CAMERA_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
CAMERA_BACKEND = "realsense"  # "opencv" for a normal webcam
COLOR_WIDTH = 640
COLOR_HEIGHT = 480
COLOR_FPS = 30
CALIBRATION_FILE = PROJECT_DIRECTORY / "calibration" / "camera_calibration.npz"

# Calib.io checkerboard: count INNER intersections, not squares.
# Set these after counting, or supply --squares ACROSS DOWN to the calibrator.
CHECKERBOARD_COLUMNS = 10  # 11 squares across => 10 inner intersections
CHECKERBOARD_ROWS = 7  # 8 squares down => 7 inner intersections
CHECKERBOARD_SQUARE_MM = 15.0

# ChArUco board settings. Generate a new board after changing any value here.
CHARUCO_DICTIONARY = cv2.aruco.DICT_4X4_50
CHARUCO_SQUARES_X = 7
CHARUCO_SQUARES_Y = 5
CHARUCO_SQUARE_LENGTH_MM = 25.0
CHARUCO_MARKER_LENGTH_MM = 18.0
CHARUCO_OUTPUT_PIXELS_PER_SQUARE = 200

# AprilTag runtime settings. Change these to exactly match the printed tag.
APRILTAG_DICTIONARY = cv2.aruco.DICT_APRILTAG_25h9
APRILTAG_SIZE_MM = 77.8  # Detection edge, excluding the 100 mm paper/white margin.
TARGET_TAG_ID = None


# Fixture checks require the camera to return to the same known robot pose.
FIXTURE_DATUM_FILE = PROJECT_DIRECTORY / "calibration" / "fixture_datum.json"
FIXTURE_POSITION_TOLERANCE_IN = 0.05  # Total translation, not per-axis.
FIXTURE_ROTATION_TOLERANCE_DEG = 1.0
FIXTURE_SAMPLE_COUNT = 30
FIXTURE_MAX_REPROJECTION_PX = 1.0
FIXTURE_STABILITY_POSITION_IN = 0.025  # Maximum sample deviation from mean.
FIXTURE_STABILITY_ROTATION_DEG = 0.75
