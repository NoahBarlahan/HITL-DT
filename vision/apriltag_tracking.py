"""Shared AprilTag pose estimation. Geometry is in mm; displays use inches."""

import math

import cv2
import numpy as np

from vision.vision_config import APRILTAG_SIZE_MM, CALIBRATION_FILE

MM_PER_INCH = 25.4


def load_calibration():
    if not CALIBRATION_FILE.exists():
        return None
    with np.load(CALIBRATION_FILE, allow_pickle=False) as data:
        return {
            "camera_matrix": data["camera_matrix"].copy(),
            "distortion": data["distortion_coefficients"].copy(),
            "image_size": (int(data["image_width"]), int(data["image_height"])),
            "reprojection_error": float(data["reprojection_error"]),
            "camera_identity": str(data["camera_identity"]) if "camera_identity" in data else None,
        }


def calibration_status(calibration, camera_identity, image_size):
    if calibration is None:
        return "No saved calibration; tag pose unavailable"
    if calibration["camera_identity"] != camera_identity:
        return "Calibration camera mismatch; tag pose unavailable"
    if calibration["image_size"] != image_size:
        return "Calibration resolution mismatch; tag pose unavailable"
    return None


def tag_center_pixel(corners):
    """Project the square's physical center, including perspective at a tilt."""
    square = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], np.float32)
    transform = cv2.getPerspectiveTransform(square, np.asarray(corners, np.float32).reshape(4, 2))
    return cv2.perspectiveTransform(np.array([[[0.5, 0.5]]], np.float32), transform)[0, 0]


def tag_object_points(tag_size_mm):
    half = tag_size_mm / 2.0
    return np.array([[-half, half, 0], [half, half, 0],
                     [half, -half, 0], [-half, -half, 0]], np.float32)


def rotation_matrix_to_euler_degrees(rotation_matrix):
    sy = math.hypot(rotation_matrix[0, 0], rotation_matrix[1, 0])
    pitch = math.atan2(-rotation_matrix[2, 0], sy)
    if sy >= 1e-6:
        roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        yaw = 0.0
    return tuple(math.degrees(angle) for angle in (roll, pitch, yaw))


def estimate_tag_pose(tag_corners, calibration):
    if calibration is None or APRILTAG_SIZE_MM is None or APRILTAG_SIZE_MM <= 0:
        return None
    objects = tag_object_points(APRILTAG_SIZE_MM)
    images = np.asarray(tag_corners, np.float32).reshape(4, 2)
    success, rotation_vector, translation_vector = cv2.solvePnP(
        objects, images, calibration["camera_matrix"], calibration["distortion"],
        flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not success or not np.isfinite(translation_vector).all() or translation_vector[2, 0] <= 0:
        return None
    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    projected, _ = cv2.projectPoints(objects, rotation_vector, translation_vector,
                                    calibration["camera_matrix"], calibration["distortion"])
    error = float(np.mean(np.linalg.norm(projected.reshape(4, 2) - images, axis=1)))
    return dict(translation_mm=translation_vector.reshape(3),
                rotation_vector=rotation_vector,
                euler_degrees=rotation_matrix_to_euler_degrees(rotation_matrix),
                reprojection_error=error)
