"""Use the same camera, sensor, and resolution for calibration and detection."""

import cv2
import numpy as np

from vision.vision_config import CAMERA_BACKEND, CAMERA_INDICES, COLOR_WIDTH, COLOR_HEIGHT, COLOR_FPS


class RealSenseColor:
    def __init__(self):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, COLOR_WIDTH, COLOR_HEIGHT,
                             rs.format.bgr8, COLOR_FPS)
        profile = self.pipeline.start(config)
        device = profile.get_device()
        self.camera_identity = "realsense:" + device.get_info(rs.camera_info.serial_number)
        print(f"Using {device.get_info(rs.camera_info.name)} color stream: "
              f"{COLOR_WIDTH}x{COLOR_HEIGHT}, {COLOR_FPS} FPS, {self.camera_identity}")

    def read(self):
        try:
            frame = self.pipeline.wait_for_frames(5000).get_color_frame()
        except RuntimeError as error:
            raise RuntimeError(
                "No RealSense color frame received. Close realsense_depth_viewer.py "
                "and all other camera applications before calibration/detection."
            ) from error
        return (True, np.asanyarray(frame.get_data()).copy()) if frame else (False, None)

    def release(self):
        self.pipeline.stop()


class Webcam:
    def __init__(self, capture, index):
        self.capture = capture
        self.camera_identity = f"opencv:index:{index}"

    def read(self):
        return self.capture.read()

    def release(self):
        self.capture.release()


def open_camera():
    if CAMERA_BACKEND == "realsense":
        return RealSenseColor()
    if CAMERA_BACKEND != "opencv":
        raise ValueError("CAMERA_BACKEND must be realsense or opencv")
    for index in CAMERA_INDICES:
        capture = cv2.VideoCapture(index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, COLOR_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, COLOR_HEIGHT)
        if capture.isOpened():
            ok, frame = capture.read()
            if ok and frame is not None:
                print(f"Using webcam index {index}")
                return Webcam(capture, index)
        capture.release()
    raise RuntimeError("No usable camera found")
