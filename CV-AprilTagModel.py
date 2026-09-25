"""Show a camera feed and the camera's pixel offset from a tracked target.

Install dependencies once before running this file:
    py -3.12 -m pip install opencv-contrib-python mediapipe

Active tracker: MediaPipe hand landmarks. The commented AprilTag section below
is ready to use once a physical tag is available.
"""

from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
import numpy as np


# Set True before running to display a mirror image (webcam-style view).
MIRROR_IMAGE = False
CAMERA_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]

# Palm landmarks: wrist plus the four finger MCP joints. Their average is a
# steadier target than a single fingertip.
HAND_POSITION_LANDMARKS = (0, 5, 9, 13, 17)
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)

# The current MediaPipe Tasks API requires a separate model bundle. The script
# downloads this official bundle automatically on its first run.
HAND_LANDMARKER_MODEL_PATH = Path(__file__).parent / "models" / "hand_landmarker.task"
HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)


def pixel_center(points):
    """Return the integer center of an Nx2 collection of pixel points."""
    return tuple(np.rint(np.mean(points, axis=0)).astype(int))


def hand_center_pixels(hand_landmarks, frame_width, frame_height):
    """Return one hand's palm center in pixel coordinates."""
    points = [
        (
            hand_landmarks[index].x * frame_width,
            hand_landmarks[index].y * frame_height,
        )
        for index in HAND_POSITION_LANDMARKS
    ]
    return pixel_center(points)


def draw_hand_landmarks(frame, hand_landmarks):
    """Draw MediaPipe Tasks hand landmarks without the retired Solutions API."""
    height, width = frame.shape[:2]
    points = [
        (int(landmark.x * width), int(landmark.y * height))
        for landmark in hand_landmarks
    ]
    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (255, 100, 0), 2)
    for point in points:
        cv2.circle(frame, point, 3, (0, 255, 0), -1)


def ensure_hand_landmarker_model():
    """Download the official hand-landmarker model once, when needed."""
    if HAND_LANDMARKER_MODEL_PATH.exists():
        return

    HAND_LANDMARKER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    print("Downloading the MediaPipe hand-landmarker model (first run only)...")
    try:
        urlretrieve(HAND_LANDMARKER_MODEL_URL, HAND_LANDMARKER_MODEL_PATH)
    except OSError as error:
        raise RuntimeError(
            "Could not download the hand-landmarker model. Check your internet "
            f"connection, then download it from {HAND_LANDMARKER_MODEL_URL} to "
            f"{HAND_LANDMARKER_MODEL_PATH}."
        ) from error


def draw_target_offset(frame, target_center, target_name):
    """Draw a target offset with the camera frame center defined as (0, 0)."""
    height, width = frame.shape[:2]
    frame_center = (width // 2, height // 2)

    # Image convention: +x is right and +y is down.
    target_offset = (
        target_center[0] - frame_center[0],
        target_center[1] - frame_center[1],
    )
    # The inverse is the camera position relative to the tracked target.
    camera_relative = (-target_offset[0], -target_offset[1])

    cv2.drawMarker(frame, frame_center, (255, 255, 0), cv2.MARKER_CROSS, 22, 2)
    cv2.circle(frame, target_center, 8, (0, 255, 0), -1)
    cv2.line(frame, frame_center, target_center, (0, 255, 0), 2)
    cv2.putText(
        frame,
        f"{target_name} offset: x={target_offset[0]:+d}, y={target_offset[1]:+d} px",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Camera relative to {target_name}: x={camera_relative[0]:+d}, y={camera_relative[1]:+d} px",
        (10, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        2,
    )


def main():
    # Download before opening the camera so a download failure leaves no device
    # handle open.
    ensure_hand_landmarker_model()

    capture = None
    for index in CAMERA_INDICES:
        candidate = cv2.VideoCapture(index)
        if candidate.isOpened():
            ok, _ = candidate.read()
            if ok:
                capture = candidate
                print(f"Using camera index {index}")
                break
        candidate.release()

    if capture is None:
        print("No camera found. Check CAMERA_INDICES and camera permissions.")
        return

    # AprilTag setup -- leave commented until a tag is available.
    # This requires opencv-contrib-python, not just opencv-python.
    # april_tag_dictionary = cv2.aruco.getPredefinedDictionary(
    #     cv2.aruco.DICT_APRILTAG_36h11
    # )
    # april_tag_detector = cv2.aruco.ArucoDetector(april_tag_dictionary)

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(HAND_LANDMARKER_MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    hand_landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
    timestamp_ms = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("Could not read a frame from the camera.")
                break

            if MIRROR_IMAGE:
                frame = cv2.flip(frame, 1)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms += 1  # VIDEO mode requires strictly increasing times.
            result = hand_landmarker.detect_for_video(
                mp_frame, timestamp_ms
            )

            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                frame_height, frame_width = frame.shape[:2]
                target_center = hand_center_pixels(hand, frame_width, frame_height)
                draw_hand_landmarks(frame, hand)
                draw_target_offset(frame, target_center, "hand")
            else:
                cv2.putText(
                    frame,
                    "Show one hand to the camera",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )

            # AprilTag detection -- enable this block when you have a tag, then
            # remove/disable the active MediaPipe hand block above.
            # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # tag_corners, tag_ids, _ = april_tag_detector.detectMarkers(gray_frame)
            # if tag_ids is not None:
            #     tag_index = 0  # Or select a particular tag ID here.
            #     tag_center = pixel_center(tag_corners[tag_index][0])
            #     cv2.aruco.drawDetectedMarkers(frame, tag_corners, tag_ids)
            #     draw_target_offset(frame, tag_center, f"AprilTag {tag_ids[tag_index][0]}")

            cv2.imshow("Hand camera offset", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        hand_landmarker.close()
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
