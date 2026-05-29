# ============================================================
# GOLF TEE DEFECT MODEL + ROBODK SORTING SCRIPT
# SPINNAKER / POINT GREY CAMERA VERSION
# OPENS ROBODK FILE AUTOMATICALLY
# ROBOT ACTS ONLY WHEN SPACEBAR IS PRESSED
# ============================================================

import os
import time
import cv2
import pickle
import numpy as np
import PySpin

from robodk.robolink import *
from robodk.robomath import *

# ============================================================
# SETTINGS
# ============================================================

model_folder_name = "trained_model"
model_file_name = "shape_size_classifier.pkl"

station_file = r"C:\Users\achivate\OneDrive - Cal Poly\DT\ADEPT\HITL-DT\achivate\HaasHC10Updated.rdk"

seconds_between_sorts = 3
confidence_threshold = 0.40
camera_serial_number = None

# ============================================================
# LOAD MODEL
# ============================================================

script_folder = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_folder, model_folder_name, model_file_name)

if not os.path.exists(model_path):
    raise Exception(f"Could not find model:\n{model_path}")

with open(model_path, "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
class_names = model_data["class_names"]
minimum_object_area = model_data["minimum_object_area"]
minimum_fragment_area = model_data.get("minimum_fragment_area", 150)

print("----------------------------------")
print("Loaded golf tee defect model:")
print(model_path)
print("Classes:", class_names)
print("Minimum object area:", minimum_object_area)
print("Minimum fragment area:", minimum_fragment_area)
print("----------------------------------")

# ============================================================
# BIN ASSIGNMENT
# ============================================================

PART_TO_BIN = {
    "perfect": "BIN_1_DROP",
    "usable": "BIN_2_DROP",
    "unusable": "BIN_3_DROP",
    "Unknown": "BIN_3_DROP"
}

# ============================================================
# SPINNAKER CAMERA
# ============================================================

class SpinnakerCamera:
    def __init__(self, serial_number=None):
        self.serial_number = serial_number
        self.system = None
        self.cam_list = None
        self.camera = None

    def start(self):
        self.system = PySpin.System.GetInstance()
        self.cam_list = self.system.GetCameras()

        if self.cam_list.GetSize() == 0:
            self.cam_list.Clear()
            self.system.ReleaseInstance()
            raise Exception("No Point Grey / FLIR cameras found.")

        if self.serial_number is None:
            self.camera = self.cam_list.GetByIndex(0)
        else:
            self.camera = self.cam_list.GetBySerial(self.serial_number)

        self.camera.Init()

        nodemap = self.camera.GetNodeMap()
        acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode("AcquisitionMode"))
        continuous_mode = acquisition_mode.GetEntryByName("Continuous")
        acquisition_mode.SetIntValue(continuous_mode.GetValue())

        self.camera.BeginAcquisition()
        print("Spinnaker camera started.")
        print("----------------------------------")

    def read(self):
        image_result = self.camera.GetNextImage(1000)

        if image_result.IsIncomplete():
            image_result.Release()
            return False, None

        try:
            frame = image_result.GetNDArray().copy()

            if len(frame.shape) == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            elif len(frame.shape) == 3:
                if frame.shape[2] == 3:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                elif frame.shape[2] == 4:
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            image_result.Release()
            return True, frame

        except Exception as e:
            print("Camera frame conversion error:", e)
            image_result.Release()
            return False, None

    def stop(self):
        try:
            if self.camera is not None:
                self.camera.EndAcquisition()
                self.camera.DeInit()
                del self.camera

            if self.cam_list is not None:
                self.cam_list.Clear()

            if self.system is not None:
                self.system.ReleaseInstance()

            print("Spinnaker camera released.")

        except Exception as e:
            print("Camera cleanup error:", e)

# ============================================================
# ROBODK CONNECTION
# ============================================================

RDK = Robolink()

if not os.path.exists(station_file):
    raise Exception("RoboDK station file not found:\n" + station_file)

print("Opening RoboDK station:")
print(station_file)

RDK.AddFile(station_file)
time.sleep(2)

robot = RDK.Item("", ITEM_TYPE_ROBOT)

if not robot.Valid():
    raise Exception("No robot found in RoboDK.")

home = RDK.Item("Home")
approach = RDK.Item("Approach")
middle = RDK.Item("Middle")
pick_app = RDK.Item("_PICK_APP")
pick = RDK.Item("_PICK")

bin_1_drop = RDK.Item("BIN_1_DROP")
bin_2_drop = RDK.Item("BIN_2_DROP")
bin_3_drop = RDK.Item("BIN_3_DROP")

BIN_TARGETS = {
    "BIN_1_DROP": bin_1_drop,
    "BIN_2_DROP": bin_2_drop,
    "BIN_3_DROP": bin_3_drop
}

def check_target(target, target_name):
    if not target.Valid():
        raise Exception(f"Target '{target_name}' not found in RoboDK.")

for target, name in [
    (home, "Home"),
    (approach, "Approach"),
    (middle, "Middle"),
    (pick_app, "_PICK_APP"),
    (pick, "_PICK"),
    (bin_1_drop, "BIN_1_DROP"),
    (bin_2_drop, "BIN_2_DROP"),
    (bin_3_drop, "BIN_3_DROP")
]:
    check_target(target, name)

print("Connected to RoboDK.")
print("Robot:", robot.Name())
print("----------------------------------")

# ============================================================
# ROBOT FUNCTIONS
# ============================================================

def close_gripper():
    print("Gripper close")

def open_gripper():
    print("Gripper open")

def pick_part():
    robot.MoveJ(home)
    robot.MoveJ(approach)
    robot.MoveJ(pick_app)
    robot.MoveL(pick)
    close_gripper()
    robot.MoveL(pick_app)

def drop_part(bin_target):
    robot.MoveJ(middle)
    robot.MoveJ(bin_target)
    open_gripper()
    robot.MoveJ(middle)
    robot.MoveJ(home)

def sort_part(part_name):
    if part_name not in PART_TO_BIN:
        part_name = "Unknown"

    bin_name = PART_TO_BIN[part_name]
    bin_target = BIN_TARGETS[bin_name]

    print("----------------------------------")
    print("SPACEBAR PRESSED")
    print("Sorting class:", part_name)
    print("Assigned bin:", bin_name)
    print("----------------------------------")

    pick_part()
    drop_part(bin_target)

# ============================================================
# FEATURE HELPERS
# ============================================================

def safe_div(a, b):
    if b == 0:
        return 0
    return a / b


def contour_straightness_features(contour):
    points = contour.reshape(-1, 2).astype(np.float32)

    if len(points) < 5:
        return 0, 0, 0, 0, 0

    mean = np.mean(points, axis=0)
    centered = points - mean

    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(covariance)

    order = np.argsort(eigenvalues)[::-1]
    major_axis = eigenvectors[:, order[0]]
    minor_axis = eigenvectors[:, order[1]]

    major_projection = centered @ major_axis
    minor_projection = centered @ minor_axis

    length = np.max(major_projection) - np.min(major_projection)
    width = np.max(minor_projection) - np.min(minor_projection)

    straightness_rms = np.sqrt(np.mean(minor_projection ** 2))
    straightness_max = np.max(np.abs(minor_projection))

    bend_ratio_rms = safe_div(straightness_rms, length)
    bend_ratio_max = safe_div(straightness_max, length)

    length_width_ratio = safe_div(length, width)

    return length, width, bend_ratio_rms, bend_ratio_max, length_width_ratio


def convexity_defect_features(contour):
    if len(contour) < 4:
        return 0, 0, 0

    hull_indices = cv2.convexHull(contour, returnPoints=False)

    if hull_indices is None or len(hull_indices) < 4:
        return 0, 0, 0

    defects = cv2.convexityDefects(contour, hull_indices)

    if defects is None:
        return 0, 0, 0

    defect_depths = []

    for i in range(defects.shape[0]):
        depth = defects[i, 0, 3] / 256.0
        defect_depths.append(depth)

    if len(defect_depths) == 0:
        return 0, 0, 0

    defect_count = len(defect_depths)
    max_defect_depth = max(defect_depths)
    mean_defect_depth = sum(defect_depths) / len(defect_depths)

    return defect_count, max_defect_depth, mean_defect_depth


def texture_scuff_features(gray, mask):
    masked_pixels = gray[mask > 0]

    if len(masked_pixels) == 0:
        return 0, 0, 0, 0

    intensity_mean = float(np.mean(masked_pixels))
    intensity_std = float(np.std(masked_pixels))

    edges = cv2.Canny(gray, 50, 150)
    object_edges = edges[mask > 0]

    edge_density = np.count_nonzero(object_edges) / len(object_edges)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_pixels = laplacian[mask > 0]

    texture_variance = float(np.var(laplacian_pixels))

    return intensity_mean, intensity_std, edge_density, texture_variance


def width_profile_features(contour):
    points = contour.reshape(-1, 2).astype(np.float32)

    if len(points) < 10:
        return 0, 0, 0, 0

    mean = np.mean(points, axis=0)
    centered = points - mean

    covariance = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eig(covariance)

    order = np.argsort(eigenvalues)[::-1]
    major_axis = eigenvectors[:, order[0]]
    minor_axis = eigenvectors[:, order[1]]

    major_projection = centered @ major_axis
    minor_projection = centered @ minor_axis

    min_major = np.min(major_projection)
    max_major = np.max(major_projection)

    bins = np.linspace(min_major, max_major, 12)

    widths = []

    for i in range(len(bins) - 1):
        section_mask = (
            (major_projection >= bins[i]) &
            (major_projection < bins[i + 1])
        )

        section = minor_projection[section_mask]

        if len(section) > 1:
            section_width = np.max(section) - np.min(section)
            widths.append(section_width)

    if len(widths) == 0:
        return 0, 0, 0, 0

    widths = np.array(widths)

    width_mean = float(np.mean(widths))
    width_std = float(np.std(widths))
    width_min = float(np.min(widths))
    width_max = float(np.max(widths))

    width_variation_ratio = safe_div(width_std, width_mean)
    tip_width_ratio = safe_div(width_min, width_max)

    return width_mean, width_std, width_variation_ratio, tip_width_ratio

# ============================================================
# GOLF TEE FEATURE EXTRACTION
# ============================================================

def extract_shape_features_from_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    threshold = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        51,
        5
    )

    kernel = np.ones((5, 5), np.uint8)

    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    threshold = cv2.morphologyEx(
        threshold,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    contours, _ = cv2.findContours(
        threshold,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = [
        c for c in contours
        if cv2.contourArea(c) >= minimum_fragment_area
    ]

    if len(valid_contours) == 0:
        return None, None, threshold, None

    contour = max(valid_contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)

    if area < minimum_object_area:
        return None, None, threshold, None

    total_area = sum(cv2.contourArea(c) for c in valid_contours)

    fragment_count = len(valid_contours)
    largest_area_ratio = safe_div(area, total_area)

    perimeter = cv2.arcLength(contour, True)

    x, y, w, h = cv2.boundingRect(contour)

    aspect_ratio = safe_div(w, h)

    circularity = safe_div(
        4 * np.pi * area,
        perimeter * perimeter
    )

    extent = safe_div(area, w * h)

    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    solidity = safe_div(area, hull_area)

    min_rect = cv2.minAreaRect(contour)
    rect_w, rect_h = min_rect[1]

    min_rect_long = max(rect_w, rect_h)
    min_rect_short = min(rect_w, rect_h)
    min_rect_ratio = safe_div(min_rect_long, min_rect_short)

    equivalent_diameter = np.sqrt(4 * area / np.pi)

    (
        tee_length,
        tee_width,
        bend_ratio_rms,
        bend_ratio_max,
        length_width_ratio
    ) = contour_straightness_features(contour)

    (
        defect_count,
        max_defect_depth,
        mean_defect_depth
    ) = convexity_defect_features(contour)

    (
        width_mean,
        width_std,
        width_variation_ratio,
        tip_width_ratio
    ) = width_profile_features(contour)

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    (
        intensity_mean,
        intensity_std,
        edge_density,
        texture_variance
    ) = texture_scuff_features(gray, mask)

    features = np.array([[
        area,
        perimeter,
        w,
        h,
        aspect_ratio,
        circularity,
        extent,
        solidity,

        min_rect_long,
        min_rect_short,
        min_rect_ratio,
        equivalent_diameter,

        tee_length,
        tee_width,
        length_width_ratio,
        bend_ratio_rms,
        bend_ratio_max,

        defect_count,
        max_defect_depth,
        mean_defect_depth,

        width_mean,
        width_std,
        width_variation_ratio,
        tip_width_ratio,

        fragment_count,
        largest_area_ratio,

        intensity_mean,
        intensity_std,
        edge_density,
        texture_variance
    ]], dtype=np.float32)

    info = {
        "area": area,
        "perimeter": perimeter,
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
        "extent": extent,
        "solidity": solidity,
        "bend_ratio_rms": bend_ratio_rms,
        "bend_ratio_max": bend_ratio_max,
        "fragment_count": fragment_count,
        "defect_count": defect_count,
        "max_defect_depth": max_defect_depth,
        "mean_defect_depth": mean_defect_depth,
        "edge_density": edge_density,
        "texture_variance": texture_variance,
        "largest_area_ratio": largest_area_ratio
    }

    return features, contour, threshold, info

# ============================================================
# PREDICTION
# ============================================================

def identify_part_from_frame(frame):
    features, contour, threshold, info = extract_shape_features_from_frame(frame)

    if features is None:
        return "Unknown", 0.0, contour, threshold, info, {}

    predicted_name = model.predict(features)[0]

    probabilities_dict = {}

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities))

        for class_name, prob in zip(model.classes_, probabilities):
            probabilities_dict[class_name] = float(prob)
    else:
        confidence = 1.0
        probabilities_dict[predicted_name] = 1.0

    if confidence < confidence_threshold:
        sorting_name = "Unknown"
    else:
        sorting_name = predicted_name

    return sorting_name, confidence, contour, threshold, info, probabilities_dict


def print_sorting_information(sorting_name, confidence, info, probabilities_dict):
    print("----------------------------------")
    print("Predicted / sorting class:", sorting_name)
    print("Confidence:", str(round(confidence * 100, 1)) + "%")

    print("Class probabilities:")
    for class_name, prob in probabilities_dict.items():
        print("   " + class_name + ":", round(prob, 3))

    if info is not None:
        print("Area:", round(info["area"], 2))
        print("Width:", info["w"])
        print("Height:", info["h"])
        print("Aspect ratio:", round(info["aspect_ratio"], 3))
        print("Solidity:", round(info["solidity"], 3))
        print("Bend RMS:", round(info["bend_ratio_rms"], 4))
        print("Bend Max:", round(info["bend_ratio_max"], 4))
        print("Fragments:", info["fragment_count"])
        print("Defects:", info["defect_count"])
        print("Edge density:", round(info["edge_density"], 4))
        print("Texture variance:", round(info["texture_variance"], 2))

    print("----------------------------------")

# ============================================================
# MAIN LOOP
# ============================================================

camera = SpinnakerCamera(serial_number=camera_serial_number)
camera.start()

print("Manual trigger mode is ON.")
print("Robot only sorts when SPACEBAR is pressed.")
print("Press q to quit.")
print("----------------------------------")

last_sorted_time = 0

try:
    while True:
        ret, frame = camera.read()

        if not ret:
            continue

        sorting_name, confidence, contour, threshold, info, probabilities_dict = identify_part_from_frame(frame)

        display_frame = frame.copy()

        if contour is not None and info is not None:
            cv2.drawContours(display_frame, [contour], -1, (0, 255, 0), 2)

            x = info["x"]
            y = info["y"]
            w = info["w"]
            h = info["h"]

            cv2.rectangle(
                display_frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

        label = f"Ready: {sorting_name} | Confidence: {confidence * 100:.1f}%"

        color = (0, 255, 0)

        if sorting_name == "usable":
            color = (0, 255, 255)
        elif sorting_name in ["unusable", "Unknown"]:
            color = (0, 0, 255)

        cv2.putText(
            display_frame,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        cv2.putText(
            display_frame,
            "Press SPACE to sort | Press q to quit",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if info is not None:
            feature_text = (
                f"Area:{info['area']:.0f} "
                f"AR:{info['aspect_ratio']:.2f} "
                f"Bend:{info['bend_ratio_max']:.3f} "
                f"Frag:{info['fragment_count']} "
                f"Def:{info['defect_count']}"
            )

            cv2.putText(
                display_frame,
                feature_text,
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        y_text = 160
        for class_name, prob in probabilities_dict.items():
            prob_text = f"{class_name}: {prob * 100:.1f}%"

            cv2.putText(
                display_frame,
                prob_text,
                (20, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

            y_text += 30

        cv2.imshow("Spinnaker Golf Tee Defect Sorting", display_frame)
        cv2.imshow("Threshold View", threshold)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == 32:
            current_time = time.time()

            if current_time - last_sorted_time < seconds_between_sorts:
                print("Wait before sorting again.")
                continue

            print_sorting_information(
                sorting_name,
                confidence,
                info,
                probabilities_dict
            )

            sort_part(sorting_name)

            last_sorted_time = time.time()

finally:
    camera.stop()
    cv2.destroyAllWindows()
    print("Program ended.")