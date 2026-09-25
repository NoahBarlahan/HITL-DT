# ============================================================
# HITL GOLF TEE EXPERIMENT SORTER
#
# A = autonomous
# N = novice assisted on low confidence
# E = expert assisted on low confidence
#
# SPACE = collect current run
#
# If low confidence in novice/expert mode:
#   1 = BIN_1_DROP
#   2 = BIN_2_DROP
#   3 = BIN_3_DROP
#
# Q = quit
# ============================================================

import os
import time
import csv
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

samples_per_condition = 10
confidence_threshold = 0.40
seconds_between_sorts = 3

camera_serial_number = None

experiment_folder_name = "hitl_experiment_data"
experiment_csv_name = "all_hitl_runs.csv"

class_names = ["perfect", "usable", "unusable"]

PART_TO_BIN = {
    "perfect": "BIN_1_DROP",
    "usable": "BIN_2_DROP",
    "unusable": "BIN_3_DROP",
    "Unknown": "BIN_3_DROP"
}

# ============================================================
# PATHS
# ============================================================

script_folder = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(script_folder, model_folder_name, model_file_name)

experiment_folder = os.path.join(script_folder, experiment_folder_name)
image_folder = os.path.join(experiment_folder, "images")

os.makedirs(experiment_folder, exist_ok=True)
os.makedirs(image_folder, exist_ok=True)

csv_path = os.path.join(experiment_folder, experiment_csv_name)

# ============================================================
# LOAD MODEL
# ============================================================

if not os.path.exists(model_path):
    raise Exception(f"Could not find model:\n{model_path}")

with open(model_path, "rb") as f:
    model_data = pickle.load(f)

model = model_data["model"]
minimum_object_area = model_data["minimum_object_area"]
minimum_fragment_area = model_data.get("minimum_fragment_area", 150)

# ============================================================
# FEATURE FUNCTIONS
# ============================================================

def safe_div(a, b):
    return 0 if b == 0 else a / b


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel, iterations=2)
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel, iterations=1)

    return gray, threshold


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

    depths = []

    for i in range(defects.shape[0]):
        depths.append(defects[i, 0, 3] / 256.0)

    if len(depths) == 0:
        return 0, 0, 0

    return len(depths), max(depths), sum(depths) / len(depths)


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

    bins = np.linspace(np.min(major_projection), np.max(major_projection), 12)

    widths = []

    for i in range(len(bins) - 1):
        mask = (major_projection >= bins[i]) & (major_projection < bins[i + 1])
        section = minor_projection[mask]

        if len(section) > 1:
            widths.append(np.max(section) - np.min(section))

    if len(widths) == 0:
        return 0, 0, 0, 0

    widths = np.array(widths)

    width_mean = float(np.mean(widths))
    width_std = float(np.std(widths))
    width_min = float(np.min(widths))
    width_max = float(np.max(widths))

    return width_mean, width_std, safe_div(width_std, width_mean), safe_div(width_min, width_max)


def texture_scuff_features(gray, mask):
    pixels = gray[mask > 0]

    if len(pixels) == 0:
        return 0, 0, 0, 0

    intensity_mean = float(np.mean(pixels))
    intensity_std = float(np.std(pixels))

    edges = cv2.Canny(gray, 50, 150)
    object_edges = edges[mask > 0]
    edge_density = np.count_nonzero(object_edges) / len(object_edges)

    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    lap_pixels = laplacian[mask > 0]
    texture_variance = float(np.var(lap_pixels))

    return intensity_mean, intensity_std, edge_density, texture_variance


def extract_features_from_image(image):
    gray, threshold = preprocess_image(image)

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
    circularity = safe_div(4 * np.pi * area, perimeter * perimeter)
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

    tee_length, tee_width, bend_ratio_rms, bend_ratio_max, length_width_ratio = (
        contour_straightness_features(contour)
    )

    defect_count, max_defect_depth, mean_defect_depth = (
        convexity_defect_features(contour)
    )

    width_mean, width_std, width_variation_ratio, tip_width_ratio = (
        width_profile_features(contour)
    )

    mask = np.zeros(gray.shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)

    intensity_mean, intensity_std, edge_density, texture_variance = (
        texture_scuff_features(gray, mask)
    )

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
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "area": area,
        "aspect_ratio": aspect_ratio,
        "bend_ratio_max": bend_ratio_max,
        "fragment_count": fragment_count,
        "defect_count": defect_count
    }

    return features, contour, threshold, info


def predict_from_frame(frame):
    features, contour, threshold, info = extract_features_from_image(frame)

    if features is None:
        return "Unknown", "Unknown", 0.0, {}, contour, threshold, info

    raw_prediction = model.predict(features)[0]

    probabilities = {}

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)[0]

        for class_name, prob in zip(model.classes_, probs):
            probabilities[class_name] = float(prob)

        confidence = float(max(probs))
    else:
        confidence = 1.0
        probabilities[raw_prediction] = 1.0

    display_prediction = raw_prediction

    if confidence < confidence_threshold:
        display_prediction = "Unknown"

    return raw_prediction, display_prediction, confidence, probabilities, contour, threshold, info

# ============================================================
# CSV LOGGING
# ============================================================

def get_next_global_run_id():
    if not os.path.exists(csv_path):
        return 1

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    return len(rows) + 1


def write_log(row):
    file_exists = os.path.exists(csv_path)

    fieldnames = [
        "global_run_id",
        "timestamp",
        "condition",
        "condition_trial_number",
        "robot_raw_prediction",
        "robot_display_prediction",
        "confidence",
        "perfect_probability",
        "usable_probability",
        "unusable_probability",
        "low_confidence",
        "human_role",
        "human_selected_bin",
        "final_decision",
        "assigned_bin",
        "robot_acted",
        "decision_time_seconds",
        "image_path",
        "area",
        "aspect_ratio",
        "bend_ratio_max",
        "fragment_count",
        "defect_count"
    ]

    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(row)


def save_trial_image(frame, condition, trial_number):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_name = f"{condition}_trial_{trial_number}_{timestamp}.jpg"
    image_path = os.path.join(image_folder, file_name)
    cv2.imwrite(image_path, frame)
    return image_path

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
# ROBODK
# ============================================================

RDK = Robolink()

if not os.path.exists(station_file):
    raise Exception("RoboDK station file not found:\n" + station_file)

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


def sort_to_bin(bin_name):
    if bin_name not in BIN_TARGETS:
        bin_name = "BIN_3_DROP"

    bin_target = BIN_TARGETS[bin_name]

    print("----------------------------------")
    print("Sorting to:", bin_name)
    print("----------------------------------")

    pick_part()
    drop_part(bin_target)

    return bin_name

# ============================================================
# EXPERIMENT STATE
# ============================================================

current_condition = None
current_trial = 0
waiting_for_human = False
decision_start_time = None

pending_frame = None
pending_raw_prediction = None
pending_display_prediction = None
pending_confidence = None
pending_probabilities = None
pending_info = None
pending_low_confidence = None
pending_human_role = None

last_sorted_time = 0

# ============================================================
# TRIAL HANDLING
# ============================================================

def start_condition(condition):
    global current_condition
    global current_trial
    global waiting_for_human

    current_condition = condition
    current_trial = 0
    waiting_for_human = False

    print("----------------------------------")
    print("Started condition:", condition)
    print("Runs to collect:", samples_per_condition)
    print("----------------------------------")


def choose_autonomous_bin(raw_prediction, display_prediction):
    if raw_prediction in PART_TO_BIN and raw_prediction != "Unknown":
        return PART_TO_BIN[raw_prediction], raw_prediction

    if display_prediction in PART_TO_BIN and display_prediction != "Unknown":
        return PART_TO_BIN[display_prediction], display_prediction

    return "BIN_3_DROP", "Unknown"


def complete_trial_with_bin(
    frame,
    raw_prediction,
    display_prediction,
    confidence,
    probabilities,
    low_confidence,
    human_role,
    human_selected_bin,
    final_decision,
    assigned_bin,
    decision_time,
    info
):
    global current_trial

    current_trial += 1

    image_path = save_trial_image(frame, current_condition, current_trial)

    assigned_bin = sort_to_bin(assigned_bin)

    row = {
        "global_run_id": get_next_global_run_id(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "condition": current_condition,
        "condition_trial_number": current_trial,
        "robot_raw_prediction": raw_prediction,
        "robot_display_prediction": display_prediction,
        "confidence": round(confidence, 4),
        "perfect_probability": round(probabilities.get("perfect", 0.0), 4),
        "usable_probability": round(probabilities.get("usable", 0.0), 4),
        "unusable_probability": round(probabilities.get("unusable", 0.0), 4),
        "low_confidence": low_confidence,
        "human_role": human_role,
        "human_selected_bin": human_selected_bin,
        "final_decision": final_decision,
        "assigned_bin": assigned_bin,
        "robot_acted": True,
        "decision_time_seconds": round(decision_time, 3),
        "image_path": image_path,
        "area": round(info["area"], 3) if info is not None else "",
        "aspect_ratio": round(info["aspect_ratio"], 4) if info is not None else "",
        "bend_ratio_max": round(info["bend_ratio_max"], 5) if info is not None else "",
        "fragment_count": info["fragment_count"] if info is not None else "",
        "defect_count": info["defect_count"] if info is not None else ""
    }

    write_log(row)

    print("----------------------------------")
    print("Logged run", current_trial, "of", samples_per_condition)
    print("Condition:", current_condition)
    print("Raw prediction:", raw_prediction)
    print("Displayed prediction:", display_prediction)
    print("Confidence:", round(confidence * 100, 1), "%")
    print("Human selected bin:", human_selected_bin)
    print("Final decision:", final_decision)
    print("Assigned bin:", assigned_bin)
    print("CSV:", csv_path)
    print("----------------------------------")

    if current_trial >= samples_per_condition:
        print("----------------------------------")
        print("Condition complete:", current_condition)
        print("Press A, N, or E to start another condition.")
        print("----------------------------------")


def handle_trial(frame, raw_prediction, display_prediction, confidence, probabilities, info):
    global waiting_for_human
    global decision_start_time
    global pending_frame
    global pending_raw_prediction
    global pending_display_prediction
    global pending_confidence
    global pending_probabilities
    global pending_info
    global pending_low_confidence
    global pending_human_role

    if current_condition is None:
        return

    if current_trial >= samples_per_condition:
        return

    low_confidence = confidence < confidence_threshold or display_prediction == "Unknown"

    if current_condition == "autonomous":
        assigned_bin, final_decision = choose_autonomous_bin(
            raw_prediction,
            display_prediction
        )

        complete_trial_with_bin(
            frame=frame,
            raw_prediction=raw_prediction,
            display_prediction=display_prediction,
            confidence=confidence,
            probabilities=probabilities,
            low_confidence=low_confidence,
            human_role="none",
            human_selected_bin="none",
            final_decision=final_decision,
            assigned_bin=assigned_bin,
            decision_time=0.0,
            info=info
        )

    elif current_condition in ["novice", "expert"]:
        if low_confidence:
            waiting_for_human = True
            decision_start_time = time.time()

            pending_frame = frame.copy()
            pending_raw_prediction = raw_prediction
            pending_display_prediction = display_prediction
            pending_confidence = confidence
            pending_probabilities = probabilities
            pending_info = info
            pending_low_confidence = low_confidence
            pending_human_role = current_condition

            print("----------------------------------")
            print("LOW CONFIDENCE.")
            print("Ask", current_condition.upper(), "to select a bin.")
            print("1 = BIN_1_DROP")
            print("2 = BIN_2_DROP")
            print("3 = BIN_3_DROP")
            print("----------------------------------")

        else:
            assigned_bin = PART_TO_BIN.get(display_prediction, "BIN_3_DROP")

            complete_trial_with_bin(
                frame=frame,
                raw_prediction=raw_prediction,
                display_prediction=display_prediction,
                confidence=confidence,
                probabilities=probabilities,
                low_confidence=low_confidence,
                human_role="none",
                human_selected_bin="none",
                final_decision=display_prediction,
                assigned_bin=assigned_bin,
                decision_time=0.0,
                info=info
            )


def handle_human_bin_selection(bin_number):
    global waiting_for_human
    global last_sorted_time

    if not waiting_for_human:
        return

    if bin_number == 1:
        selected_bin = "BIN_1_DROP"
    elif bin_number == 2:
        selected_bin = "BIN_2_DROP"
    elif bin_number == 3:
        selected_bin = "BIN_3_DROP"
    else:
        return

    decision_time = time.time() - decision_start_time

    complete_trial_with_bin(
        frame=pending_frame,
        raw_prediction=pending_raw_prediction,
        display_prediction=pending_display_prediction,
        confidence=pending_confidence,
        probabilities=pending_probabilities,
        low_confidence=pending_low_confidence,
        human_role=pending_human_role,
        human_selected_bin=selected_bin,
        final_decision="human_selected_" + selected_bin,
        assigned_bin=selected_bin,
        decision_time=decision_time,
        info=pending_info
    )

    waiting_for_human = False
    last_sorted_time = time.time()

# ============================================================
# MAIN LOOP
# ============================================================

camera = SpinnakerCamera(serial_number=camera_serial_number)
camera.start()

print("----------------------------------")
print("HITL Golf Tee Experiment Sorter")
print("A = autonomous")
print("N = novice")
print("E = expert")
print("SPACE = collect run")
print("Low confidence in N/E: 1 BIN_1, 2 BIN_2, 3 BIN_3")
print("Q = quit")
print("CSV:", csv_path)
print("----------------------------------")

try:
    while True:
        ret, frame = camera.read()

        if not ret:
            continue

        raw_prediction, display_prediction, confidence, probabilities, contour, threshold, info = predict_from_frame(frame)

        display = frame.copy()

        if contour is not None and info is not None:
            cv2.drawContours(display, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = info["x"], info["y"], info["w"], info["h"]
            cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if display_prediction == "perfect":
            color = (0, 255, 0)
        elif display_prediction == "usable":
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        condition_text = "No condition selected"

        if current_condition is not None:
            condition_text = f"{current_condition} | Run {current_trial + 1}/{samples_per_condition}"

            if current_trial >= samples_per_condition:
                condition_text = current_condition + " complete"

        cv2.putText(display, condition_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        cv2.putText(
            display,
            f"Prediction: {display_prediction} | Raw: {raw_prediction} | Conf: {confidence * 100:.1f}%",
            (20, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            color,
            2
        )

        if waiting_for_human:
            instruction = "LOW CONFIDENCE: 1 BIN_1 | 2 BIN_2 | 3 BIN_3"
        else:
            instruction = "A auto | N novice | E expert | SPACE collect | Q quit"

        cv2.putText(display, instruction, (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        y_text = 155

        for class_name in class_names:
            prob = probabilities.get(class_name, 0.0)
            cv2.putText(display, f"{class_name}: {prob * 100:.1f}%", (20, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
            y_text += 28

        if info is not None:
            cv2.putText(
                display,
                f"Area:{info['area']:.0f} Bend:{info['bend_ratio_max']:.3f} Frag:{info['fragment_count']} Def:{info['defect_count']}",
                (20, y_text + 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (255, 255, 255),
                2
            )

        cv2.imshow("HITL Golf Tee Experiment Sorter", display)
        cv2.imshow("Threshold View", threshold)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("a"):
            start_condition("autonomous")

        elif key == ord("n"):
            start_condition("novice")

        elif key == ord("e"):
            start_condition("expert")

        elif key == ord("1"):
            handle_human_bin_selection(1)

        elif key == ord("2"):
            handle_human_bin_selection(2)

        elif key == ord("3"):
            handle_human_bin_selection(3)

        elif key == 32:
            if current_condition is None:
                print("Select condition first: A, N, or E.")
                continue

            if waiting_for_human:
                print("Waiting for human bin selection: 1, 2, or 3.")
                continue

            current_time = time.time()

            if current_time - last_sorted_time < seconds_between_sorts:
                print("Wait before collecting next run.")
                continue

            handle_trial(
                frame=frame.copy(),
                raw_prediction=raw_prediction,
                display_prediction=display_prediction,
                confidence=confidence,
                probabilities=probabilities,
                info=info
            )

            last_sorted_time = time.time()

finally:
    camera.stop()
    cv2.destroyAllWindows()
    print("Program ended.")
    print("CSV saved to:")
    print(csv_path)