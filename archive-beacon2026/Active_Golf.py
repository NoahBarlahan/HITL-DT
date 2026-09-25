# ============================================================
# ACTIVE LEARNING GOLF TEE SORTER
# SPINNAKER CAMERA + ROBODK + HUMAN-IN-THE-LOOP LEARNING
#
# Keys:
#   S = accept model prediction and sort
#   1 = correct label as perfect, retrain, sort
#   2 = correct label as usable, retrain, sort
#   3 = correct label as unusable, retrain, sort
#   Q = quit
# ============================================================

import os
import time
import csv
import cv2
import pickle
import shutil
import numpy as np
import PySpin

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from robodk.robolink import *
from robodk.robomath import *

# ============================================================
# SETTINGS
# ============================================================

dataset_folder_name = "Golf_tees_augmented"
active_folder_name = "active_learning_images"

model_folder_name = "trained_model"
model_file_name = "shape_size_classifier.pkl"
csv_file_name = "active_learning_features.csv"

station_file = r"C:\Users\achivate\OneDrive - Cal Poly\DT\ADEPT\HITL-DT\achivate\HaasHC10Updated.rdk"

class_names = ["perfect", "usable", "unusable"]

seconds_between_sorts = 3
confidence_threshold = 0.50

camera_serial_number = None

minimum_object_area = 800
minimum_fragment_area = 150

valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

# ============================================================
# PATHS
# ============================================================

script_folder = os.path.dirname(os.path.abspath(__file__))

dataset_folder = os.path.join(script_folder, dataset_folder_name)
active_folder = os.path.join(script_folder, active_folder_name)

model_folder = os.path.join(script_folder, model_folder_name)
os.makedirs(model_folder, exist_ok=True)

model_path = os.path.join(model_folder, model_file_name)
csv_path = os.path.join(model_folder, csv_file_name)

for class_name in class_names:
    os.makedirs(os.path.join(active_folder, class_name), exist_ok=True)

# ============================================================
# FEATURE NAMES
# ============================================================

feature_names = [
    "area",
    "perimeter",
    "width",
    "height",
    "aspect_ratio",
    "circularity",
    "extent",
    "solidity",
    "min_rect_long",
    "min_rect_short",
    "min_rect_ratio",
    "equivalent_diameter",
    "tee_length",
    "tee_width",
    "length_width_ratio",
    "bend_ratio_rms",
    "bend_ratio_max",
    "defect_count",
    "max_defect_depth",
    "mean_defect_depth",
    "width_mean",
    "width_std",
    "width_variation_ratio",
    "tip_width_ratio",
    "fragment_count",
    "largest_area_ratio",
    "intensity_mean",
    "intensity_std",
    "edge_density",
    "texture_variance"
]

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def safe_div(a, b):
    if b == 0:
        return 0
    return a / b


def make_new_model():
    random_forest = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=123
    )

    extra_trees = ExtraTreesClassifier(
        n_estimators=500,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=123
    )

    voting_model = VotingClassifier(
        estimators=[
            ("rf", random_forest),
            ("et", extra_trees)
        ],
        voting="soft"
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", voting_model)
    ])

    return pipeline

# ============================================================
# IMAGE PREPROCESSING + FEATURES
# ============================================================

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

    defect_depths = []

    for i in range(defects.shape[0]):
        depth = defects[i, 0, 3] / 256.0
        defect_depths.append(depth)

    if len(defect_depths) == 0:
        return 0, 0, 0

    return len(defect_depths), max(defect_depths), sum(defect_depths) / len(defect_depths)


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
        section_mask = (
            (major_projection >= bins[i]) &
            (major_projection < bins[i + 1])
        )

        section = minor_projection[section_mask]

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

    features = [
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
    ]

    info = {
        "x": x,
        "y": y,
        "w": w,
        "h": h,
        "area": area,
        "aspect_ratio": aspect_ratio,
        "bend_ratio_max": bend_ratio_max,
        "fragment_count": fragment_count,
        "defect_count": defect_count,
        "edge_density": edge_density,
        "texture_variance": texture_variance
    }

    return np.array([features], dtype=np.float32), contour, threshold, info

# ============================================================
# DATASET LOADING + TRAINING
# ============================================================

def load_folder_dataset():
    X = []
    y = []

    if not os.path.exists(dataset_folder):
        print("Base dataset folder not found:", dataset_folder)
        return X, y

    for class_name in class_names:
        class_folder = os.path.join(dataset_folder, class_name)

        if not os.path.exists(class_folder):
            continue

        for file_name in os.listdir(class_folder):
            ext = os.path.splitext(file_name)[1].lower()

            if ext not in valid_extensions:
                continue

            path = os.path.join(class_folder, file_name)
            image = cv2.imread(path)

            if image is None:
                continue

            features, _, _, _ = extract_features_from_image(image)

            if features is not None:
                X.append(features[0])
                y.append(class_name)

    return X, y


def load_active_csv():
    X = []
    y = []

    if not os.path.exists(csv_path):
        return X, y

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            label = row["label"]

            features = []

            for name in feature_names:
                features.append(float(row[name]))

            X.append(features)
            y.append(label)

    return X, y


def train_and_save_model():
    X_base, y_base = load_folder_dataset()
    X_active, y_active = load_active_csv()

    X = X_base + X_active
    y = y_base + y_active

    if len(X) < 6:
        raise Exception(
            "Not enough training samples. Add images to Golf_tees_augmented first."
        )

    unique_classes = sorted(list(set(y)))

    if len(unique_classes) < 2:
        raise Exception("Need at least two classes to train.")

    X = np.array(X, dtype=np.float32)
    y = np.array(y)

    model = make_new_model()
    model.fit(X, y)

    model_data = {
        "model_type": "shape_size_classifier",
        "model_version": "active_learning_golf_tee_v1",
        "class_names": class_names,
        "feature_names": feature_names,
        "minimum_object_area": minimum_object_area,
        "minimum_fragment_area": minimum_fragment_area,
        "model": model
    }

    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)

    print("----------------------------------")
    print("Model trained and saved.")
    print("Total samples:", len(X))
    print("Active samples:", len(X_active))
    print("----------------------------------")

    return model


def load_or_train_model():
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            model_data = pickle.load(f)

        print("----------------------------------")
        print("Loaded existing model:")
        print(model_path)
        print("----------------------------------")

        return model_data["model"]

    print("No model found. Training initial model...")
    return train_and_save_model()


def save_active_example(frame, features, label):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    image_name = f"{label}_{timestamp}.jpg"

    image_path = os.path.join(active_folder, label, image_name)
    cv2.imwrite(image_path, frame)

    file_exists = os.path.exists(csv_path)

    with open(csv_path, "a", newline="") as f:
        fieldnames = ["label", "image_path"] + feature_names

        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        row = {
            "label": label,
            "image_path": image_path
        }

        for name, value in zip(feature_names, features[0]):
            row[name] = float(value)

        writer.writerow(row)

    print("----------------------------------")
    print("Saved active learning example:")
    print("Label:", label)
    print("Image:", image_path)
    print("----------------------------------")

# ============================================================
# PREDICTION
# ============================================================

model = load_or_train_model()

def predict_from_frame(frame):
    features, contour, threshold, info = extract_features_from_image(frame)

    if features is None:
        return "Unknown", 0.0, {}, features, contour, threshold, info

    prediction = model.predict(features)[0]

    probabilities_dict = {}

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]

        for class_name, prob in zip(model.classes_, probabilities):
            probabilities_dict[class_name] = float(prob)

        confidence = float(max(probabilities))
    else:
        confidence = 1.0
        probabilities_dict[prediction] = 1.0

    if confidence < confidence_threshold:
        display_prediction = "Unknown"
    else:
        display_prediction = prediction

    return display_prediction, confidence, probabilities_dict, features, contour, threshold, info

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

def sort_part(label):
    if label not in PART_TO_BIN:
        label = "Unknown"

    bin_name = PART_TO_BIN[label]
    bin_target = BIN_TARGETS[bin_name]

    print("----------------------------------")
    print("Sorting as:", label)
    print("Assigned bin:", bin_name)
    print("----------------------------------")

    pick_part()
    drop_part(bin_target)

# ============================================================
# ACTIVE LEARNING ACTION
# ============================================================

def accept_or_correct(frame, features, label):
    global model

    if features is None:
        print("No usable object features found. Not saving.")
        return

    save_active_example(frame, features, label)

    print("Retraining model with new example...")
    model = train_and_save_model()

    sort_part(label)

# ============================================================
# MAIN LOOP
# ============================================================

camera = SpinnakerCamera(serial_number=camera_serial_number)
camera.start()

print("----------------------------------")
print("Active learning sorter running.")
print("S = accept model prediction and sort")
print("1 = label as perfect, retrain, sort")
print("2 = label as usable, retrain, sort")
print("3 = label as unusable, retrain, sort")
print("Q = quit")
print("----------------------------------")

last_sorted_time = 0

try:
    while True:
        ret, frame = camera.read()

        if not ret:
            continue

        prediction, confidence, probabilities, features, contour, threshold, info = predict_from_frame(frame)

        display = frame.copy()

        if contour is not None and info is not None:
            cv2.drawContours(display, [contour], -1, (0, 255, 0), 2)
            x, y, w, h = info["x"], info["y"], info["w"], info["h"]
            cv2.rectangle(display, (x, y), (x + w, y + h), (255, 0, 0), 2)

        color = (0, 255, 0)

        if prediction == "usable":
            color = (0, 255, 255)
        elif prediction in ["unusable", "Unknown"]:
            color = (0, 0, 255)

        cv2.putText(
            display,
            f"Prediction: {prediction} | Confidence: {confidence * 100:.1f}%",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2
        )

        cv2.putText(
            display,
            "S accept | 1 perfect | 2 usable | 3 unusable | Q quit",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        if info is not None:
            cv2.putText(
                display,
                f"Area:{info['area']:.0f} Bend:{info['bend_ratio_max']:.3f} Frag:{info['fragment_count']} Def:{info['defect_count']}",
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        y_text = 160

        for class_name, prob in probabilities.items():
            cv2.putText(
                display,
                f"{class_name}: {prob * 100:.1f}%",
                (20, y_text),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )
            y_text += 28

        cv2.imshow("Active Learning Golf Tee Sorter", display)
        cv2.imshow("Threshold View", threshold)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        current_time = time.time()

        if current_time - last_sorted_time < seconds_between_sorts:
            continue

        if key == ord("s"):
            if prediction == "Unknown":
                print("Prediction is Unknown. Choose 1, 2, or 3 instead.")
                continue

            print("Human accepted prediction:", prediction)
            accept_or_correct(frame, features, prediction)
            last_sorted_time = time.time()

        elif key == ord("1"):
            print("Human corrected label: perfect")
            accept_or_correct(frame, features, "perfect")
            last_sorted_time = time.time()

        elif key == ord("2"):
            print("Human corrected label: usable")
            accept_or_correct(frame, features, "usable")
            last_sorted_time = time.time()

        elif key == ord("3"):
            print("Human corrected label: unusable")
            accept_or_correct(frame, features, "unusable")
            last_sorted_time = time.time()

finally:
    camera.stop()
    cv2.destroyAllWindows()
    print("Program ended.")