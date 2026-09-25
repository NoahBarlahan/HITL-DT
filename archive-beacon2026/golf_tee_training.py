# ============================================================
# GOLF TEE DEFECT FEATURE MODEL TRAINING SCRIPT
# Learns:
#     perfect
#     usable
#     unusable
#
# Features include:
#     size
#     shape
#     straightness
#     bend/curvature
#     broken-fragment count
#     convexity defects
#     edge/scuff texture
# ============================================================

import os
import cv2
import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ============================================================
# SETTINGS
# ============================================================

dataset_folder_name = "Golf_tees_augmented"
model_folder_name = "trained_model"
model_file_name = "shape_size_classifier.pkl"

class_names = ["perfect", "usable", "unusable"]

valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

random_seed = 123
validation_percent = 0.2

minimum_object_area = 800
minimum_fragment_area = 150

# ============================================================
# PATHS
# ============================================================

script_folder = os.path.dirname(os.path.abspath(__file__))

dataset_folder = os.path.join(script_folder, dataset_folder_name)

model_folder = os.path.join(script_folder, model_folder_name)
os.makedirs(model_folder, exist_ok=True)

model_path = os.path.join(model_folder, model_file_name)

if not os.path.exists(dataset_folder):
    raise Exception(
        "Dataset folder not found.\n\n"
        "Expected:\n"
        + dataset_folder_name
        + "/\n"
        "    perfect/\n"
        "    usable/\n"
        "    unusable/"
    )

# ============================================================
# PREPROCESSING
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
        mask = (
            (major_projection >= bins[i]) &
            (major_projection < bins[i + 1])
        )

        section = minor_projection[mask]

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

    return width_mean, width_std, width_variation_ratio, safe_div(width_min, width_max)

# ============================================================
# MAIN FEATURE EXTRACTION
# ============================================================

def extract_golf_tee_features(image_path):
    image = cv2.imread(image_path)

    if image is None:
        return None

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
        return None

    largest_contour = max(valid_contours, key=cv2.contourArea)
    largest_area = cv2.contourArea(largest_contour)

    if largest_area < minimum_object_area:
        return None

    total_area = sum(cv2.contourArea(c) for c in valid_contours)

    fragment_count = len(valid_contours)
    largest_area_ratio = safe_div(largest_area, total_area)

    contour = largest_contour

    area = cv2.contourArea(contour)
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

    return features

# ============================================================
# LOAD DATASET
# ============================================================

X = []
y = []
image_paths_used = []

print("----------------------------------")
print("Reading dataset and extracting golf tee defect features...")
print("----------------------------------")

for class_name in class_names:
    class_folder = os.path.join(dataset_folder, class_name)

    if not os.path.exists(class_folder):
        raise Exception(f"Missing folder: {class_folder}")

    count = 0
    skipped = 0

    for file_name in os.listdir(class_folder):
        ext = os.path.splitext(file_name)[1].lower()

        if ext not in valid_extensions:
            continue

        image_path = os.path.join(class_folder, file_name)

        features = extract_golf_tee_features(image_path)

        if features is None:
            skipped += 1
            continue

        X.append(features)
        y.append(class_name)
        image_paths_used.append(image_path)

        count += 1

    print(class_name)
    print("  Used images:", count)
    print("  Skipped images:", skipped)

print("----------------------------------")

if len(X) == 0:
    raise Exception("No usable images found.")

X = np.array(X, dtype=np.float32)
y = np.array(y)

# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=validation_percent,
    random_state=random_seed,
    stratify=y
)

print("Training samples:", len(X_train))
print("Validation samples:", len(X_val))
print("----------------------------------")

# ============================================================
# TRAIN MODEL
# ============================================================

random_forest = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=random_seed
)

extra_trees = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=random_seed
)

model = VotingClassifier(
    estimators=[
        ("rf", random_forest),
        ("et", extra_trees)
    ],
    voting="soft"
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", model)
])

pipeline.fit(X_train, y_train)

# ============================================================
# VALIDATE MODEL
# ============================================================

y_pred = pipeline.predict(X_val)

accuracy = accuracy_score(y_val, y_pred)

print("Validation accuracy:", round(accuracy, 3))
print("----------------------------------")
print("Confusion matrix:")
print(confusion_matrix(y_val, y_pred, labels=class_names))
print("----------------------------------")
print(classification_report(y_val, y_pred))
print("----------------------------------")

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
# SAVE MODEL
# ============================================================

model_data = {
    "model_type": "shape_size_classifier",
    "model_version": "golf_tee_defect_features_v2",
    "class_names": class_names,
    "feature_names": feature_names,
    "minimum_object_area": minimum_object_area,
    "minimum_fragment_area": minimum_fragment_area,
    "model": pipeline
}

with open(model_path, "wb") as f:
    pickle.dump(model_data, f)

print("Golf tee defect model saved to:")
print(model_path)
print("----------------------------------")
print("Feature count:", len(feature_names))
print("----------------------------------")