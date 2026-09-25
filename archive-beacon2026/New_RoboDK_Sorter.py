# ============================================================
# PYTORCH SIMILARITY MODEL + ROBODK SORTING SCRIPT
# SPINNAKER / POINT GREY CAMERA VERSION
# ROBOT ACTS ONLY WHEN SPACEBAR IS PRESSED
# ============================================================

import os
import time
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms, models

from robodk.robolink import *
from robodk.robomath import *

import PySpin

# ============================================================
# SETTINGS
# ============================================================

model_folder_name = "trained_model"
model_file_name = "pytorch_similarity_reference_model.pth"

seconds_between_sorts = 3
image_size = 224

similarity_threshold_override = 0.75

camera_serial_number = None

# ============================================================
# FIND TRAINED MODEL
# ============================================================

script_folder = os.path.dirname(os.path.abspath(__file__))
model_folder = os.path.join(script_folder, model_folder_name)
model_path = os.path.join(model_folder, model_file_name)

if not os.path.exists(model_path):
    raise Exception(f"Could not find model:\n{model_path}")

print("----------------------------------")
print("Similarity model found at:")
print(model_path)
print("----------------------------------")

# ============================================================
# SELECT DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)
print("----------------------------------")

# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(model_path, map_location=device)

if checkpoint.get("model_type") != "similarity_reference_model":
    raise Exception("This is not a similarity reference model.")

class_names = checkpoint["class_names"]
number_of_classes = checkpoint["number_of_classes"]
class_reference_profiles = checkpoint["class_reference_profiles"]

if similarity_threshold_override is None:
    similarity_threshold = checkpoint["similarity_threshold"]
else:
    similarity_threshold = similarity_threshold_override

for class_name in class_reference_profiles:
    class_reference_profiles[class_name] = class_reference_profiles[class_name].to(device)

print("Loaded similarity reference model.")
print("Classes found:")

for class_name in class_names:
    print(class_name)

print("Similarity threshold:", similarity_threshold)
print("----------------------------------")

# ============================================================
# CLASS TO BIN ASSIGNMENT
# ============================================================

PART_TO_BIN = {}

if number_of_classes >= 1:
    PART_TO_BIN[class_names[0]] = "BIN_1_DROP"

if number_of_classes >= 2:
    PART_TO_BIN[class_names[1]] = "BIN_2_DROP"

PART_TO_BIN["Unknown"] = "BIN_3_DROP"

print("Class-to-bin assignments:")

for part_name, bin_name in PART_TO_BIN.items():
    print(part_name, "->", bin_name)

print("----------------------------------")

# ============================================================
# IMAGE TRANSFORM
# ============================================================

camera_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# FEATURE EXTRACTOR
# ============================================================

resnet_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

feature_extractor = nn.Sequential(*list(resnet_model.children())[:-1])
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

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

        camera_count = self.cam_list.GetSize()

        if camera_count == 0:
            self.cam_list.Clear()
            self.system.ReleaseInstance()
            raise Exception("No Point Grey / FLIR cameras found.")

        if self.serial_number is None:
            self.camera = self.cam_list.GetByIndex(0)
        else:
            self.camera = self.cam_list.GetBySerial(self.serial_number)

        self.camera.Init()

        nodemap = self.camera.GetNodeMap()

        acquisition_mode = PySpin.CEnumerationPtr(
            nodemap.GetNode("AcquisitionMode")
        )

        continuous_mode = acquisition_mode.GetEntryByName("Continuous")

        acquisition_mode.SetIntValue(continuous_mode.GetValue())

        self.camera.BeginAcquisition()

        print("Spinnaker camera started.")
        print("----------------------------------")

    def read(self):
        image_result = self.camera.GetNextImage(1000)

        if image_result.IsIncomplete():
            print("Incomplete image. Status:", image_result.GetImageStatus())
            image_result.Release()
            return False, None

        try:
            frame = image_result.GetNDArray()
            frame = frame.copy()

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
            print("Camera frame conversion error:")
            print(e)
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
            print("Camera cleanup error:")
            print(e)

# ============================================================
# ROBODK CONNECTION
# ============================================================

RDK = Robolink()
robot = RDK.Item('', ITEM_TYPE_ROBOT)

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
        raise Exception(f"Target '{target_name}' not found.")

check_target(home, "Home")
check_target(approach, "Approach")
check_target(middle, "Middle")
check_target(pick_app, "_PICK_APP")
check_target(pick, "_PICK")
check_target(bin_1_drop, "BIN_1_DROP")
check_target(bin_2_drop, "BIN_2_DROP")
check_target(bin_3_drop, "BIN_3_DROP")

print("Connected to RoboDK successfully.")
print("Robot found:", robot.Name())
print("----------------------------------")

# ============================================================
# ROBOT FUNCTIONS
# ============================================================

def close_gripper():
    print("Gripper close")

def open_gripper():
    print("Gripper open")

def pick_part():
    print("Moving to pick position...")
    robot.MoveJ(home)
    robot.MoveJ(approach)
    robot.MoveJ(pick_app)
    robot.MoveL(pick)
    close_gripper()
    robot.MoveL(pick_app)

def drop_part(bin_target):
    print("Moving to bin:", bin_target.Name())
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
# ML FUNCTIONS
# ============================================================

def extract_feature_vector_from_frame(frame_bgr):
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    image_tensor = camera_transform(pil_image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        features = feature_extractor(image_tensor)

    features = features.view(features.size(0), -1)
    features = F.normalize(features, p=2, dim=1)

    return features.squeeze(0)

def identify_part_from_frame(frame_bgr):
    test_feature = extract_feature_vector_from_frame(frame_bgr)

    similarity_scores = {}

    for class_name, class_profile in class_reference_profiles.items():
        similarity = torch.dot(test_feature, class_profile).item()
        similarity_scores[class_name] = similarity

    predicted_name = max(similarity_scores, key=similarity_scores.get)
    similarity_score = similarity_scores[predicted_name]

    strong_prediction = similarity_score >= similarity_threshold

    if predicted_name not in PART_TO_BIN:
        sorting_name = "Unknown"
        strong_prediction = False
    elif strong_prediction:
        sorting_name = predicted_name
    else:
        sorting_name = "Unknown"

    return (
        sorting_name,
        predicted_name,
        similarity_score,
        strong_prediction,
        similarity_scores
    )

def print_sorting_information(
    predicted_name,
    similarity_percent,
    strong_prediction,
    sorting_name,
    similarity_scores
):
    print("----------------------------------")
    print("Best model match:", predicted_name)
    print("Similarity:", str(similarity_percent) + "%")
    print("Similarity threshold:", str(round(similarity_threshold * 100, 1)) + "%")
    print("Strong prediction:", strong_prediction)
    print("Sorting as:", sorting_name)
    print("All similarity scores:")

    for class_name, score in similarity_scores.items():
        print("   " + class_name + ":", round(score, 3))

    print("----------------------------------")

# ============================================================
# MAIN LOOP
# ============================================================

camera = SpinnakerCamera(serial_number=camera_serial_number)
camera.start()

print("Manual trigger mode is ON.")
print("Robot will only sort when SPACEBAR is pressed.")
print("Press q to quit.")
print("----------------------------------")

last_sorted_time = 0

try:
    while True:
        ret, frame = camera.read()

        if not ret:
            print("Camera read failed.")
            continue

        (
            sorting_name,
            predicted_name,
            similarity_score,
            strong_prediction,
            similarity_scores
        ) = identify_part_from_frame(frame)

        similarity_percent = round(similarity_score * 100, 1)

        if strong_prediction:
            display_sorting_name = sorting_name
            label = (
                "Ready: "
                + display_sorting_name
                + " | Match: "
                + predicted_name
                + " | Similarity: "
                + str(similarity_percent)
                + "%"
            )
            color = (0, 255, 0)
        else:
            display_sorting_name = "Unknown"
            label = (
                "Ready: Unknown"
                + " | Best match: "
                + predicted_name
                + " | Similarity: "
                + str(similarity_percent)
                + "%"
            )
            color = (0, 0, 255)

        cv2.putText(
            frame,
            label,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )

        instruction_text = "Press SPACE to sort current part | Press q to quit"

        cv2.putText(
            frame,
            instruction_text,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        threshold_text = "Threshold: " + str(round(similarity_threshold * 100, 1)) + "%"

        cv2.putText(
            frame,
            threshold_text,
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "Spinnaker PyTorch Similarity Sorting - Spacebar Trigger",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        current_time = time.time()

        enough_time_passed = (
            current_time - last_sorted_time >= seconds_between_sorts
        )

        if key == 32:
            if not enough_time_passed:
                print("Wait before sorting again.")
                continue

            print_sorting_information(
                predicted_name,
                similarity_percent,
                strong_prediction,
                sorting_name,
                similarity_scores
            )

            sort_part(sorting_name)

            last_sorted_time = time.time()

finally:
    camera.stop()
    cv2.destroyAllWindows()
    print("Program ended.")