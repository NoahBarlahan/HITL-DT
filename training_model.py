# ============================================================
# PYTORCH SIMILARITY-BASED TRAINING SCRIPT
# Uses generated augmented data:
# augmented_dataset/
#     good/
#     bad/
# ============================================================

import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets, transforms, models

# ============================================================
# SETTINGS
# ============================================================

dataset_folder_name = "augmented_dataset"

model_folder_name = "trained_model"
model_file_name = "pytorch_similarity_reference_model.pth"

max_images_per_class = 100000
image_size = 224
validation_percent = 0.2
random_seed = 123

valid_extensions = [".jpg", ".jpeg", ".png", ".bmp"]

similarity_threshold = 0.70

# Since the data is already augmented, keep this as 1
augmentation_passes_per_image = 1

# ============================================================
# FIND DATASET FOLDER
# ============================================================

script_folder = os.path.dirname(os.path.abspath(__file__))
dataset_folder = os.path.join(script_folder, dataset_folder_name)

if not os.path.exists(dataset_folder):
    raise Exception(
        "The augmented_dataset folder was not found.\n\n"
        "Make sure your folder structure is:\n"
        "Project/\n"
        "    train_similarity_model.py\n"
        "    augmented_dataset/\n"
        "        good/\n"
        "        bad/\n"
    )

print("----------------------------------")
print("Using dataset folder:")
print(dataset_folder)
print("----------------------------------")

# ============================================================
# SELECT DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print("----------------------------------")

# ============================================================
# IMAGE TRANSFORMS
# ============================================================

reference_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

validation_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ============================================================
# LOAD DATASET STRUCTURE
# ============================================================

folder_dataset = datasets.ImageFolder(root=dataset_folder)

class_names = folder_dataset.classes
number_of_classes = len(class_names)

if number_of_classes < 1:
    raise Exception("You need at least one class folder inside augmented_dataset.")

print("Classes found:")
for class_name in class_names:
    print(class_name)
print("----------------------------------")

# ============================================================
# SELECT IMAGE PATHS PER CLASS
# ============================================================

class_image_paths = {class_index: [] for class_index in range(number_of_classes)}

for image_path, class_index in folder_dataset.samples:

    file_extension = os.path.splitext(image_path)[1].lower()

    if file_extension not in valid_extensions:
        continue

    if len(class_image_paths[class_index]) >= max_images_per_class:
        continue

    class_image_paths[class_index].append(image_path)

for class_index, image_paths in class_image_paths.items():
    if len(image_paths) < 2:
        raise Exception(
            f"Class '{class_names[class_index]}' needs at least 2 images."
        )

print("Images selected for this run:")
for class_index, image_paths in class_image_paths.items():
    print(class_names[class_index] + ":", len(image_paths))
print("----------------------------------")

# ============================================================
# SPLIT EACH CLASS INTO REFERENCE AND VALIDATION IMAGES
# ============================================================

random.seed(random_seed)

reference_image_paths = {class_index: [] for class_index in range(number_of_classes)}
validation_image_paths = {class_index: [] for class_index in range(number_of_classes)}

for class_index, image_paths in class_image_paths.items():

    shuffled_paths = image_paths.copy()
    random.shuffle(shuffled_paths)

    validation_count = int(len(shuffled_paths) * validation_percent)

    if validation_count == 0:
        validation_count = 1

    validation_image_paths[class_index] = shuffled_paths[:validation_count]
    reference_image_paths[class_index] = shuffled_paths[validation_count:]

    if len(reference_image_paths[class_index]) == 0:
        raise Exception(
            f"Class '{class_names[class_index]}' does not have enough images "
            "after the validation split."
        )

print("Reference and validation split:")
for class_index in range(number_of_classes):
    print(
        class_names[class_index] + ":",
        len(reference_image_paths[class_index]),
        "reference images,",
        len(validation_image_paths[class_index]),
        "validation images"
    )
print("----------------------------------")

# ============================================================
# BUILD FEATURE EXTRACTOR
# ============================================================

resnet_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

feature_extractor = nn.Sequential(*list(resnet_model.children())[:-1])
feature_extractor = feature_extractor.to(device)
feature_extractor.eval()

# ============================================================
# HELPER FUNCTION
# ============================================================

def extract_feature_vector(image_path, transform):
    image = Image.open(image_path).convert("RGB")

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        features = feature_extractor(image_tensor)

    features = features.view(features.size(0), -1)
    features = F.normalize(features, p=2, dim=1)

    return features.squeeze(0).cpu()

# ============================================================
# BUILD CLASS REFERENCE PROFILES
# ============================================================

class_reference_profiles = {}

print("Building class reference profiles...")

for class_index in range(number_of_classes):

    class_name = class_names[class_index]
    feature_vectors = []

    for image_path in reference_image_paths[class_index]:

        feature_vector = extract_feature_vector(
            image_path=image_path,
            transform=reference_transform
        )

        feature_vectors.append(feature_vector)

    stacked_features = torch.stack(feature_vectors)

    class_profile = torch.mean(stacked_features, dim=0)
    class_profile = F.normalize(
        class_profile.unsqueeze(0),
        p=2,
        dim=1
    ).squeeze(0)

    class_reference_profiles[class_name] = class_profile

    print(
        class_name,
        "profile built using",
        len(feature_vectors),
        "feature samples."
    )

print("----------------------------------")

# ============================================================
# VALIDATION CHECK
# ============================================================

print("Checking reference profiles on validation images...")

correct_predictions = 0
total_validation_images = 0
all_true_class_scores = []

for true_class_index in range(number_of_classes):

    true_class_name = class_names[true_class_index]

    for image_path in validation_image_paths[true_class_index]:

        test_feature = extract_feature_vector(
            image_path=image_path,
            transform=validation_transform
        )

        similarity_scores = {}

        for class_name, class_profile in class_reference_profiles.items():
            similarity = torch.dot(test_feature, class_profile).item()
            similarity_scores[class_name] = similarity

        predicted_class_name = max(similarity_scores, key=similarity_scores.get)
        best_similarity = similarity_scores[predicted_class_name]
        true_class_similarity = similarity_scores[true_class_name]

        all_true_class_scores.append(true_class_similarity)
        total_validation_images += 1

        if predicted_class_name == true_class_name:
            correct_predictions += 1

        print("----------------------------------")
        print("Validation image:", os.path.basename(image_path))
        print("True class:", true_class_name)
        print("Best match:", predicted_class_name)
        print("Best similarity:", round(best_similarity, 3))
        print("True class similarity:", round(true_class_similarity, 3))

        if best_similarity >= similarity_threshold:
            print("Decision using threshold:", predicted_class_name)
        else:
            print("Decision using threshold: Unknown")

average_true_similarity = sum(all_true_class_scores) / len(all_true_class_scores)

print("----------------------------------")
print("Validation summary:")
print("Total validation images:", total_validation_images)
print("Average true-class similarity:", round(average_true_similarity, 3))

if number_of_classes > 1:
    validation_accuracy = correct_predictions / total_validation_images
    print("Closest-profile accuracy:", round(validation_accuracy, 3))
else:
    print("Closest-profile accuracy is not meaningful with only one class.")

print("Current similarity threshold:", similarity_threshold)
print("----------------------------------")

# ============================================================
# SAVE MODEL
# ============================================================

model_folder = os.path.join(script_folder, model_folder_name)
os.makedirs(model_folder, exist_ok=True)

model_path = os.path.join(model_folder, model_file_name)

torch.save({
    "model_type": "similarity_reference_model",
    "class_names": class_names,
    "number_of_classes": number_of_classes,
    "image_size": image_size,
    "max_images_per_class": max_images_per_class,
    "similarity_threshold": similarity_threshold,
    "augmentation_passes_per_image": augmentation_passes_per_image,
    "dataset_folder_name": dataset_folder_name,
    "class_reference_profiles": class_reference_profiles
}, model_path)

print("Similarity reference model complete.")
print("Model saved to:")
print(model_path)
print("----------------------------------")