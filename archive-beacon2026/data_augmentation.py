import os
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms

# =========================================================
# CONFIG
# =========================================================

# Input dataset structure:
# dataset/
#     good/
#     bad/

INPUT_DIR = "dataset"
OUTPUT_DIR = "augmented_dataset"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")

# 4 augmentations + 1 original = 5x dataset size
AUGS_PER_IMAGE = 4

# Resize all images to this size
IMAGE_SIZE = (224, 224)

# =========================================================
# AUGMENTATION PIPELINE
# =========================================================

augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        size=IMAGE_SIZE,
        scale=(0.8, 1.0)
    ),

    transforms.RandomHorizontalFlip(p=0.5),

    transforms.RandomRotation(degrees=20),

    transforms.ColorJitter(
        brightness=0.3,
        contrast=0.3,
        saturation=0.3,
        hue=0.05
    ),
])

# =========================================================
# EXTRA AUGMENTATION FUNCTIONS
# =========================================================

def add_noise(img):
    """Add Gaussian noise."""
    arr = np.array(img).astype(np.float32)

    noise = np.random.normal(0, 10, arr.shape)

    arr = arr + noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def random_blur(img):
    """Random Gaussian blur."""
    if random.random() < 0.3:
        radius = random.uniform(0.5, 1.5)
        img = img.filter(ImageFilter.GaussianBlur(radius))

    return img


def random_sharpness(img):
    """Random sharpness adjustment."""
    if random.random() < 0.5:
        enhancer = ImageEnhance.Sharpness(img)
        factor = random.uniform(0.8, 1.5)
        img = enhancer.enhance(factor)

    return img


# =========================================================
# AUGMENT ONE IMAGE
# =========================================================

def augment_image(img):

    # Main torchvision augmentations
    img = augment_transform(img)

    # Optional extra augmentations
    img = random_blur(img)

    img = random_sharpness(img)

    if random.random() < 0.5:
        img = add_noise(img)

    return img


# =========================================================
# PROCESS DATASET
# =========================================================

def process_dataset():

    total_original = 0
    total_augmented = 0

    # Create output folder
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Loop through class folders: good / bad
    for class_name in os.listdir(INPUT_DIR):

        class_input_path = os.path.join(INPUT_DIR, class_name)

        # Skip non-folders
        if not os.path.isdir(class_input_path):
            continue

        print(f"\nProcessing class: {class_name}")

        # Create matching output folder
        class_output_path = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(class_output_path, exist_ok=True)

        # Process images
        for filename in os.listdir(class_input_path):

            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue

            image_path = os.path.join(class_input_path, filename)

            try:
                img = Image.open(image_path).convert("RGB")

            except Exception as e:
                print(f"Could not open {image_path}")
                print(e)
                continue

            total_original += 1

            # Base filename
            base_name = os.path.splitext(filename)[0]

            # =================================================
            # SAVE ORIGINAL (RESIZED)
            # =================================================

            original_img = img.resize(IMAGE_SIZE)

            original_save_path = os.path.join(
                class_output_path,
                f"{base_name}_orig.jpg"
            )

            original_img.save(original_save_path)

            # =================================================
            # CREATE AUGMENTED IMAGES
            # =================================================

            for i in range(AUGS_PER_IMAGE):

                aug_img = augment_image(img)

                save_path = os.path.join(
                    class_output_path,
                    f"{base_name}_aug_{i}.jpg"
                )

                aug_img.save(save_path)

                total_augmented += 1

            print(f"Processed: {filename}")

    # =========================================================
    # SUMMARY
    # =========================================================

    total_final = total_original + total_augmented

    print("\n=================================================")
    print("DONE")
    print("=================================================")

    print(f"Original images:   {total_original}")
    print(f"Augmented images:  {total_augmented}")
    print(f"Final dataset:     {total_final}")

    print(f"\nSaved to: {OUTPUT_DIR}")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    process_dataset()