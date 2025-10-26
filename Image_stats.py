import os
from PIL import Image
from tqdm import tqdm  # Optional: for a progress bar
import cv2
import numpy as np

# Set the path to the main dataset directory
data_dir = "IE4483Dataset/datasets"  #replace with your path if its different

image_counter = 0
avg_height = 0.0
avg_width = 0.0

# --- Get a list of all image files ---
image_paths = []
for split in ['train', 'val']:
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        continue
    for label in os.listdir(split_dir):
        animal_dir = os.path.join(split_dir, label)
        if os.path.isdir(animal_dir):
            for img_file in os.listdir(animal_dir):
                image_paths.append(os.path.join(animal_dir, img_file))

# --- Process images with a progress bar ---
if not image_paths:
    print("No images found in the specified directory structure.")
    target_size = (0, 0)
else:
    # Use tqdm for a helpful progress bar, especially with many files
    for img_path in tqdm(image_paths, desc="Processing images"):
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                image_counter += 1
                # --- Use a running average to prevent overflow ---
                avg_height += (height - avg_height) / image_counter
                avg_width += (width - avg_width) / image_counter
        except Exception as e:
            print(f"Could not read {img_path}: {e}")

    # Calculate the final target size
    target_size = (int(avg_height * 0.1), int(avg_width * 0.1))
    
    print(f"\nTotal images found: {image_counter}")
    print(f"Average Height: {avg_height}")
    print(f"Average Width: {avg_width}")
    print(f"Calculated Target Size: {target_size}")