import os
import cv2
import numpy as np
from tqdm import tqdm # For a progress bar

# --- Configuration ---
# Set the path to the main dataset directory
data_dir = "IE4483Dataset/datasets" 

# --- Initialization ---
image_counter = 0.0
avg_height = 0.0
avg_width = 0.0
avg_brightness = 0.0
avg_contrast = 0.0
avg_channels = np.array([0.0, 0.0, 0.0]) # B, G, R running averages

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
    for img_path in tqdm(image_paths, desc="Analyzing images"):
        try:
            # Load image using OpenCV
            img = cv2.imread(img_path)
            
            if img is not None:
                height, width, _ = img.shape
                image_counter += 1
                
                # --- Use a running average to prevent overflow ---
                # Dimensions
                avg_height += (height - avg_height) / image_counter
                avg_width += (width - avg_width) / image_counter
                
                # Brightness (mean of grayscale version)
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                avg_brightness += (gray_img.mean() - avg_brightness) / image_counter
                
                # Contrast (std deviation of grayscale version)
                avg_contrast += (gray_img.std() - avg_contrast) / image_counter
                
                # Color Channels (mean of each B, G, R channel)
                channel_means = np.mean(img, axis=(0, 1))
                avg_channels += (channel_means - avg_channels) / image_counter
                
        except Exception as e:
            print(f"Could not process {img_path}: {e}")

    # --- Calculate and Print Final Statistics ---
    if image_counter > 0:
        target_size = (int(avg_height * 0.1), int(avg_width * 0.1))
        
        # Determine dominant color channel
        channel_names = ['Blue', 'Green', 'Red']
        dominant_channel = channel_names[np.argmax(avg_channels)]
        
        print("\n--- Dataset Analysis Complete ---")
        print(f"Total images processed: {int(image_counter)}")
        
        print("\n--- Geometric Properties ---")
        print(f"Average Height: {avg_height:.2f} pixels")
        print(f"Average Width: {avg_width:.2f} pixels")
        print(f"Calculated Target Size (10%): {target_size}")
        
        print("\n--- Color & Brightness Properties ---")
        print(f"Average Brightness (0-255): {avg_brightness:.2f}")
        print(f"Average Contrast (Std Dev): {avg_contrast:.2f}")
        print(f"Average BGR Channel Values: B={avg_channels[0]:.2f}, G={avg_channels[1]:.2f}, R={avg_channels[2]:.2f}")
        print(f"Dominant Color Channel: {dominant_channel}")
        print("-----------------------------------")
    else:
        print("No valid images were processed.")