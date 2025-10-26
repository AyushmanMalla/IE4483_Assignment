import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models, datasets # Corrected import
from PIL import Image
from tqdm import tqdm
import csv

# --- 1. Configuration ---
BATCH_SIZE = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "vit_b_16"
NUM_WORKERS = 10 # Adjust for your HPC environment

# Paths
DATA_DIR = "IE4483Dataset/datasets"
TRAIN_DIR = os.path.join(DATA_DIR, "train") # Needed for class mapping
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_PATH = "best_vit_model.pth"
SUBMISSION_FILE = "submission.csv"

# --- 2. Data Preparation ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Use the same transforms as validation
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# --- New: Custom Dataset for an UNLABELLED, Flat Test Directory ---
class InferenceDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('jpg', 'jpeg', 'png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
            
        # Return the image and its original filename
        return image, os.path.basename(img_path)

# --- Create Datasets and Dataloaders ---
# Get the class mapping from the training directory to ensure predictions are correct
# Corrected: Use datasets.ImageFolder directly
train_ds_for_mapping = datasets.ImageFolder(TRAIN_DIR)
class_to_idx = train_ds_for_mapping.class_to_idx
idx_to_class = {i: cls for cls, i in class_to_idx.items()}

# Create the test dataset using our new inference class
test_dataset = InferenceDataset(TEST_DIR, transform=test_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"Loading model from: {MODEL_PATH}")
print(f"Found {len(test_dataset)} images in the test set.")
print(f"Class mapping used for predictions: {class_to_idx}")

# --- 3. Load Model ---
model = models.get_model(MODEL_NAME, weights=None)
num_features = model.heads.head.in_features
model.heads.head = nn.Linear(num_features, len(class_to_idx))

model.load_state_dict(torch.load(MODEL_PATH))
model.to(DEVICE)
model.eval()

# --- 4. Inference and CSV Generation ---
results = []
with torch.no_grad():
    for images, filenames in tqdm(test_loader, desc="Running Inference"):
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted_indices = torch.max(outputs, 1)
        
        # Convert indices to class labels ('0' or '1')
        predicted_labels = predicted_indices.cpu().numpy()
        
        for i in range(len(filenames)):
            results.append([filenames[i], predicted_labels[i]])

# --- 5. Save Results to CSV ---
with open(SUBMISSION_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image', 'label']) # Write header
    writer.writerows(results)

print(f"\nInference complete. Results saved to {SUBMISSION_FILE}")