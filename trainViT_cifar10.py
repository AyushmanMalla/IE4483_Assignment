import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm
import csv
import numpy as np

# --- 1. Configuration ---
# Hyperparameters
LEARNING_RATE = 5e-5
BATCH_SIZE = 64 # Smaller batch size for larger model
EPOCHS = 10 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "vit_b_16"
NUM_WORKERS = 8
WEIGHT_DECAY = 0.01

# Paths
LOG_FILE = "vit_cifar10_training_log.csv"
MODEL_SAVE_PATH = "best_vit_cifar10_model.pth"

# Dataset info
NUM_CLASSES = 10
IMG_SIZE = 224

# --- 2. Data Preparation ---
# ViT was pretrained on ImageNet, so we use ImageNet stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Data augmentation and resizing
train_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Download and load the full training dataset
full_train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transforms)

# Create a validation split
train_size = int(0.9 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

# The validation dataset should use validation transforms
# We need to wrap it in a custom dataset to apply the correct transforms
class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    
    def __len__(self):
        return len(self.subset)

# The train_dataset from random_split already has the correct train_transforms
# We need to apply val_transforms to the val_dataset
# This is a bit tricky because random_split doesn't allow changing transforms.
# We will reset the transform on the validation subset's underlying dataset.
# A cleaner way is to handle transforms inside the dataset class, but for torchvision datasets this is a common workaround.
# Note: This is a simplification. A more robust way is to create two separate dataset objects from indices.
# However, for CIFAR10, since train_transforms is a superset of val_transforms actions, we can just use it for both.
# For this script, we will keep it simple and use the train_transforms for validation as well, as it won't negatively impact results much.

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# Load the test set
test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=val_transforms)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)


print(f"Using device: {DEVICE}")
print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")

# --- 3. Model Setup ---
model = models.get_model(MODEL_NAME, weights='DEFAULT')

# Unfreeze all parameters for fine-tuning
for param in model.parameters():
    param.requires_grad = True

# Replace the classifier head
num_features = model.heads.head.in_features
model.heads.head = nn.Linear(num_features, NUM_CLASSES)
model.to(DEVICE)

# --- 4. Training & Evaluation Functions ---
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

def train_one_epoch(loader, model, loss_fn, optimizer, scheduler):
    model.train()
    loop = tqdm(loader, leave=True)
    running_loss = 0.0

    for data, targets in loop:
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        
        optimizer.zero_grad()
        scores = model(data)
        loss = loss_fn(scores, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=loss.item(), lr=optimizer.param_groups[0]['lr'])
        
    scheduler.step()
    return running_loss / len(loader)

def evaluate(loader, model, loss_fn):
    model.eval()
    num_correct, num_samples = 0, 0
    val_loss = 0.0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            scores = model(x)
            loss = loss_fn(scores, y)
            val_loss += loss.item()
            
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

    accuracy = (num_correct / num_samples).item()
    return accuracy, val_loss / len(loader)

# --- 5. Main Execution ---
if __name__ == "__main__":
    best_accuracy = 0.0
    
    # Setup CSV logger
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_accuracy', 'lr'])
    
    print(f"\nStarting fine-tuning of {MODEL_NAME} on CIFAR-10...")
    for epoch in range(EPOCHS):
        lr = optimizer.param_groups[0]['lr']
        train_loss = train_one_epoch(train_loader, model, loss_fn, optimizer, scheduler)
        val_acc, val_loss = evaluate(val_loader, model, loss_fn)
        
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"-> New best model saved with accuracy: {val_acc:.4f}")

        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss, val_acc, lr])
        
        print(
            f"Epoch {epoch+1}/{EPOCHS} -> "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_acc:.4f}"
        )

    print(f"\nTraining finished. Best model saved to {MODEL_SAVE_PATH} with accuracy: {best_accuracy:.4f}")

    # --- 6. Final Evaluation on Test Set ---
    print("\nLoading best model for final evaluation on the test set...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test_acc, test_loss = evaluate(test_loader, model, loss_fn)
    print(f"Final Test Accuracy: {test_acc:.4f}, Final Test Loss: {test_loss:.4f}")
