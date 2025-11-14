import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models, datasets
from PIL import Image
from tqdm import tqdm
import csv # New: for logging

# --- 1. Configuration ---
# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 128 # Updated batch size
EPOCHS = 20 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "vit_b_16"
NUM_WORKERS = 10 # Recommended for NSCC

# Paths
DATA_DIR = "IE4483Dataset/datasets"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
LOG_FILE = "training_log.csv"
MODEL_SAVE_PATH = "best_vit_model.pth"

# --- 2. Data Preparation ---
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
])

# Use ImageFolder for simplicity, as the directory structure matches
train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)
val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

print(f"Using device: {DEVICE}")
print(f"Found {len(train_dataset)} images for training and {len(val_dataset)} for validation.")
print(f"Classes: {train_dataset.class_to_idx}")

# --- 3. Model Setup ---
model = models.get_model(MODEL_NAME, weights='DEFAULT')

# Freeze parameters
for param in model.parameters():
    param.requires_grad = False

# Tweak the classifier
num_features = model.heads.head.in_features
model.heads.head = nn.Linear(num_features, 2)
model.to(DEVICE)

# --- 4. Training ---
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# New: Add LR Scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

def train_one_epoch(loader, model, loss_fn, optimizer):
    model.train()
    loop = tqdm(loader, leave=True)
    running_loss = 0.0

    for batch_idx, (data, targets) in enumerate(loop):
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        scores = model(data)
        loss = loss_fn(scores, targets)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())
        
    return running_loss / len(loader)

def check_accuracy(loader, model, loss_fn):
    model.eval()
    num_correct = 0
    num_samples = 0
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

    accuracy = float(num_correct) / float(num_samples)
    return accuracy, val_loss / len(loader)


if __name__ == "__main__":
    best_accuracy = 0.0
    
    # New: Setup CSV logger
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_accuracy'])
    
    print(f"\nStarting training with {MODEL_NAME}...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(train_loader, model, loss_fn, optimizer)
        val_acc, val_loss = check_accuracy(val_loader, model, loss_fn)
        
        # New: LR Scheduler step
        scheduler.step(val_loss)
        
        # New: Save the best model
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"-> New best model saved with accuracy: {val_acc:.4f}")

        # New: Log to CSV
        with open(LOG_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss, val_acc])
        
        print(
            f"Epoch {epoch+1}/{EPOCHS} -> "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_acc:.4f}"
        )

    print(f"\nTraining finished. Best model saved to {MODEL_SAVE_PATH} with accuracy: {best_accuracy:.4f}")