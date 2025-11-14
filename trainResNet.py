import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
from tqdm import tqdm
import csv
import time

# --- 1. Configuration ---

# Hyperparameters

LEARNING_RATE = 1e-4

BATCH_SIZE = 128

EPOCHS = 10

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "resnet50"

NUM_WORKERS = 10      # Adjust for your HPC environment

IMG_SIZE = 224       # Image size for the model input



# Paths

DATA_DIR = "IE4483Dataset/datasets"

LOG_FILE = "resnet_training_log.csv"

MODEL_SAVE_PATH = "best_resnet_model.pth"



# --- 2. Data Preparation ---

IMAGENET_MEAN = [0.485, 0.456, 0.406]

IMAGENET_STD = [0.229, 0.224, 0.225]



# Transforms now use the standard stretch/squish resize method

train_transforms = transforms.Compose([

    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(10),

    transforms.ToTensor(),

    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

])



val_transforms = transforms.Compose([

    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

])



# Use ImageFolder for simplicity, as the directory structure matches

train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_transforms)

val_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=val_transforms)



train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)



class_names = train_dataset.classes

print(f"Using device: {DEVICE}")

print(f"Found {len(train_dataset)} images for training and {len(val_dataset)} for validation.")

print(f"Classes: {train_dataset.class_to_idx}")





# --- 3. Model Setup ---

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)



# Freeze parameters

for param in model.parameters():

    param.requires_grad = False



# Tweak the classifier

num_features = model.fc.in_features

model.fc = nn.Linear(num_features, len(class_names))

model.to(DEVICE)





# --- 4. Training & Evaluation Functions ---

loss_fn = nn.CrossEntropyLoss()



def train_one_epoch(loader, model, optimizer):

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

        loop.set_postfix(loss=loss.item())

        

    return running_loss / len(loader)



def evaluate(loader, model):

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



    accuracy = float(num_correct) / float(num_samples)

    return accuracy, val_loss / len(loader)



# --- 5. Main Execution ---

if __name__ == "__main__":

    best_accuracy = 0.0

    

    # Setup CSV logger

    with open(LOG_FILE, 'w', newline='') as f:

        writer = csv.writer(f)

        writer.writerow(['epoch', 'train_loss', 'val_loss', 'val_accuracy', 'lr'])



    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)



    for epoch in range(EPOCHS):

        lr = optimizer.param_groups[0]['lr']

        train_loss = train_one_epoch(train_loader, model, optimizer)

        val_acc, val_loss = evaluate(val_loader, model)

        scheduler.step(val_loss)

        

        if val_acc > best_accuracy:

            best_accuracy = val_acc

            torch.save(model.state_dict(), MODEL_SAVE_PATH)

            print(f"-> New best model saved with accuracy: {val_acc:.4f}")



        with open(LOG_FILE, 'a', newline='') as f:

            csv.writer(f).writerow([epoch + 1, train_loss, val_loss, val_acc, lr])

        

        print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")



    print(f"\nTraining finished. Best model saved to {MODEL_SAVE_PATH} with accuracy: {best_accuracy:.4f}")
