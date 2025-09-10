# Importing Libraries
import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from utils.data import CelebAJointConcept
from torch.utils.data import DataLoader
from utils.CBM import ConceptBottleneckModel
from utils.ImageNetModels import TRANSFORM_DICT

# Constants
B_SIZE = 512
NUM_EPOCHS = 5
LR = 0.001
DATA_DIR = "../Datasets/CelebA/"
SAVE_DIR = "./models/"
LAST_STAGE = "linear"
ENCODER = "resnet18"
POLY_POW = 3
if LAST_STAGE == "linear":
    MODEL_NAME = "celebA_CBM_1.pth"
else:
    MODEL_NAME = f"celebA_CBM_{LAST_STAGE}_{POLY_POW}.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ALPHA = 0.05
SEED = 0

# Setting up random seeds
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Dataset Definition and init
target_attr = "Attractive"
concepts = ["Age", "Gender", "Skin", "Bald"]
transform = TRANSFORM_DICT[ENCODER]

print("Loading Data....")
train_data = CelebAJointConcept(
    data_dir=DATA_DIR,
    split="train",
    target=target_attr,
    concept=concepts,
    transform=transform,
    concept_num=5000,
)
val_data = CelebAJointConcept(
    data_dir=DATA_DIR,
    split="valid",
    target=target_attr,
    concept=concepts,
    transform=transform,
    concept_num=1000,
)

train_loader = DataLoader(train_data, batch_size=B_SIZE)
val_loader = DataLoader(val_data, batch_size=B_SIZE)


# Model Definition
print("Preparing for training...")

model = ConceptBottleneckModel(
    len(concepts),
    1,
    encoder_name=ENCODER,
    pretrained=True,
    task_predictor_type=LAST_STAGE,
    poly_pow=POLY_POW,
).to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    n = len(train_loader)
    for i, (imgs, concepts, labels) in enumerate(train_loader):
        print(f"\rWorking on batch {i}/{n}", end="")
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1).float()
        concepts = concepts.to(DEVICE)
        _, pred_concepts, pred_labels = model(imgs, return_intermediate=True)
        loss_pred = criterion(pred_labels, labels)
        loss_concepts = criterion(pred_concepts, concepts.float())
        loss = loss_pred + ALPHA * loss_concepts

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")
    model.eval()
    correct_labels = 0
    correct_concepts = 0
    total = 0
    with torch.no_grad():
        for imgs, concepts, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1).float()
            _, pred_concepts, pred_labels = model(imgs, return_intermediate=True)
            concepts = concepts.to(DEVICE)
            pred_labels = (
                torch.sigmoid(pred_labels) > 0.5
            )  # Convert logits to binary predictions
            correct_labels = (pred_labels.float() == labels).sum().item()
            pred_concepts = pred_concepts > 0.5
            correct_concepts += (
                (pred_concepts.float() == concepts).float().mean(axis=1).sum().item()
            )
            total += labels.size(0)
    val_l_acc = correct_labels / total
    val_c_acc = correct_concepts / total
    print(
        f"Validation Label Accuracy: {val_l_acc:.4f}| Validation Concept Accuracy: {val_c_acc:.4f}"
    )

# === Save ===
if not (os.path.isdir(SAVE_DIR)):
    os.makedirs(SAVE_DIR)

save_path = SAVE_DIR + MODEL_NAME

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    },
    save_path,
)

print(f"Model saved to {save_path}")
