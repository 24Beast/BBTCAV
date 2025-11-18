# Importing Libraries
import os
import yaml
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from utils.data import CelebAJointConcept
from torch.utils.data import DataLoader
from utils.CBM import ConceptBottleneckModel
from utils.ImageNetModels import TRANSFORM_DICT
from utils.debug import set_seed


# Parser for args
parser = argparse.ArgumentParser(description="AE-TCAV Runner with YAML Config")
parser.add_argument(
    "--config", type=str, required=True, help="Path to YAML configuration file"
)
args = parser.parse_args()

with open(args.config, "r") as f:
    config = yaml.safe_load(f)

print(f"{config=}")

# Constants
B_SIZE = config["training"].get("batch_size", 512)
NUM_EPOCHS = config["training"].get("num_epochs", 5)
LR_C = config["training"].get("lr_c", 0.001)
LR_T = config["training"].get("lr_t", 0.001)
NUM_C = config["training"].get("num_c", 0)
ALPHA = config["training"].get("alpha", 0.5)
DATA_DIR = config["data"].get("data_dir", "../Datasets/CelebA/")
SAVE_DIR = config["model"].get("save_dir", "./models/")
LAST_STAGE = config["model"].get("last_stage", "linear")
ENCODER = config["model"].get("encoder", "resnet18")
POLY_POW = 1
if LAST_STAGE == "linear":
    MODEL_NAME = f"celebA_CBM_linear_{NUM_EPOCHS}_{ALPHA:.3f}_{NUM_C}.pth"
else:
    POLY_POW = config["model"].get("poly_pow", 3)
    MODEL_NAME = (
        f"celebA_CBM_{LAST_STAGE}_{POLY_POW}_{NUM_EPOCHS}_{ALPHA:.3f}_{NUM_C}.pth"
    )
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TODO: Add some kind of scheduler for ALPHA, we want it to decrease from 1->0 as training progresses.
SEED = config.get("seed", 0)

# Setting up random seeds
set_seed(SEED)

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
optimizer_C = optim.Adam(model.concept_predictor.parameters(), lr=LR_C)
optimizer_T = optim.Adam(model.task_predictor.parameters(), lr=LR_T)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    n = len(train_loader)
    if epoch < NUM_C:
        curr_alpha = 1.0
    else:
        curr_alpha = ALPHA
    for i, (imgs, concepts, labels) in enumerate(train_loader):
        optimizer_C.zero_grad()
        optimizer_T.zero_grad()
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1).float()
        concepts = concepts.to(DEVICE)
        _, pred_concepts, pred_labels = model(imgs, return_intermediate=True)
        loss_pred = criterion(pred_labels, labels)
        loss_concepts = criterion(pred_concepts, concepts.float())
        loss = ((1 - curr_alpha) * loss_pred) + (curr_alpha * loss_concepts)
        loss.backward()
        optimizer_C.step()
        optimizer_T.step()
        print(
            f"\rWorking on batch {i+1}/{n}, prev_loss = {loss.item()}, {loss_concepts.item()=}, {loss_pred.item()=}",
            end="",
        )

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
        "optimizer_C_state_dict": optimizer_C.state_dict(),
        "optimizer_T_state_dict": optimizer_T.state_dict(),
    },
    save_path,
)

print(f"Model saved to {save_path}")
