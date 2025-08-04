# Importing Libraries
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from utils.data import CelebAConcept
from torch.utils.data import DataLoader


# Constants
B_SIZE = 2048
NUM_EPOCHS = 5
LR = 0.001
DATA_DIR = "../Datasets/CelebA/"
SAVE_DIR = "./models/"
MODEL_NAME = "celebA_CNN.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset Definition and init
target_attr = "Attractive"
concept_attr = "Gender"
transform = transforms.Compose(
    [
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ]
)

print("Loading Data....")
train_data = CelebAConcept(
    DATA_DIR,
    split="train",
    target=target_attr,
    concept=concept_attr,
    transform=transform,
)
val_data = CelebAConcept(
    DATA_DIR,
    split="valid",
    target=target_attr,
    concept=concept_attr,
    transform=transform,
)
train_loader = DataLoader(train_data, batch_size=B_SIZE)
val_loader = DataLoader(val_data, batch_size=B_SIZE)


# Model Definition
print("Preparing for training...")


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


model = SimpleCNN().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(NUM_EPOCHS):
    model.train()
    total_loss = 0
    n = len(train_loader)
    for i, (imgs, labels, concepts) in enumerate(train_loader):
        print(f"\rWorking on batch {i}/{n}", end="")
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1).float()
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels, concepts in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE).unsqueeze(1).float()
            outputs = model(imgs)
            preds = torch.sigmoid(outputs) > 0.5  # Convert logits to binary predictions
            correct += (preds.float() == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total
    print(f"Validation Accuracy: {val_acc:.4f}")

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
