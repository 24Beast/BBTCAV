# Importing Libraries
import os
import torch
import random
import numpy as np
import torch.nn as nn
from BBTCAV import BBTCAV
from torchvision import transforms
from utils.data import CelebAConcept
from torch.utils.data import DataLoader

# Random Seeding
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
# torch.use_deterministic_algorithms(True)

# Setting Constants
MODEL_PATH = "models/celebA_CNN.pth"
DATA_DIR = "../Datasets/CelebA/"
TRAIN_PARAMS = {
    "epochs": 50,
    "loss_function": torch.nn.BCEWithLogitsLoss,
    "learning_rate": 1e-2,
    "post_activation": torch.nn.Sigmoid(),
}
B_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALS_PATH = "./vals/"
CONCEPTS = ["Age", "Gender", "Skin", "Bald"]


# Setting up Path
if not (os.path.isdir(VALS_PATH)):
    os.makedirs(VALS_PATH)


# Loading Data
for concept_attr in CONCEPTS:
    print("Loading Data")
    target_attr = "Attractive"
    #    concept_attr = "Gender"
    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )
    train_concept_data = CelebAConcept(
        data_dir=DATA_DIR,
        split="train",
        target=target_attr,
        concept=concept_attr,
        mode="concept",
        transform=transform,
        concept_num=2500,
    )
    train_concept_loader = DataLoader(train_concept_data, batch_size=B_SIZE)
    val_concept_data = CelebAConcept(
        data_dir=DATA_DIR,
        split="valid",
        target=target_attr,
        concept=concept_attr,
        mode="concept",
        transform=transform,
        concept_num=1000,
    )
    val_concept_loader = DataLoader(val_concept_data, batch_size=B_SIZE)
    # test_data = CelebAConcept(DATA_DIR, split="test", transform = transform, target = target_attr)
    # test_loader = DataLoader(test_data, batch_size= B_SIZE)

    # Concept Model
    print("Preparing for Training.")

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
                nn.Sigmoid(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            return self.net(x)

    model = SimpleCNN()

    # Init Interpretability
    interpreter = BBTCAV(model, TRAIN_PARAMS, DEVICE)
    interpreter.train(train_concept_loader, val_concept_loader)

    # Loading Main model
    checkpoint = torch.load(MODEL_PATH)
    main_model = SimpleCNN().to(DEVICE)
    main_model.load_state_dict(checkpoint["model_state_dict"])

    # Get Scores
    pred_list = torch.zeros(len(val_concept_data))
    score_list = torch.zeros(len(val_concept_data))
    start = 0
    with torch.no_grad():
        model.eval()
        for i, (imgs, concepts) in enumerate(val_concept_loader, 1):
            print(f"\rWorking on Batch {i}/{len(val_concept_loader)}", end="")
            l = len(imgs)
            imgs = imgs.to(DEVICE)
            preds = torch.sigmoid(main_model(imgs)).detach().cpu()
            pred_list[start : start + l] = preds.reshape(-1)
            scores = interpreter.getScores(imgs).detach().cpu()
            score_list[start : start + l] = scores.reshape(-1)

    vals = interpreter.calcAttribution(pred_list, score_list)

    np.save(VALS_PATH + f"CelebA_SimpleCNN_{target_attr}_{concept_attr}.npy", vals)
