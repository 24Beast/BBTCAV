# Importing Libraries
import os
import torch
import random
import numpy as np
from torchvision import transforms
from utils.models import SimpleCNN
from utils.data import CelebAConcept
from torch.utils.data import DataLoader, Subset
from captum.concept import TCAV, Concept
from captum.attr import LayerIntegratedGradients

# Setting Random state
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

# Setting Constants
MODEL_PATH = "models/celebA_CNN.pth"
DATA_DIR = "../Datasets/CelebA/"
B_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALS_PATH = "./vals/TCAV/"
H, W = (64, 64)

# Setting up Path
if not(os.path.isdir(VALS_PATH)):
    os.makedirs(VALS_PATH)


# Helper Function
def getConceptfromDataset(
    dataset: torch.utils.data.Dataset, val: int, name: str, batch_size: int = B_SIZE
) -> Subset:
    idxs = [i for i, (_, c) in enumerate(dataset) if c == val]
    sub_dataset = Subset(dataset, idxs)
    iterator = DataLoader(
        sub_dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: torch.stack([item[0] for item in batch]).to(DEVICE),
    )
    concept = Concept(getConceptfromDataset.num, name, iterator)
    getConceptfromDataset.num += 1
    return concept


getConceptfromDataset.num = 0


# Loading Data
target_attr = "Attractive"
concept_attrs = ["Age", "Gender", "Skin", "Bald", "Fat", "Smiling"]
transform = transforms.Compose(
    [
        transforms.Resize((H, W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ]
)
for num in range(len(concept_attrs)):
    print("Loading Data")
    train_concept_data = CelebAConcept(
        data_dir=DATA_DIR,
        split="train",
        target=target_attr,
        concept=concept_attrs[num],
        mode="concept",
        transform=transform,
        concept_num=2500,
    )
    positive_concept = getConceptfromDataset(
        train_concept_data, val=1, name=concept_attrs[num]
    )
    negative_concept = getConceptfromDataset(train_concept_data, val=0, name="random")
    concepts = [[positive_concept, negative_concept]]
    
    val_concept_data = CelebAConcept(
        data_dir=DATA_DIR,
        split="valid",
        target=target_attr,
        concept=concept_attrs[num],
        mode="labels",
        transform=transform,
        concept_num=1000,
    )
    val_concept_loader = DataLoader(val_concept_data, batch_size=B_SIZE)
    
    print("Loading Model.")
    checkpoint = torch.load(MODEL_PATH)
    model = SimpleCNN().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model_layer = "net.9"
    
    # Initializing TCAV
    tcav_obj = TCAV(
        model=model,
        layers=model_layer,
        layer_attr_method=LayerIntegratedGradients(model, None, multiply_by_inputs=False),
    )
    
    vals = []
    for imgs, _, labels in val_concept_loader:
        if len(labels.shape) == 1:
            labels = labels.reshape(-1, 1)
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        scores = tcav_obj.interpret(inputs=imgs, experimental_sets=concepts)
        vals.append(scores[f"{2*num}-{(2*num)+1}"][model_layer]['magnitude'].cpu())
    vals = np.array(vals)
    
    np.save(VALS_PATH + f"CelebA_SimpleCNN_{target_attr}_{concept_attrs[num]}.npy", vals)
    
