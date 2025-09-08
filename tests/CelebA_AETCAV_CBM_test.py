import torch
import random
import numpy as np
from AE_TCAV import AETCAV
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.data import CelebAJointConcept
from utils.CBM import ConceptBottleneckModel
from utils.models import AutoEncoderWithClassifier

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


MODEL_PATH = "models/celebA_CBM_1.pth"
DATA_DIR = "../Datasets/CelebA/"
TRAIN_PARAMS_NEW = {
    "epochs": 25,
    "recon_loss_function": torch.nn.MSELoss,
    "cls_loss_function": torch.nn.BCEWithLogitsLoss,
    "learning_rate": 1e-3,
    "alpha": 0.01,
}
B_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALS_PATH = "./vals/AE_TCAV/"
LATENT_DIMS = 256
H, W = (64, 64)
LAST_STAGE = "linear"
POLY_POW = 3

print("Loading Data")
target_attr = "Attractive"
concept_attrs = ["Age", "Gender", "Skin", "Bald"]
transform = transforms.Compose(
    [
        transforms.Resize((H, W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ]
)
train_concept_data = CelebAJointConcept(
    data_dir=DATA_DIR,
    split="train",
    target=target_attr,
    concept=concept_attrs,
    mode="concept",
    transform=transform,
    concept_num=4000,
)
train_concept_loader = DataLoader(train_concept_data, batch_size=B_SIZE)
val_concept_data = CelebAJointConcept(
    data_dir=DATA_DIR,
    split="valid",
    target=target_attr,
    concept=concept_attrs,
    mode="concept",
    transform=transform,
    concept_num=2000,
)
val_concept_loader = DataLoader(val_concept_data, batch_size=B_SIZE)

print("Preparing for Training.")
model = AutoEncoderWithClassifier(
    latent_dim=LATENT_DIMS, num_classes=len(concept_attrs), H=H, W=W
)
checkpoint = torch.load(MODEL_PATH)
main_model = ConceptBottleneckModel(
    len(concept_attrs),
    1,
    pretrained=True,
    task_predictor_type=LAST_STAGE,
    poly_pow=POLY_POW,
).to(DEVICE)
main_model.load_state_dict(checkpoint["model_state_dict"])
interpreter = AETCAV(model, TRAIN_PARAMS_NEW, DEVICE)
interpreter.train(train_concept_loader, val_concept_loader)

print("Getting Attribution")
attributions = {}
for c_num, c_name in enumerate(concept_attrs):
    grads = torch.zeros(len(val_concept_data))
    start = 0
    with torch.no_grad():
        main_model.eval()
        for i, (imgs, concepts) in enumerate(val_concept_loader, 1):
            l = len(imgs)
            imgs = imgs.to(DEVICE)
            curr_grads = interpreter.getAttribution(main_model, imgs, c_num, 0, eps=1)
            grads[start : start + l] = curr_grads
            start += l
    print(
        f"{c_name=},{len(grads)=},{grads.mean()=},{2.00 * grads.std()/(len(grads)**0.5)=}"
    )
