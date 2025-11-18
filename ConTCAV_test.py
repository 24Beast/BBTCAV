# Importing Libraries
import torch
from torchvision import transforms
from ConTCAV import ConstrastiveTCAV
from utils.losses import MixedReconLoss
from torch.utils.data import DataLoader
from utils.data import CelebAJointConcept
from utils.CBM import ConceptBottleneckModel
from utils.models import SimpleCNN, AutoEncoder
from utils.debug import visualize, set_seed, PCA_vis
from utils.ImageNetModels import TRANSFORM_DICT

set_seed(0)

MODEL_PATH = "./models/CBM/celebA_CBM_linear_10_0.500_2.pth"
MODEL_TYPE = "CBM"
DATA_DIR = "../Datasets/CelebA/"
CONCEPTS = ["Age", "Gender", "Skin", "Bald"]
TRAIN_PARAMS = {
    "epochs": 100,
    "recon_loss_function": MixedReconLoss(
        alpha=0.25, beta=0.50, gamma=0.25
    ),  # torch.nn.MSELoss,
    "learning_rate": 1e-3,
    "alpha": 0.1,
    "Num_Concepts": len(CONCEPTS),
}
B_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VALS_PATH = "./vals/"
LATENT_DIMS = 2048
H, W = (64, 64)

save_dir = "./models/ConTCAV_models/epochs_100_alpha_0.05_lr_0.0050"
if MODEL_TYPE == "CBM":
    ENCODER = "resnet18"
    LAST_STAGE = "linear"
    POLY_POW = 3

print("Loading Data")
target_attr = "Attractive"
concept_attrs = CONCEPTS
pre_transform = transforms.Compose(
    [
        transforms.Resize((H, W)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ]
)
external_transform = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
if MODEL_TYPE == "CBM":
    curr_transform = TRANSFORM_DICT[ENCODER]
    external_transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            curr_transform,
        ]
    )
train_concept_data = CelebAJointConcept(
    data_dir=DATA_DIR,
    split="train",
    target=target_attr,
    concept=concept_attrs,
    mode="concept",
    transform=pre_transform,
    concept_num=8000,
)
train_concept_loader = DataLoader(train_concept_data, batch_size=B_SIZE)
val_concept_data = CelebAJointConcept(
    data_dir=DATA_DIR,
    split="valid",
    target=target_attr,
    concept=concept_attrs,
    mode="concept",
    transform=pre_transform,
    concept_num=4000,
)
val_concept_loader = DataLoader(val_concept_data, batch_size=B_SIZE)

print("Preparing for Training.")
model = AutoEncoder(latent_dim=LATENT_DIMS, H=64, W=64)
interpreter = ConstrastiveTCAV(model, DEVICE)
interpreter.train(TRAIN_PARAMS, train_concept_loader, val_concept_loader)
# interpreter.loadModel(save_dir, model)

print("Loading Main Model")
if MODEL_TYPE == "CNN":
    main_model = SimpleCNN().to(DEVICE)
elif MODEL_TYPE == "CBM":
    main_model = ConceptBottleneckModel(
        len(concept_attrs),
        1,
        encoder_name=ENCODER,
        pretrained=True,
        task_predictor_type=LAST_STAGE,
        poly_pow=POLY_POW,
    ).to(DEVICE)
checkpoint = torch.load(MODEL_PATH)
main_model.load_state_dict(checkpoint["model_state_dict"])


print("Getting Attribution")
print(f"{MODEL_PATH=}")
attributions = {}
for c_num, c_name in enumerate(concept_attrs):
    grads = torch.zeros(len(val_concept_data))
    start = 0
    with torch.no_grad():
        main_model.eval()
        for i, (imgs, concepts) in enumerate(val_concept_loader, 1):
            l = len(imgs)
            imgs = imgs.to(DEVICE)
            curr_grads = interpreter.getAttribution(
                main_model,
                imgs,
                c_num,
                0,
                eps=0.1,
                transform_func=external_transform,
                c_delta=True,
            )
            grads[start : start + l] = curr_grads
            start += l
    print(
        f"{c_name=},{len(grads)=},{grads.mean()=},{2.00 * grads.std()/(len(grads)**0.5)=}"
    )

recon, z = interpreter.model(imgs)
visualize(imgs, recon)

num = len(train_concept_loader.dataset)
z_collected = torch.zeros((num, LATENT_DIMS))
l_collected = torch.zeros((num, len(concept_attrs)))
start = 0
for i, (imgs, concepts) in enumerate(train_concept_loader, 1):
    imgs = imgs.to(DEVICE)
    l = len(imgs)
    _, z = interpreter.model(imgs)
    z_collected[start : start + l] = z
    l_collected[start : start + l] = concepts

PCA_vis(z_collected.detach(), l_collected.detach(), num_components=5)
# interpreter.saveModel(
#     f"./models/ConTCAV_models/epochs_{epochs}_alpha_{alpha}_lr_{lr:.4f}/"
# )
# dists = interpreter.getClusterDistances()

if MODEL_TYPE == "CBM":
    print(f"CBM Weights = {main_model.task_predictor[0].weight}")
    print(f"Concepts: {concept_attrs}")
