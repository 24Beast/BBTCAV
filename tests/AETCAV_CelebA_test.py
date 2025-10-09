import os
import yaml
import torch
import argparse
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader

# --- Local imports ---
from AE_TCAV import AETCAV
from utils.data import CelebAJointConcept
from utils.models import AutoEncoderWithClassifier, SimpleCNN
from utils.CBM import ConceptBottleneckModel
from utils.ImageNetModels import TRANSFORM_DICT
from utils.debug import visualize, PCA_vis, set_seed


# ----------------------------
#  Helper Functions
# ----------------------------
def get_dataloaders(
    data_dir,
    target_attr,
    concept_attrs,
    encoder,
    h,
    w,
    batch_size,
    concept_num_train,
    concept_num_val,
):
    print("Loading Data...")

    if encoder in TRANSFORM_DICT:
        transform = TRANSFORM_DICT[encoder]
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((h, w)),
                transforms.ToTensor(),
            ]
        )

    train_dataset = CelebAJointConcept(
        data_dir=data_dir,
        split="train",
        target=target_attr,
        concept=concept_attrs,
        mode="concept",
        transform=transform,
        concept_num=concept_num_train,
    )
    val_dataset = CelebAJointConcept(
        data_dir=data_dir,
        split="valid",
        target=target_attr,
        concept=concept_attrs,
        mode="concept",
        transform=transform,
        concept_num=concept_num_val,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=4, pin_memory=True
    )
    return train_loader, val_loader, val_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AE-TCAV Runner with YAML Config")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML configuration file"
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(0)
    os.makedirs(config["data"]["vals_path"], exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_attr = config["data"]["target_attr"]
    concept_attrs = config["data"]["concept_attrs"]

    train_loader, val_loader, val_dataset = get_dataloaders(
        config["data"]["data_dir"],
        target_attr,
        concept_attrs,
        config["model"]["encoder"],
        config["data"]["height"],
        config["data"]["width"],
        config["training"]["batch_size"],
        config["data"]["concept_num_train"],
        config["data"]["concept_num_val"],
    )

    print(f"Preparing {config['model']['type'].upper()} model for training...")

    ae_model = AutoEncoderWithClassifier(
        latent_dim=config["model"]["latent_dim"],
        num_classes=len(concept_attrs),
        H=config["data"]["height"],
        W=config["data"]["width"],
    ).to(device)

    if config["model"]["type"] == "resnet":
        checkpoint = torch.load(config["model"]["model_path"], map_location=device)
        main_model = ConceptBottleneckModel(
            len(concept_attrs),
            1,
            encoder_name=config["model"]["encoder"],
            pretrained=True,
            task_predictor_type=config["model"]["last_stage"],
            poly_pow=config["model"]["poly_pow"],
        ).to(device)
        main_model.load_state_dict(checkpoint["model_state_dict"])
        external_transform = None

    elif config["model"]["type"] == "cnn":
        checkpoint = torch.load(config["model"]["model_path"], map_location=device)
        main_model = SimpleCNN().to(device)
        main_model.load_state_dict(checkpoint["model_state_dict"])
        external_transform = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)

    else:
        raise ValueError("Invalid model type. Choose 'resnet' or 'cnn'.")

    train_params = {
        "epochs": config["training"]["epochs"],
        "recon_loss_function": torch.nn.MSELoss,
        "cls_loss_function": torch.nn.BCEWithLogitsLoss,
        "learning_rate": config["training"]["lr"],
        "alpha": config["training"]["alpha"],
    }

    interpreter = AETCAV(ae_model, train_params, device)
    interpreter.train(train_loader, val_loader)

    print("Computing Attributions...")
    main_model.eval()
    attributions = {}

    with torch.no_grad():
        for c_idx, c_name in enumerate(concept_attrs):
            grads_list = []
            for imgs, _ in tqdm(val_loader, desc=f"Attribution for {c_name}"):
                imgs = imgs.to(device)
                grads = interpreter.getAttribution(
                    main_model, imgs, c_idx, 0, eps=1, transform_func=external_transform
                )
                grads_list.append(grads.cpu())
            grads = torch.cat(grads_list)
            attributions[c_name] = grads
            print(
                f"{c_name}: mean={grads.mean():.4f}, stderr={2.00 * grads.std() / (len(grads)**0.5):.4f}"
            )

    imgs_cpu = imgs.cpu().detach()
    recon, _, _ = interpreter.model(imgs_cpu.to(device))
    visualize(imgs_cpu, recon.cpu().detach())

    print("Collecting Latent Representations...")
    z_list, l_list = [], []
    for imgs, concepts in tqdm(train_loader, desc="Encoding Latents"):
        imgs = imgs.to(device)
        _, _, z = interpreter.model(imgs)
        z_list.append(z.cpu())
        l_list.append(concepts)

    z_all = torch.cat(z_list)
    l_all = torch.cat(l_list)
    PCA_vis(z_all.detach(), l_all.detach(), num_components=5)
