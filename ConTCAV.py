# Importing Libraries
import os
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from utils.losses import SupConLossMultiLabel


# Helper Class
class ConstrastiveTCAV:

    def __init__(self, model, device):
        self.device = device
        self.initModel(model)

    def train(self, train_params: dict, trainloader, testloader=None):
        self.train_params = train_params
        print("Training Auxiliary Model")
        self.model.train()
        recon_criterion = self.train_params["recon_loss_function"]()
        cls_criterion = SupConLossMultiLabel()
        alpha = self.train_params["alpha"]
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.train_params["learning_rate"]
        )
        latent_dim = None
        for epoch in range(1, self.train_params["epochs"] + 1):
            print(f"Working on epoch: {epoch}/{self.train_params['epochs']}")
            total_recon, total_cls = 0.0, 0.0
            num = 0
            for imgs, labels in trainloader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                if len(labels.shape) == 1:
                    labels = labels.reshape(-1, 1)
                recon, z = self.model(imgs)
                latent_dim = z.shape[1]
                loss_recon = recon_criterion(recon, imgs)
                labels = labels.float()
                loss_cls = cls_criterion(z, labels)
                loss = (1 - alpha) * loss_recon + alpha * loss_cls

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_recon += loss_recon.item()
                total_cls += loss_cls.item()
                num += 1

            total_recon = total_recon / num
            total_cls = total_cls / num

            if epoch % 1 == 0:
                print(
                    f"Train Recon Loss: {total_recon:.4f} | Train Class loss: {total_cls:.4f}"
                )
                if testloader != None:
                    self.test(testloader)
                    self.model.train()
        print("\nModel training completed")
        self.locateClusters(trainloader, latent_dim, train_params["Num_Concepts"])

    def test(self, testloader):
        self.model.eval()
        recon_criterion = self.train_params["recon_loss_function"]()
        cls_criterion = SupConLossMultiLabel()
        total_recon, total_cls = 0.0, 0.0
        num = 0
        with torch.no_grad():
            for imgs, labels in testloader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                if len(labels.shape) == 1:
                    labels = labels.reshape(-1, 1)

                recon, z = self.model(imgs)
                loss_recon = recon_criterion(recon, imgs)
                labels = labels.float()
                loss_cls = cls_criterion(z, labels)

                total_recon += loss_recon.item()
                total_cls += loss_cls.item()
                num += 1
        total_recon = total_recon / num
        total_cls = total_cls / num

        print(
            f"Validation Recon Loss: {total_recon:.4f} | Validation Class Loss: {total_cls:.4f}\n"
        )

    def initModel(self, model: torch.nn.Module):
        self.model = model
        self.model.to(self.device)

    def locateClusters(self, trainloader, latent_dim, num_classes) -> None:
        print("Locating Clusters")
        mu = torch.zeros((num_classes, latent_dim)).to(self.device)
        sigma = torch.zeros((num_classes, latent_dim, latent_dim)).to(self.device)
        num = len(trainloader.dataset)
        z_all = torch.zeros((num, latent_dim))
        concepts = torch.zeros((num, num_classes))
        with torch.no_grad():
            start = 0
            for imgs, labels in trainloader:
                imgs = imgs.to(self.device)
                _, z = self.model(imgs)
                k = len(labels)
                z_all[start : start + k] = z
                concepts[start : start + k] = labels
        for i in range(num_classes):
            pos = concepts[:, i] == 1
            mu[i] = z_all[pos].mean(axis=0)
            sigma[i] = z_all[pos].T.cov()
        self.mu = mu
        self.sigma = sigma

    def getAttribution(
        self,
        pred_model,
        imgs,
        concept_num,
        class_num,
        eps=0.01,
        transform_func=lambda x: x,
    ):
        recon, z = self.model(imgs)
        c_vector = (
            self.mu[concept_num] - z
        )  # TODO: Check if we can integrate sigma values to improve the vector?
        c_vector = c_vector / torch.norm(c_vector)  # converting to unit vector
        z_new = z + (eps * c_vector)
        imgs_new = self.model.decoder(z_new)
        preds = self.getPreds(pred_model, recon, class_num, transform_func)
        new_preds = self.getPreds(pred_model, imgs_new, class_num, transform_func)
        grads = (new_preds - preds) / eps
        return grads

    def saveModel(self, save_dir: Path = "ConTCAV_vals/"):
        save_dir = Path(save_dir)
        if not (os.path.isdir(save_dir)):
            os.makedirs(save_dir)
        torch.save(self.model.state_dict(), save_dir / "model.pt")
        torch.save(self.mu, save_dir / "mu.pt")
        torch.save(self.sigma, save_dir / "sigma.pt")

    def loadModel(self, save_dir: Path, model: nn.Module):
        save_dir = Path(save_dir)
        state_dict = torch.load(save_dir / "model.pt")
        model.load_state_dict(state_dict)
        self.model = model
        self.mu = torch.load(save_dir / "mu.pt")
        self.sigma = torch.load(save_dir / "sigma.pt")

    def getPreds(self, pred_model, imgs, class_num=None, transform_func=lambda x: x):
        preds = pred_model(transform_func(imgs))
        if len(preds.shape) == 1:
            preds = preds.reshape(-1, 1)
        if class_num == None:
            return preds
        return preds[:, class_num]


# Testing
if __name__ == "__main__":
    import random
    from torchvision import transforms
    from utils.data import CelebAJointConcept
    from torch.utils.data import DataLoader
    from utils.models import SimpleCNN, AutoEncoder
    from utils.debug import visualize, PCA_vis

    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    MODEL_PATH = "models/celebA_CNN.pth"
    DATA_DIR = "../Datasets/CelebA/"
    CONCEPTS = ["Age", "Gender", "Skin", "Bald"]
    TRAIN_PARAMS = {
        "epochs": 10,
        "recon_loss_function": torch.nn.MSELoss,
        "learning_rate": 5e-3,
        "alpha": 0.05,
        "Num_Concepts": len(CONCEPTS),
    }
    B_SIZE = 512
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    VALS_PATH = "./vals/"
    LATENT_DIMS = 2048
    H, W = (64, 64)

    print("Loading Data")
    target_attr = "Attractive"
    concept_attrs = CONCEPTS
    transform = transforms.Compose(
        [
            transforms.Resize((H, W)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
        ]
    )
    external_transform = transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
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
    model = AutoEncoder(latent_dim=LATENT_DIMS, H=64, W=64)
    checkpoint = torch.load(MODEL_PATH)
    main_model = SimpleCNN().to(DEVICE)
    main_model.load_state_dict(checkpoint["model_state_dict"])
    interpreter = ConstrastiveTCAV(model, DEVICE)
    interpreter.train(TRAIN_PARAMS, train_concept_loader, val_concept_loader)

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
                curr_grads = interpreter.getAttribution(
                    main_model,
                    imgs,
                    c_num,
                    0,
                    eps=0.1,
                    transform_func=external_transform,
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

    PCA_vis(z_collected.detach(), l_collected.detach())
