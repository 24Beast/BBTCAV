# Importing Libraries
import os
import torch
import numpy as np
import torch.nn as nn
from pathlib import Path
from utils.losses import SupConLossMultiLabel
from scipy.stats import chi2
from utils.debug import visualize


# Helper Class
class PCA_CAV:

    def __init__(self, model, device):
        self.device = device
        self.initModel(model)

    def train(self, train_params: dict, trainloader, testloader=None):
        self.train_params = train_params
        print("Training Auxiliary Model")
        self.model.train()
        recon_criterion = self.train_params["recon_loss_function"]
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
        recon_criterion = self.train_params["recon_loss_function"]
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
        self.fit_pca(z_all)
        z_new = self.transform_pca(z_all)
        mu = torch.zeros((num_classes, self.pca_k)).to(self.device)
        sigma = torch.zeros((num_classes, self.pca_k, self.pca_k)).to(self.device)
        for i in range(num_classes):
            pos = concepts[:, i] == 1
            mu[i] = z_new[pos].mean(axis=0)
            sigma[i] = z_new[pos].T.cov()
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
        c_delta=False,
        debug=True,
    ):
        recon, z = self.model(imgs)
        if not (c_delta):
            c_vector = self.mu[concept_num] - z
            c_vector = c_vector / torch.norm(c_vector)  # converting to unit vector
            z_new = z + (eps * c_vector)
        else:
            c_vector = self.calc_c_vector(z, concept_num, eps)
            c_vector = c_vector.to(self.device)
            z_new = z + c_vector
        imgs_new = self.model.decoder(z_new)
        if debug:
            visualize(recon[:20].detach(), imgs_new[:20].detach())
        preds = self.getPreds(pred_model, recon, class_num, transform_func)
        new_preds = self.getPreds(pred_model, imgs_new, class_num, transform_func)
        grads = (new_preds - preds) / eps
        return grads

    def calc_c_vector(self, z, concept_num, delta=0.1):
        mu = self.mu[concept_num]
        sigma = self.sigma[concept_num]
        z = self.transform_pca(z)
        n = len(mu)
        c_vector = torch.zeros(z.shape)
        sigma_inv = sigma.inverse()
        z_c = mu - z
        for i in range(len(z)):
            k = -1 * (z_c[i].T @ sigma_inv @ z_c[i]) * np.log(2) / chi2.ppf(0.95, n)
            if k > 0:
                k = k * -1
            k = k / len(z)
            probs = torch.exp(k)
            probs_d = torch.clamp(probs + delta, 0.00, 1.00)
            a = 1 - (torch.log(probs_d) / k) ** 0.5
            c_vector[i] = z_c[i] * a
        c_vector = self.inverse_transform_pca(c_vector)
        return c_vector

    def saveModel(self, save_dir: Path = "PCA_CAV_vals/"):
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
        transform_imgs = [transform_func(imgs[i]) for i in range(len(imgs))]
        preds = pred_model(torch.stack(transform_imgs).to(self.device))
        if len(preds.shape) == 1:
            preds = preds.reshape(-1, 1)
        # preds = torch.sigmoid(preds)
        if class_num == None:
            return preds
        return preds[:, class_num]

    def fit_pca(
        self,
        z_all: torch.Tensor,
        n_components: int = None,
        energy: float = 0.95,
        min_k: int = 1,
        max_k: int = None,
        eps: float = 1e-8,
    ):
        """
        Fit PCA (SVD) on z_all (N x D) and choose k dynamically based on explained variance.

        Args:
            z_all: (N, D) tensor of latent vectors (device-aware).
            n_components: If provided, acts as an upper bound (still may be reduced by energy).
            energy: target cumulative explained variance to retain (0 < energy <= 1).
            min_k: minimum number of components to keep.
            max_k: maximum number of components to allow (if None, uses min(D, N-1)).
            eps: small numerical constant.

        Stores:
            self.pca_mean (D,)
            self.pca_components (k, D)   # rows are principal directions
            self.pca_k (int)
            self.explained_variance_ratio (k,)
            self.cumulative_explained_variance (k,)
        Returns:
            (pca_mean, pca_components)
        """
        device = z_all.device
        N, D = z_all.shape
        if N == 0:
            raise ValueError("z_all must contain at least one sample")
        # sensible default caps
        if max_k is None:
            max_k = min(
                D, max(1, N - 1)
            )  # can't have more meaningful components than N-1
        # center
        pca_mean = z_all.mean(dim=0, keepdim=True)  # (1,D)
        Xc = z_all - pca_mean  # (N,D)

        # Compute compact SVD. If GPU SVD fails for memory, fall back to CPU.
        try:
            U, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        except RuntimeError:
            U, S, Vh = torch.linalg.svd(Xc.cpu(), full_matrices=False)
            U, S, Vh = U.to(device), S.to(device), Vh.to(device)

        # singular values S shape = (r,) where r = min(N, D)
        # compute explained variance per component:
        # variance_i = S_i^2 / (N - 1)  (same formulation sklearn uses)
        denom = max(1, N - 1)
        variances = (S**2) / float(denom)  # (r,)
        total_var = variances.sum().item() + eps
        explained_variance_ratio = (
            (variances / total_var).cpu().numpy()
        )  # numpy for easy cum
        cumvar = np.cumsum(explained_variance_ratio)

        # determine k using energy threshold
        # find first index where cumulative >= energy
        idx = np.searchsorted(cumvar, float(energy), side="left")
        # idx is 0-based; we want k = idx+1
        k_by_energy = int(min(len(cumvar), idx + 1))
        # apply bounds
        if n_components is not None:
            k_upper = min(int(n_components), max_k)
        else:
            k_upper = max_k
        k = max(min_k, min(k_by_energy, k_upper))
        # clamp to available components (r)
        r = S.shape[0]
        k = min(k, r)

        # final PCA components: top-k rows of Vh (Vh shape r x D)
        components = Vh[:k, :].to(device)  # (k, D)

        # store explained variance arrays restricted to chosen k
        self.explained_variance_ratio = torch.tensor(
            explained_variance_ratio[:k], device=device
        )
        self.cumulative_explained_variance = torch.tensor(cumvar[:k], device=device)

        # persist
        self.pca_mean = pca_mean.squeeze(0).to(device)  # (D,)
        self.pca_components = components  # (k, D)
        self.pca_k = k

        # logging-ish return
        return self.pca_mean, self.pca_components

    def transform_pca(self, z: torch.Tensor) -> torch.Tensor:
        """
        Project z (B x D) into PCA space (B x k).
        Requires self.pca_components and self.pca_mean to be set.
        """
        if not hasattr(self, "pca_components") or not hasattr(self, "pca_mean"):
            raise RuntimeError("PCA not fitted. Call fit_pca first.")
        device = z.device
        comps = self.pca_components.to(device)  # (k, D)
        mean = self.pca_mean.to(device)  # (D,)
        z_centered = z - mean.unsqueeze(0)  # (B, D)
        z_pca = z_centered @ comps.T  # (B, k)
        return z_pca

    def inverse_transform_pca(self, z_pca: torch.Tensor) -> torch.Tensor:
        """
        Inverse transform from PCA-space (B x k) back to original latent space (B x D).
        """
        if not hasattr(self, "pca_components") or not hasattr(self, "pca_mean"):
            raise RuntimeError("PCA not fitted. Call fit_pca first.")
        device = z_pca.device
        comps = self.pca_components.to(device)  # (k, D)
        mean = self.pca_mean.to(device)  # (D,)
        z_orig = z_pca @ comps  # (B, D)
        z_orig = z_orig + mean.unsqueeze(0)
        return z_orig


# Testing
if __name__ == "__main__":
    from torchvision import transforms
    from utils.data import CelebAJointConcept
    from torch.utils.data import DataLoader
    from utils.models import SimpleCNN, AutoEncoder
    from utils.debug import visualize, PCA_vis, set_seed
    from utils.losses import MixedReconLoss
    from utils.ImageNetModels import TRANSFORM_DICT
    from utils.CBM import ConceptBottleneckModel

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
    H, W = (128, 128)

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
        concept_num=4000,
    )
    train_concept_loader = DataLoader(train_concept_data, batch_size=B_SIZE)
    val_concept_data = CelebAJointConcept(
        data_dir=DATA_DIR,
        split="valid",
        target=target_attr,
        concept=concept_attrs,
        mode="concept",
        transform=pre_transform,
        concept_num=2000,
    )
    val_concept_loader = DataLoader(val_concept_data, batch_size=B_SIZE)

    print("Preparing for Training.")
    model = AutoEncoder(latent_dim=LATENT_DIMS, H=H, W=W)
    interpreter = PCA_CAV(model, DEVICE)
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
                    eps=0.01,
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
