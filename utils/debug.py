# Importing Libraries
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def visualize(imgs, recons, num_imgs=10, img_label="image", recon_label="reconstruct"):
    if len(imgs) < num_imgs:
        num_imgs = len(imgs)
    idx = torch.randperm(len(imgs))[:num_imgs]
    imgs = imgs[idx]
    recons = recons[idx]
    for i in range(num_imgs):
        img = np.array(to_pil_image(imgs[i]))
        recon = np.array(to_pil_image(recons[i]))
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(img)
        ax[0].set_title(img_label)
        ax[1].imshow(recon)
        ax[1].set_title(recon_label)
        plt.show()


# PCA Visualization
def PCA_vis(z, labels, num_components=2):
    # TODO: Add functionality to view plots across multiple axes.
    if len(labels.shape) > 1:
        if labels.shape[1] == 1:
            labels = labels[:, 0]
        else:
            exp = torch.pow(1 + torch.arange(labels.shape[1]), 2)
            labels = (labels * exp).sum(axis=1)
    z_mean = z.mean(dim=0)
    z_centered = z - z_mean

    # Compute covariance matrix
    cov_matrix = torch.mm(z_centered.T, z_centered) / (z_centered.shape[0] - 1)

    # SVD on covariance matrix
    U, S, V = torch.svd(cov_matrix)

    # Select the top 'num_components'
    components = U[:, :num_components]

    # Project data onto principal components
    z_pca = torch.mm(z_centered, components)

    unique_labels = labels.unique()
    for i in range(num_components):
        for j in range(i + 1, num_components):
            for label in unique_labels:
                idx = labels == label
                z_curr = z_pca[idx]
                plt.scatter(z_curr[:, i], z_curr[:, j], label=label)
            plt.title(f"PCA Visualizations for axis {i} and {j}.")
            plt.legend()
            plt.show()
