# Importing Libraries
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image


def visualize(imgs, recons, num_imgs=10):
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
        ax[0].set_title("image")
        ax[1].imshow(recon)
        ax[1].set_title("reconstruct")
        plt.show()
