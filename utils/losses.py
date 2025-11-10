import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F


def ssim(img1, img2, window_size=11, size_average=True, val_range=None):
    # Compute SSIM between two images (supports batches)
    if val_range is None:
        max_val = 1 if torch.max(img1) <= 1 else 255
        min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = window_size // 2
    channel = img1.size(1)

    # Proper tensor-based Gaussian kernel
    def gaussian(window_size, sigma):
        coords = torch.arange(window_size).float() - window_size // 2
        gauss = torch.exp(-(coords**2) / (2 * sigma**2))
        return gauss / gauss.sum()

    def create_window(window_size, channel):
        _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window

    window = create_window(window_size, channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (Khosla et al., 2020)."""

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: [batch_size, latent_dim]
        labels: [batch_size]
        """
        device = features.device
        batch_size = features.shape[0]

        # Normalize features
        features = F.normalize(features, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        # Mask self-contrast
        mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(mask, 0)

        # Build label mask
        labels = labels.contiguous().view(-1, 1)
        match_mask = torch.eq(labels, labels.T).float().to(device)

        # Compute log prob
        exp_sim = torch.exp(sim_matrix)
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # Only keep positives
        mean_log_prob_pos = (match_mask * log_prob).sum(1) / match_mask.sum(1)

        # Loss
        loss = -mean_log_prob_pos.mean()
        return loss


class SupConLossMultiLabel(nn.Module):
    """
    Supervised Contrastive Loss for multi-label classification.
    Each sample can have multiple labels.
    Positives = samples that share at least one label.
    """

    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        Args:
            features: [batch_size, latent_dim]
            labels:   [batch_size, num_classes]  multi-hot (0/1) vectors
        """
        device = features.device
        batch_size = features.size(0)

        # Normalize feature embeddings
        features = F.normalize(features, dim=1)
        # print(f"{features.mean()=}, {features.std()=}")

        # Cosine similarity matrix [B, B]
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        # print(f"{sim_matrix.mean()=}, {sim_matrix.std()=}")

        # Mask out self-comparisons
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(self_mask, 0)  # - inf causes nan values!
        # print(f"{sim_matrix.mean()=}, {sim_matrix.std()=}")

        # Positive mask: [B, B] where samples share ≥1 label
        pos_mask = (labels @ labels.T) > 0  # bool

        # Compute log-probabilities
        logsumexp = torch.logsumexp(sim_matrix, dim=1, keepdim=True)
        log_prob = sim_matrix - logsumexp
        # print(f"{log_prob.mean()=}, {log_prob.std()=}")

        # Only keep positives
        pos_mask = pos_mask.float()
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / pos_mask.sum(1).clamp(min=1)

        # Final loss
        loss = -mean_log_prob_pos.mean()
        return loss


def MixedReconLoss(alpha=0.33, beta=0.33, gamma=0.33):
    mse_loss = torch.nn.MSELoss()
    lpips_loss = lpips.LPIPS(net="alex")
    lpips_loss = lpips_loss.to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    fn = (
        lambda y, y_pred: (alpha * mse_loss(y, y_pred))
        + (beta * lpips_loss(y, y_pred)).mean()
        + (gamma * (1 - ssim(y, y_pred)))
    )
    return fn


if __name__ == "__main__":
    from utils.models import AutoEncoder

    x = torch.rand(50, 3, 32, 32)  # e.g. MNIST flattened
    y = torch.randint(0, 10, (50,))  # class labels

    model = AutoEncoder(1024)
    recon, z = model(x)

    # Losses
    recon_loss = F.mse_loss(recon, x)
    contrastive_loss = SupConLoss()(z, y)

    # Combine losses (tune lambda)
    lambda_contrast = 0.1
    loss = recon_loss + lambda_contrast * contrastive_loss

    loss.backward()
