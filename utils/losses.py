import torch
import torch.nn as nn
import torch.nn.functional as F


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
