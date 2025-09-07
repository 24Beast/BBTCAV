import torch
import torch.nn as nn
from typing import Literal
from torchvision import models

VALID_TASK_PRED_TYPES = ["linear", "poly"]


class ConceptBottleneckModel(nn.Module):

    def __init__(
        self,
        num_concepts: int,
        num_classes: int,
        encoder_name: str = "resnet18",
        pretrained: bool = False,
        bottleneck_activation: str = "sigmoid",
        encoder_output_dim: int = 512,
        task_predictor_type: Literal["linear", "poly"] = "linear",
        poly_pow: int = 2,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.bottleneck_activation = bottleneck_activation
        if not (task_predictor_type in VALID_TASK_PRED_TYPES):
            raise ValueError(
                f"Unexpected value: {task_predictor_type} given for task_predictor_type expected values in {VALID_TASK_PRED_TYPES}"
            )
        self.task_predictor_type = task_predictor_type
        self.poly_pow = poly_pow

        if encoder_name == "resnet18":
            enc = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif encoder_name == "vit_b_16":
            enc = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        else:
            raise ValueError("Unsupported encoder_name: %s" % encoder_name)
        self.encoder = nn.Sequential(*list(enc.children())[:-1])  # remove fc
        encoder_output_dim = enc.fc.in_features

        self.concept_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(encoder_output_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_concepts),
        )

        if task_predictor_type == "linear":
            self.task_predictor = nn.Sequential(
                nn.Linear(num_concepts, num_classes),
            )
        elif task_predictor_type == "poly":
            self.exponents = torch.arange(1, self.poly_pow + 1)
            self.task_predictor = nn.Sequential(
                nn.Linear(int(num_concepts * self.poly_pow), num_classes),
            )

    def apply_concept_activation(self, concept_logits: torch.Tensor):
        if self.bottleneck_activation == "sigmoid":
            concept_probs = torch.sigmoid(concept_logits)
        elif self.bottleneck_activation == "softmax":
            concept_probs = torch.softmax(concept_logits)
        else:
            raise ValueError(
                "Unknown bottleneck activation: %s" % self.bottleneck_activation
            )
        return concept_probs

    def forward(self, x: torch.Tensor, return_intermediate: bool = False):
        features = self.encoder(x)
        concept_logits = self.concept_predictor(features)

        concept_probs = self.apply_concept_activation(concept_logits)

        temp = concept_probs

        if self.task_predictor_type == "poly":
            self.exponents = self.exponents.to(temp.device)
            temp = temp.unsqueeze(-1) ** self.exponents.view(1, 1, -1)
            temp = temp.reshape(concept_probs.size(0), -1)

        task_logits = self.task_predictor(temp)

        if return_intermediate:
            return concept_logits, concept_probs, task_logits
        return task_logits

    def predict_concepts(self, x: torch.Tensor):
        """Return concept probabilities only."""
        features = self.encoder(x)
        concept_logits = self.concept_predictor(features)
        return torch.sigmoid(concept_logits)

    def predict_from_concepts(self, c: torch.Tensor):
        if c.max() > 1 or c.min() < 0:
            c = self.apply_concept_activation(c)
        task_logits = self.task_predictor(c)
        return task_logits
