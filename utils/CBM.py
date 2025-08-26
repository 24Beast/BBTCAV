import torch
import torch.nn as nn
from torchvision import models


class ConceptBottleneckModel(nn.Module):

    def __init__(
        self,
        num_concepts: int,
        num_classes: int,
        encoder_name: str = "resnet18",
        pretrained: bool = False,
        bottleneck_activation: str = "sigmoid",
        encoder_output_dim: int = 512,
    ):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes
        self.bottleneck_activation = bottleneck_activation

        if encoder_name == "resnet18":
            enc = models.resnet18(pretrained=pretrained)
            self.encoder = nn.Sequential(*list(enc.children())[:-1])  # remove fc
            encoder_output_dim = enc.fc.in_features
        else:
            raise ValueError("Unsupported encoder_name: %s" % encoder_name)

        self.concept_predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(encoder_output_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_concepts),
        )

        self.task_predictor = nn.Sequential(
            nn.Linear(num_concepts, num_classes),
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

        task_logits = self.task_predictor(concept_probs)

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
