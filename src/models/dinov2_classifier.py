import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, embed_dim: int, num_classes: int = 5):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits
