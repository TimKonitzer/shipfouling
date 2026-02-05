import torch
import torch.nn as nn


class DinoV2LinearClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, embed_dim: int, num_classes: int = 5):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # backbone outputs embeddings; we assume [B, D]
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits
