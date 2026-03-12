import torch
import torch.nn as nn


class LinearClassifier(nn.Module):
    """Generic linear probe on top of any backbone that returns a feature vector.

    Previously named DinoV2LinearClassifier, the class now supports ResNet50 or
    other feature extractors as long as ``backbone(x)`` yields a tensor of shape
    [B, embed_dim].
    """

    def __init__(self, backbone: nn.Module, embed_dim: int, num_classes: int = 5):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # backbone outputs embeddings; we assume [B, D]
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits
