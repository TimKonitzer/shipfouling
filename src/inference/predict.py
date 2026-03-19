from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.models.dinov2_classifier import LinearClassifier
from src.data.transforms import IMAGENET_MEAN, IMAGENET_STD


def build_model_from_checkpoint(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location=device)

    backbone_name = ckpt.get("backbone", "dinov2_vits14")
    num_classes = int(ckpt.get("num_classes", 5))
    img_size = int(ckpt.get("img_size", 224))
    if backbone_name.startswith("dinov2_"):
        backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
    elif backbone_name == "resnet50":
        from torchvision.models import resnet50

        backbone = resnet50(pretrained=True)
        backbone.fc = torch.nn.Identity()
    else:
        raise ValueError(f"unsupported backbone '{backbone_name}' in checkpoint")

    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    dummy = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        out = backbone(dummy)
    embed_dim = out.shape[-1]

    model = LinearClassifier(backbone=backbone, embed_dim=embed_dim, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    meta = {"backbone": backbone_name, "num_classes": num_classes, "img_size": img_size}
    return model, tf, meta


@torch.no_grad()
def predict_image(model, tf, image_path: Path, device: str) -> Tuple[int, Dict[int, float]]:
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=1).squeeze(0).cpu()
    pred = int(torch.argmax(probs).item())
    prob_dict = {i: float(probs[i].item()) for i in range(probs.shape[0])}
    return pred, prob_dict
