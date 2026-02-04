from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from src.data.labelstudio_parser import load_labelstudio_json
from src.data.dataset import ShipFoulingDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.dinov2_classifier import DinoV2LinearClassifier
from src.training.train_one_epoch import train_one_epoch
from src.training.eval import evaluate


def main():
    # --- paths ---
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    images_dir = data_dir / "images"
    label_path = data_dir / "label.json"

    # --- load json ---
    raw_entries = load_labelstudio_json(label_path)

    # --- transforms ---
    train_tf = get_train_transforms(224)
    val_tf = get_val_transforms(224)

    # --- dataset ---
    full_ds = ShipFoulingDataset(images_dir=images_dir, raw_entries=raw_entries, transform=None)

    # We'll apply transforms via wrapper to keep it simple here:
    # easiest: create two datasets with different transforms using the same raw_entries
    # (for now, just re-instantiate)
    full_ds_train = ShipFoulingDataset(images_dir=images_dir, raw_entries=raw_entries, transform=train_tf)
    full_ds_val = ShipFoulingDataset(images_dir=images_dir, raw_entries=raw_entries, transform=val_tf)

    n = len(full_ds_train)
    n_train = int(0.8 * n)
    n_val = n - n_train

    train_ds, _ = random_split(full_ds_train, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    _, val_ds = random_split(full_ds_val, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # --- load DINOv2 backbone from torch.hub ---
    # NOTE: requires internet once for download.
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval()

    # Freeze backbone
    for p in backbone.parameters():
        p.requires_grad = False

    # Determine embedding dim (ViT-S/14 usually 384)
    # We'll infer it by a dummy forward.
    dummy = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        out = backbone(dummy)
    embed_dim = out.shape[-1]
    print("Embed dim:", embed_dim)

    model = DinoV2LinearClassifier(backbone=backbone, embed_dim=embed_dim, num_classes=5).to(device)

    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    # --- train ---
    for epoch in range(1, 41):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        print(f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={metrics['loss']:.4f} val_acc={metrics['acc']:.4f}")

    ckpt_dir = project_root / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    ckpt_path = ckpt_dir / "dinov2_linear_probe.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "backbone": "dinov2_vits14",
            "num_classes": 5,
            "img_size": 224,
            "epoch": epoch,
        },
        ckpt_path,
    )
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
