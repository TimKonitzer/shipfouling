import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from src.data.labelstudio_parser import load_labelstudio_json
from src.data.dataset import ShipFoulingDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.dinov2_classifier import DinoV2LinearClassifier
from src.training.train_one_epoch import train_one_epoch
from src.training.eval import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train linear probe on DINOv2 embeddings.")
    parser.add_argument("--backbone", default="dinov2_vitb14", help="e.g. dinov2_vits14, dinov2_vitb14")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--disable-early-stopping", action="store_true")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    return parser.parse_args()


def main():
    args = parse_args()
    # --- paths ---
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    images_dir = data_dir / "images"
    label_path = data_dir / "label.json"
    num_classes = 5

    # --- load json ---
    raw_entries = load_labelstudio_json(label_path)

    # --- transforms ---
    train_tf = get_train_transforms(args.img_size)
    val_tf = get_val_transforms(args.img_size)

    # --- dataset ---
    full_ds_train = ShipFoulingDataset(
        images_dir=images_dir, raw_entries=raw_entries, transform=train_tf, num_classes=num_classes
    )
    full_ds_val = ShipFoulingDataset(
        images_dir=images_dir, raw_entries=raw_entries, transform=val_tf, num_classes=num_classes
    )

    n = len(full_ds_train)
    n_train = int(0.8 * n)
    n_val = n - n_train

    train_ds, _ = random_split(full_ds_train, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    _, val_ds = random_split(full_ds_val, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # --- load DINOv2 backbone from torch.hub ---
    # NOTE: requires internet once for download.
    backbone_name = args.backbone
    backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
    backbone.eval()

    # Freeze backbone
    for p in backbone.parameters():
        p.requires_grad = False

    # Determine embedding dim (inferred by a dummy forward)
    # We'll infer it by a dummy forward.
    dummy = torch.randn(1, 3, args.img_size, args.img_size)
    with torch.no_grad():
        out = backbone(dummy)
    embed_dim = out.shape[-1]
    print("Embed dim:", embed_dim)

    model = DinoV2LinearClassifier(backbone=backbone, embed_dim=embed_dim, num_classes=num_classes).to(device)

    optimizer = torch.optim.AdamW(
        model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # --- train ---
    ckpt_dir = project_root / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    base_name = f"{backbone_name}_linear_probe"
    best_ckpt_path = ckpt_dir / f"{base_name}_best.pt"
    last_ckpt_path = ckpt_dir / f"{base_name}.pt"

    best_val_loss = float("inf")
    epochs_no_improve = 0
    last_epoch = 0
    last_val_loss = None

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        val_loss = float(metrics["loss"])
        last_epoch = epoch
        last_val_loss = val_loss
        print(f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_acc={metrics['acc']:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "backbone": backbone_name,
                    "num_classes": num_classes,
                    "img_size": args.img_size,
                    "epoch": epoch,
                    "val_loss": val_loss,
                },
                best_ckpt_path,
            )
            print(f"Saved best checkpoint to {best_ckpt_path}")
        else:
            epochs_no_improve += 1

        if not args.disable_early_stopping and args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"Early stopping: no val_loss improvement for {args.patience} epochs.")
            break

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "backbone": backbone_name,
            "num_classes": num_classes,
            "img_size": args.img_size,
            "epoch": last_epoch,
            "val_loss": last_val_loss,
        },
        last_ckpt_path,
    )
    print(f"Saved last checkpoint to {last_ckpt_path}")


if __name__ == "__main__":
    main()
