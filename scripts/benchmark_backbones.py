"""
Benchmark script comparing backbone configurations:
  1. DINOv2  – fully frozen
  2. ResNet50 – fully frozen
  3. DINOv2  – last 2 blocks unfrozen
  4. ResNet50 – last 2 layers unfrozen

Run as:
  python -m scripts.benchmark_backbones [--epochs 10] [--batch-size 32]
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from src.data.labelstudio_parser import load_labelstudio_json
from src.data.dataset import ShipFoulingDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.dinov2_classifier import LinearClassifier
from src.training.train_one_epoch import train_one_epoch
from src.training.eval import evaluate


# ---------------------------------------------------------------------------
# Backbone factories
# ---------------------------------------------------------------------------

def build_dinov2(unfreeze_last_n: int = 0) -> tuple:
    backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    backbone.eval()
    _freeze_all(backbone)
    if unfreeze_last_n > 0:
        params = list(backbone.parameters())
        for p in params[-unfreeze_last_n:]:
            p.requires_grad = True
    return backbone, "dinov2_vits14"


def build_resnet50(unfreeze_last_n: int = 0) -> tuple:
    from torchvision.models import resnet50
    backbone = resnet50(pretrained=True)
    backbone.fc = torch.nn.Identity()
    backbone.eval()
    _freeze_all(backbone)
    if unfreeze_last_n > 0:
        params = list(backbone.parameters())
        for p in params[-unfreeze_last_n:]:
            p.requires_grad = True
    return backbone, "resnet50"


def _freeze_all(module: torch.nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def get_embed_dim(backbone: torch.nn.Module, img_size: int) -> int:
    dummy = torch.randn(1, 3, img_size, img_size)
    with torch.no_grad():
        out = backbone(dummy)
    return out.shape[-1]


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_experiment(name, backbone, backbone_name, train_loader, val_loader,
                   device, epochs, patience, lr, weight_decay, img_size):
    print(f"\n{'='*60}")
    print(f"  Experiment: {name}")
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in backbone.parameters())
    print(f"  Backbone trainable params: {trainable:,} / {total:,}")
    print(f"{'='*60}")

    embed_dim = get_embed_dim(backbone, img_size)
    model = LinearClassifier(backbone=backbone, embed_dim=embed_dim, num_classes=5).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)

    best_val_loss = float("inf")
    best_val_acc  = 0.0
    epochs_no_improve = 0
    history = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, val_loader, device)
        val_loss = float(metrics["loss"])
        val_acc  = float(metrics["acc"])
        history.append({"epoch": epoch, "train_loss": tr_loss,
                        "val_loss": val_loss, "val_acc": val_acc})
        print(f"  Epoch {epoch:02d}: train_loss={tr_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  Early stopping after epoch {epoch}.")
                break

    elapsed = time.time() - t0
    return {
        "name": name,
        "best_val_loss": best_val_loss,
        "best_val_acc":  best_val_acc,
        "epochs_run":    len(history),
        "time_s":        elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--patience",     type=int,   default=5)
    parser.add_argument("--batch-size",   type=int,   default=32)
    parser.add_argument("--num-workers",  type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--img-size",     type=int,   default=224)
    parser.add_argument("--unfreeze-last-n", type=int, default=20,
                        help="Number of backbone params to unfreeze for partial experiments")
    parser.add_argument("--label-file",   type=str,   default=None,
                        help="Override label JSON path (default: data/label_final.json)")
    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parent.parent
    data_dir     = project_root / "data"
    images_dir   = data_dir / "images"
    label_path   = Path(args.label_file) if args.label_file else data_dir / "label_final.json"

    print(f"Loading labels from {label_path} ...")
    raw_entries = load_labelstudio_json(label_path)

    train_tf = get_train_transforms(args.img_size)
    val_tf   = get_val_transforms(args.img_size)

    full_train = ShipFoulingDataset(images_dir, raw_entries, transform=train_tf)
    full_val   = ShipFoulingDataset(images_dir, raw_entries, transform=val_tf)

    n       = len(full_train)
    n_train = int(0.8 * n)
    n_val   = n - n_train
    gen     = torch.Generator().manual_seed(42)

    train_ds, _ = random_split(full_train, [n_train, n_val], generator=gen)
    _,   val_ds = random_split(full_val,   [n_train, n_val], generator=torch.Generator().manual_seed(42))

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers)
    train_loader = DataLoader(train_ds, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **loader_kwargs)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  Train: {n_train}  Val: {n_val}")

    experiments = [
        ("DINOv2  – frozen",          lambda: build_dinov2(unfreeze_last_n=0)),
        ("ResNet50 – frozen",         lambda: build_resnet50(unfreeze_last_n=0)),
        ("DINOv2  – partial unfreeze", lambda: build_dinov2(unfreeze_last_n=args.unfreeze_last_n)),
        ("ResNet50 – partial unfreeze",lambda: build_resnet50(unfreeze_last_n=args.unfreeze_last_n)),
    ]

    results = []
    for name, builder in experiments:
        backbone, backbone_name = builder()
        result = run_experiment(
            name=name,
            backbone=backbone,
            backbone_name=backbone_name,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            patience=args.patience,
            lr=args.lr,
            weight_decay=args.weight_decay,
            img_size=args.img_size,
        )
        results.append(result)

    # --- Summary table ---
    print(f"\n{'='*70}")
    print(f"  {'Experiment':<35} {'Val Loss':>9} {'Val Acc':>9} {'Epochs':>7} {'Time(s)':>9}")
    print(f"  {'-'*35} {'-'*9} {'-'*9} {'-'*7} {'-'*9}")
    for r in results:
        print(f"  {r['name']:<35} {r['best_val_loss']:>9.4f} {r['best_val_acc']:>9.4f} "
              f"{r['epochs_run']:>7} {r['time_s']:>9.1f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
