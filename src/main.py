import argparse 
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from tqdm import tqdm
import numpy as np

from src.data.labelstudio_parser import load_labelstudio_json
from src.data.dataset import ShipFoulingDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.dinov2_classifier import LinearClassifier
from src.training.train_one_epoch import train_one_epoch
from src.training.eval import evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train linear probe on top of a backbone (DINOv2 or ResNet50).")
    parser.add_argument("--backbone", default="dinov2_vits14", help="e.g. dinov2_vits14, resnet50, etc.")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--disable-early-stopping", action="store_true")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--unfreeze-last-n", type=int, default=0)
    parser.add_argument("--pretrained-encoder", type=str, default=None, help="Path to custom pretrained encoder .pt file")
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    images_dir = data_dir / "images"
    label_path = data_dir / "label.json"
    num_classes = 5

    raw_entries = load_labelstudio_json(label_path)

    train_tf = get_train_transforms(args.img_size)
    val_tf = get_val_transforms(args.img_size)

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

    train_indices = train_ds.indices
    train_labels = []
    for idx in train_indices:
        _fname, probs = full_ds_train.samples[idx]
        train_labels.append(np.argmax(probs))
    
    class_counts = np.bincount(train_labels, minlength=num_classes)
    class_weights = 1.0 / (np.sqrt(class_counts) + 1e-6)
    sample_weights = [class_weights[label] for label in train_labels]
    
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size, 
        sampler=sampler, 
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers,
        pin_memory=True
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    backbone_name = args.backbone
    if backbone_name.startswith("dinov2_"):
        backbone = torch.hub.load("facebookresearch/dinov2", backbone_name)
    elif backbone_name == "resnet50":
        from torchvision.models import resnet50

        backbone = resnet50(pretrained=True)
        backbone.fc = torch.nn.Identity()
    else:
        raise ValueError(f"unsupported backbone '{backbone_name}'")

    if getattr(args, "pretrained_encoder", None):
        print(f"Loading custom pretrained encoder from {args.pretrained_encoder}")
        state_dict = torch.load(args.pretrained_encoder, map_location="cpu")
        backbone.load_state_dict(state_dict)
    backbone.eval()

    backbone_params = list(backbone.parameters())
    for p in backbone_params:
        p.requires_grad = False
    if args.unfreeze_last_n > 0:
        for p in backbone_params[-args.unfreeze_last_n :]:
            p.requires_grad = True

    dummy = torch.randn(1, 3, args.img_size, args.img_size)
    with torch.no_grad():
        out = backbone(dummy)
    embed_dim = out.shape[-1]
    print("Embed dim:", embed_dim)

    model = LinearClassifier(backbone=backbone, embed_dim=embed_dim, num_classes=num_classes).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

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
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        model.train()
        epoch_tr_loss = 0.0
        n_samples = 0
        
        for images, targets, _meta in pbar:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            from src.training.train_one_epoch import soft_target_loss
            loss = soft_target_loss(logits, targets)
            loss.backward()
            optimizer.step()
            
            bs = images.size(0)
            epoch_tr_loss += float(loss.item()) * bs
            n_samples += bs
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        tr_loss = epoch_tr_loss / max(n_samples, 1)
        
        metrics = evaluate(model, val_loader, device)
        val_loss = float(metrics["loss"])
        last_epoch = epoch
        last_val_loss = val_loss
        
        scheduler.step()
        curr_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch}: train_loss={tr_loss:.4f} val_loss={val_loss:.4f} val_acc={metrics['acc']:.4f} lr={curr_lr:.6e}")

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
