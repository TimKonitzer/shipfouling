import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from tqdm import tqdm

import sys
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.data.underwater_dataset import UnderwaterImageDataset, UnderwaterPatchDataset
from src.models.dinov2_classifier import LinearClassifier

def parse_args():
    parser = argparse.ArgumentParser(description="Pretrain Encoder on Underwater Data.")
    parser.add_argument("--csv-path", required=True, type=str, help="Path to the underwater dataset CSV.")
    parser.add_argument("--images-dir", required=True, type=str, help="Path to the underwater dataset images.")
    parser.add_argument("--dataset-mode", choices=["patch", "image"], default="patch", 
                        help="Mode for dataloading. Point annotations suggest 'patch', full image context suggests 'image'.")
    parser.add_argument("--backbone", default="resnet50", help="e.g. resnet50, dinov2_vits14")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--save-path", default="checkpoints/underwater_encoder.pt", type=str)
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    if args.dataset_mode == "patch":
        dataset = UnderwaterPatchDataset(args.csv_path, args.images_dir, transform=tf, patch_size=args.img_size)
    else:
        dataset = UnderwaterImageDataset(args.csv_path, args.images_dir, transform=tf)
        
    num_classes = len(dataset.classes)
    print(f"Loaded dataset ({args.dataset_mode} mode) with {len(dataset)} items, {num_classes} classes.")
    
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    if args.dataset_mode == "patch":
        targets = [dataset.class_to_idx[dataset.df.iloc[i]['Label']] for i in train_ds.indices]
        class_sample_count = torch.tensor([(torch.tensor(targets) == t).sum() for t in range(num_classes)])
        weight = 1. / class_sample_count.float()
        samples_weight = torch.tensor([weight[t] for t in targets])
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
        
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, num_workers=4)
    else:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
        
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    if args.backbone.startswith("dinov2_"):
        backbone = torch.hub.load("facebookresearch/dinov2", args.backbone)
    elif args.backbone == "resnet50":
        from torchvision.models import resnet50
        backbone = resnet50(pretrained=True)
        backbone.fc = nn.Identity()
    else:
        raise ValueError("Unsupported backbone")
        
    for p in backbone.parameters():
        p.requires_grad = True

    dummy = torch.randn(1, 3, args.img_size, args.img_size)
    embed_dim = backbone(dummy).shape[-1]
    
    model = LinearClassifier(backbone, embed_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for imgs, targets in pbar:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({"loss": loss.item()})
        train_loss /= len(train_ds)
        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]"):
                imgs, targets = imgs.to(device), targets.to(device)
                logits = model(imgs)
                loss = criterion(logits, targets)
                val_loss += loss.item() * imgs.size(0)
                preds = logits.argmax(dim=-1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)
                    
        val_loss /= len(val_ds)
        acc = correct / max(total, 1)
        
        print(f"Epoch {epoch}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val Acc/Metric: {acc:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.backbone.state_dict(), args.save_path)
            print(f" -> Saved new best encoder to {args.save_path}")

if __name__ == "__main__":
    main()
