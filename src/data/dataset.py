from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .labelstudio_parser import parse_entry, soft_label_from_annotations


class ShipFoulingDataset(Dataset):
    """
    Returns:
      image_tensor: torch.FloatTensor [3,H,W]
      target_probs: torch.FloatTensor [6]   (soft labels)
      meta: dict with filename + etc.
    """

    def __init__(
        self,
        images_dir: Path,
        raw_entries: List[Dict[str, Any]],
        transform=None,
        num_classes: int = 5,
    ):
        self.images_dir = images_dir
        self.transform = transform
        self.num_classes = num_classes

        self.samples: List[Tuple[str, np.ndarray]] = []
        for e in raw_entries:
            pe = parse_entry(e)
            fname = pe["image"]
            if not fname:
                continue

            anns = pe["annotations"]
            probs = soft_label_from_annotations(anns, num_classes=num_classes)

            img_path = images_dir / fname
            if img_path.exists():
                self.samples.append((fname, probs))
            # sonst skip (oder warnen)

        if len(self.samples) == 0:
            raise RuntimeError("No samples found. Check images_dir and label.json filenames.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        fname, probs = self.samples[idx]
        img_path = self.images_dir / fname

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)
        else:
            # fallback: minimal
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        target = torch.tensor(probs, dtype=torch.float32)
        meta = {"filename": fname}
        return img, target, meta
