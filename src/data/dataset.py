from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True

from .labelstudio_parser import parse_entry, soft_label_from_annotations


class ShipFoulingDataset(Dataset):


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
                try:
                    with Image.open(img_path) as img:
                        img.verify()
                    self.samples.append((fname, probs))
                except Exception as e:
                    print(f"Skipping corrupt image {img_path}: {e}")

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
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0

        target = torch.tensor(probs, dtype=torch.float32)
        meta = {"filename": fname}
        return img, target, meta
