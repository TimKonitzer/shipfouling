import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk

import torch
import torch.nn.functional as F
from PIL import Image, ImageTk

from src.data.labelstudio_parser import parse_entry, soft_label_from_annotations
from src.inference.predict import build_model_from_checkpoint, predict_image


NUM_CLASSES = 5

CLASS_NAMES = {
    0: "0 - Clean",
    1: "1 - Microfouling",
    2: "2",
    3: "3",
    4: "4",
}


def resolve_default_checkpoint(project_root: Path) -> Path:
    ckpt_dir = project_root / "checkpoints"
    preferred = ckpt_dir / "dinov2_vitb14_linear_probe_best.pt"
    if preferred.exists():
        return preferred

    best_candidates = list(ckpt_dir.glob("*_best.pt"))
    if best_candidates:
        return max(best_candidates, key=lambda p: p.stat().st_mtime)

    fallback = ckpt_dir / "dinov2_vitb14_linear_probe.pt"
    if fallback.exists():
        return fallback

    other_candidates = list(ckpt_dir.glob("*.pt"))
    if other_candidates:
        return max(other_candidates, key=lambda p: p.stat().st_mtime)

    return preferred


class ModelErrorViewer(tk.Tk):
    def __init__(self, images_dir: Path, label_json: Path, ckpt_path: Path):
        super().__init__()
        self.title("Ship Fouling Model Error Review")
        self.geometry("1300x820")

        self.images_dir = images_dir
        self.label_json = label_json
        self.ckpt_path = ckpt_path

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.tf, self.model_meta = build_model_from_checkpoint(ckpt_path, self.device)

        self.entries = self._load_entries(label_json)
        self.items = self._build_items()

        if not self.items:
            raise RuntimeError("No images with valid labels found in label.json.")

        self.idx = 0
        self._photo = None

        self._build_ui()
        self._bind_keys()
        self._update_view()

    def _load_entries(self, path: Path) -> List[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise TypeError(f"Expected list in label.json, got {type(data)}")
        return data

    def _build_items(self) -> List[Dict[str, Any]]:
        items = []
        for entry in self.entries:
            pe = parse_entry(entry)
            fname = pe.get("image")
            if not fname:
                continue
            img_path = self.images_dir / fname
            if not img_path.exists():
                continue

            probs = soft_label_from_annotations(pe.get("annotations", []), num_classes=NUM_CLASSES)
            target = torch.tensor(probs, dtype=torch.float32, device=self.device).unsqueeze(0)

            img = Image.open(img_path).convert("RGB")
            x = self.tf(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(x)
                log_probs = F.log_softmax(logits, dim=1)
                loss = F.kl_div(log_probs, target, reduction="sum").item()

            items.append(
                {
                    "image": fname,
                    "loss": float(loss),
                    "target": probs,
                }
            )

        items.sort(key=lambda x: x["loss"], reverse=True)
        return items

    def _build_ui(self):
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)

        self.image_label = ttk.Label(self)
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        right.rowconfigure(4, weight=1)
        right.columnconfigure(0, weight=1)

        self.title_lbl = ttk.Label(right, text="", font=("Segoe UI", 12, "bold"))
        self.title_lbl.grid(row=0, column=0, sticky="w")

        self.meta_lbl = ttk.Label(right, text="", font=("Segoe UI", 10))
        self.meta_lbl.grid(row=1, column=0, sticky="w", pady=(6, 6))

        self.pred_lbl = ttk.Label(right, text="", font=("Segoe UI", 10))
        self.pred_lbl.grid(row=2, column=0, sticky="w")

        self.target_lbl = ttk.Label(right, text="", font=("Segoe UI", 10))
        self.target_lbl.grid(row=3, column=0, sticky="w", pady=(6, 0))

        footer = ttk.Label(
            right,
            text="Keys: Left/Right to navigate, Esc to quit",
            font=("Segoe UI", 9),
        )
        footer.grid(row=5, column=0, sticky="w", pady=(8, 0))

    def _bind_keys(self):
        self.bind("<Left>", lambda _e: self._prev())
        self.bind("<Right>", lambda _e: self._next())
        self.bind("<Escape>", lambda _e: self.destroy())

    def _prev(self):
        self.idx = (self.idx - 1) % len(self.items)
        self._update_view()

    def _next(self):
        self.idx = (self.idx + 1) % len(self.items)
        self._update_view()

    def _update_view(self):
        item = self.items[self.idx]
        img_path = self.images_dir / item["image"]

        pred, prob_dict = predict_image(self.model, self.tf, img_path, self.device)
        pred_name = CLASS_NAMES.get(pred, str(pred))
        probs_str = " | ".join([f"{k}:{prob_dict[k]*100:5.1f}%" for k in sorted(prob_dict.keys())])
        target_str = " | ".join([f"{i}:{item['target'][i]*100:5.1f}%" for i in range(len(item["target"]))])

        self.title_lbl.config(text=f"[{self.idx+1}/{len(self.items)}] {item['image']}")
        self.meta_lbl.config(text=f"KL divergence (soft labels): {item['loss']:.4f}")
        self.pred_lbl.config(text=f"Prediction: {pred_name}\nProbs: {probs_str}\nDevice: {self.device}")
        self.target_lbl.config(text=f"Target (soft): {target_str}")

        pil = Image.open(img_path).convert("RGB")
        max_w, max_h = 820, 780
        pil.thumbnail((max_w, max_h))
        self._photo = ImageTk.PhotoImage(pil)
        self.image_label.config(image=self._photo)


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    images_dir = project_root / "data" / "images"
    label_json = project_root / "data" / "label.json"
    ckpt_path = resolve_default_checkpoint(project_root)

    app = ModelErrorViewer(images_dir=images_dir, label_json=label_json, ckpt_path=ckpt_path)
    app.mainloop()


if __name__ == "__main__":
    main()
