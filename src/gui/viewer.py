import glob
from pathlib import Path
from typing import Dict, List, Any

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from src.inference.predict import build_model_from_checkpoint, predict_image
from src.data.labelstudio_index import build_image_to_annotations


CLASS_NAMES = {
    0: "0 - Clean",
    1: "1 - Microfouling",
    2: "2",
    3: "3",
    4: "4",
}


class ImageModelViewer(tk.Tk):
    def __init__(self, images_dir: Path, label_json: Path, ckpt_path: Path):
        super().__init__()
        self.title("Ship Fouling Viewer")
        self.geometry("1200x800")

        self.images_dir = images_dir
        self.files = sorted([Path(p) for p in glob.glob(str(images_dir / "*.jpg"))])
        if not self.files:
            raise RuntimeError(f"No .jpg files found in {images_dir}")

        self.ann_index: Dict[str, List[Dict[str, Any]]] = build_image_to_annotations(label_json)

        self.device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        self.model, self.tf, self.model_meta = build_model_from_checkpoint(ckpt_path, self.device)

        self.idx = 0
        self._photo = None  # keep reference

        self._build_ui()
        self._bind_keys()
        self._update_view()

    def _build_ui(self):
        self.columnconfigure(0, weight=3)
        self.columnconfigure(1, weight=2)
        self.rowconfigure(0, weight=1)

        # Left: image
        self.image_label = ttk.Label(self)
        self.image_label.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        # Right: info panel
        right = ttk.Frame(self)
        right.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        right.rowconfigure(2, weight=1)
        right.columnconfigure(0, weight=1)

        self.title_lbl = ttk.Label(right, text="", font=("Segoe UI", 12, "bold"))
        self.title_lbl.grid(row=0, column=0, sticky="w")

        self.pred_lbl = ttk.Label(right, text="", font=("Segoe UI", 11))
        self.pred_lbl.grid(row=1, column=0, sticky="w", pady=(8, 8))

        self.ann_text = tk.Text(right, height=30, wrap="word")
        self.ann_text.grid(row=2, column=0, sticky="nsew")
        self.ann_text.configure(state="disabled")

        footer = ttk.Label(
            right,
            text="Keys: ← / → to navigate, Esc to quit",
            font=("Segoe UI", 9),
        )
        footer.grid(row=3, column=0, sticky="w", pady=(8, 0))

    def _bind_keys(self):
        self.bind("<Left>", lambda e: self._prev())
        self.bind("<Right>", lambda e: self._next())
        self.bind("<Escape>", lambda e: self.destroy())

    def _prev(self):
        self.idx = (self.idx - 1) % len(self.files)
        self._update_view()

    def _next(self):
        self.idx = (self.idx + 1) % len(self.files)
        self._update_view()

    def _update_view(self):
        img_path = self.files[self.idx]
        fname = img_path.name

        # --- model prediction ---
        pred, prob_dict = predict_image(self.model, self.tf, img_path, self.device)
        pred_name = CLASS_NAMES.get(pred, str(pred))
        probs_str = " | ".join([f"{k}:{prob_dict[k]*100:5.1f}%" for k in sorted(prob_dict.keys())])

        self.title_lbl.config(text=f"[{self.idx+1}/{len(self.files)}] {fname}")
        self.pred_lbl.config(text=f"Prediction: {pred_name}\nProbs: {probs_str}\nDevice: {self.device}")

        # --- annotations display ---
        anns = self.ann_index.get(fname, [])
        lines = []
        if not anns:
            lines.append("No annotations found for this image in label.json.")
        else:
            lines.append(f"Annotations ({len(anns)}):")
            # nice stable ordering: by completed_by
            anns_sorted = sorted(anns, key=lambda a: (a.get("completed_by") is None, a.get("completed_by")))
            for a in anns_sorted:
                aid = a.get("completed_by")
                g = a.get("grade")
                c = a.get("confidence_grade")
                g_name = CLASS_NAMES.get(g, str(g))
                lines.append(f" - annotator {aid}: grade={g_name}, confidence={c}")

        self.ann_text.configure(state="normal")
        self.ann_text.delete("1.0", tk.END)
        self.ann_text.insert(tk.END, "\n".join(lines))
        self.ann_text.configure(state="disabled")

        # --- image display (scaled to fit) ---
        pil = Image.open(img_path).convert("RGB")
        max_w, max_h = 800, 760
        pil.thumbnail((max_w, max_h))
        self._photo = ImageTk.PhotoImage(pil)
        self.image_label.config(image=self._photo)


def resolve_default_checkpoint(project_root: Path) -> Path:
    ckpt_dir = project_root / "checkpoints"
    preferred = ckpt_dir / "dinov2_linear_probe_best.pt"
    if preferred.exists():
        return preferred

    best_candidates = list(ckpt_dir.glob("*_best.pt"))
    if best_candidates:
        return max(best_candidates, key=lambda p: p.stat().st_mtime)

    fallback = ckpt_dir / "dinov2_linear_probe.pt"
    if fallback.exists():
        return fallback

    other_candidates = list(ckpt_dir.glob("*.pt"))
    if other_candidates:
        return max(other_candidates, key=lambda p: p.stat().st_mtime)

    return preferred


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    images_dir = project_root / "data" / "images"
    label_json = project_root / "data" / "label.json"
    ckpt_path = resolve_default_checkpoint(project_root)

    app = ImageModelViewer(images_dir=images_dir, label_json=label_json, ckpt_path=ckpt_path)
    app.mainloop()


if __name__ == "__main__":
    main()
