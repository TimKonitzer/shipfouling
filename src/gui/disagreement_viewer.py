import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk


NUM_CLASSES = 5

CLASS_NAMES = {
    0: "0 - Clean",
    1: "1 - Microfouling",
    2: "2",
    3: "3",
    4: "4",
}


def _get_choice(item: Dict[str, Any]) -> Optional[str]:
    v = item.get("value", {})
    choices = v.get("choices")
    if isinstance(choices, list) and choices:
        return str(choices[0])
    return None


def _parse_leading_int(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    m = re.match(r"\s*(\d+)", s)
    return int(m.group(1)) if m else None


def parse_annotation(ann: Dict[str, Any]) -> Optional[Tuple[int, Optional[int], Optional[int]]]:
    if ann.get("was_cancelled", False):
        return None

    by_from = {}
    for item in ann.get("result", []):
        fn = item.get("from_name")
        if isinstance(fn, str):
            by_from[fn] = item

    if "grade" not in by_from:
        return None

    grade = _parse_leading_int(_get_choice(by_from["grade"]))
    if grade is None or not (0 <= grade < NUM_CLASSES):
        return None

    conf = None
    for key in ["confidence", "grade_confidence", "confidence_grade"]:
        if key in by_from:
            conf = _parse_leading_int(_get_choice(by_from[key]))
            break
    if conf is not None and conf < 0:
        conf = None

    completed_by = ann.get("completed_by")
    if not isinstance(completed_by, int):
        completed_by = None

    return grade, conf, completed_by


def soft_dist(votes: List[Tuple[int, Optional[int]]]) -> List[float]:
    counts = [0.0 for _ in range(NUM_CLASSES)]
    confs = [c for (_g, c) in votes if isinstance(c, int) and c >= 0]
    max_conf = max(confs) if confs else None

    for g, c in votes:
        if c is not None and c >= 0 and max_conf and max_conf > 0:
            w = float(c) / float(max_conf)
            if w <= 0:
                w = 0.1
        else:
            w = 1.0
        counts[g] += w

    total = sum(counts)
    if total == 0:
        return [1.0 / NUM_CLASSES for _ in range(NUM_CLASSES)]
    return [c / total for c in counts]


def disagreement_score(votes: List[Tuple[int, Optional[int]]]) -> float:
    if not votes:
        return -1.0
    p = soft_dist(votes)
    ent = 0.0
    for v in p:
        if v > 0:
            ent -= v * math.log2(v)
    return ent


class DisagreementViewer(tk.Tk):
    def __init__(self, images_dir: Path, label_json: Path, output_json: Path):
        super().__init__()
        self.title("Ship Fouling Disagreement Review")
        self.geometry("1300x820")

        self.images_dir = images_dir
        self.label_json = label_json
        self.output_json = output_json

        self.entries = self._load_entries(label_json)
        self.items = self._build_items()

        if not self.items:
            raise RuntimeError("No images with annotations found in label.json.")

        self.idx = 0
        self._photo = None
        self.current_ann_map: List[int] = []

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
        for idx, entry in enumerate(self.entries):
            image_name = (entry.get("data", {}) or {}).get("image", None)
            if not image_name:
                continue
            votes = self._collect_votes(entry)
            score = disagreement_score(votes)
            items.append(
                {
                    "entry_idx": idx,
                    "image": image_name,
                    "score": score,
                    "vote_count": len(votes),
                }
            )

        items.sort(key=lambda x: x["score"], reverse=True)
        return items

    def _collect_votes(self, entry: Dict[str, Any]) -> List[Tuple[int, Optional[int]]]:
        votes = []
        for ann in entry.get("annotations", []):
            parsed = parse_annotation(ann)
            if parsed is None:
                continue
            grade, conf, _completed_by = parsed
            votes.append((grade, conf))
        return votes

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

        self.ann_list = tk.Listbox(right, height=16)
        self.ann_list.grid(row=2, column=0, sticky="nsew")

        btn_row = ttk.Frame(right)
        btn_row.grid(row=3, column=0, sticky="w", pady=(8, 8))

        self.delete_btn = ttk.Button(btn_row, text="Delete selected annotation", command=self._delete_selected)
        self.delete_btn.grid(row=0, column=0, padx=(0, 8))

        self.save_btn = ttk.Button(btn_row, text="Save copy", command=self._save_copy)
        self.save_btn.grid(row=0, column=1, padx=(0, 8))

        self.save_lbl = ttk.Label(right, text=f"Output: {self.output_json}", font=("Segoe UI", 9))
        self.save_lbl.grid(row=4, column=0, sticky="w")

        footer = ttk.Label(
            right,
            text="Keys: Left/Right to navigate, Delete to remove, S to save, Esc to quit",
            font=("Segoe UI", 9),
        )
        footer.grid(row=5, column=0, sticky="w", pady=(8, 0))

    def _bind_keys(self):
        self.bind("<Left>", lambda _e: self._prev())
        self.bind("<Right>", lambda _e: self._next())
        self.bind("<Escape>", lambda _e: self.destroy())
        self.bind("<Delete>", lambda _e: self._delete_selected())
        self.bind("s", lambda _e: self._save_copy())
        self.bind("S", lambda _e: self._save_copy())

    def _prev(self):
        self.idx = (self.idx - 1) % len(self.items)
        self._update_view()

    def _next(self):
        self.idx = (self.idx + 1) % len(self.items)
        self._update_view()

    def _update_view(self):
        item = self.items[self.idx]
        entry = self.entries[item["entry_idx"]]
        img_path = self.images_dir / item["image"]

        self.title_lbl.config(text=f"[{self.idx+1}/{len(self.items)}] {item['image']}")
        self.meta_lbl.config(
            text=f"Disagreement (entropy): {item['score']:.3f} | Votes: {item['vote_count']}"
        )

        self.ann_list.delete(0, tk.END)
        self.current_ann_map = []
        for i, ann in enumerate(entry.get("annotations", [])):
            parsed = parse_annotation(ann)
            if parsed is None:
                self.ann_list.insert(tk.END, f"[{i}] invalid annotation")
                self.current_ann_map.append(i)
                continue
            grade, conf, completed_by = parsed
            grade_name = CLASS_NAMES.get(grade, str(grade))
            self.ann_list.insert(
                tk.END,
                f"[{i}] annotator {completed_by} | grade={grade_name} | conf={conf}",
            )
            self.current_ann_map.append(i)

        pil = Image.open(img_path).convert("RGB")
        max_w, max_h = 820, 780
        pil.thumbnail((max_w, max_h))
        self._photo = ImageTk.PhotoImage(pil)
        self.image_label.config(image=self._photo)

    def _delete_selected(self):
        if not self.current_ann_map:
            return
        sel = self.ann_list.curselection()
        if not sel:
            return
        row = sel[0]
        ann_index = self.current_ann_map[row]

        item = self.items[self.idx]
        entry = self.entries[item["entry_idx"]]
        anns = entry.get("annotations", [])
        if 0 <= ann_index < len(anns):
            del anns[ann_index]

        self._refresh_item(item["entry_idx"])
        self._update_view()

    def _refresh_item(self, entry_idx: int):
        for item in self.items:
            if item["entry_idx"] == entry_idx:
                entry = self.entries[entry_idx]
                votes = self._collect_votes(entry)
                item["score"] = disagreement_score(votes)
                item["vote_count"] = len(votes)
                break

    def _save_copy(self):
        self.output_json.parent.mkdir(parents=True, exist_ok=True)
        with self.output_json.open("w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)
        self.save_lbl.config(text=f"Saved: {self.output_json}")


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    images_dir = project_root / "data" / "images"
    label_json = project_root / "data" / "label.json"
    output_json = project_root / "data" / "label_reviewed.json"

    app = DisagreementViewer(images_dir=images_dir, label_json=label_json, output_json=output_json)
    app.mainloop()


if __name__ == "__main__":
    main()
