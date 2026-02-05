import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

NUM_CLASSES = 5  # grades 0..4


def load_labelstudio_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected list at top-level, got {type(data)}")
    return data


def _get_choice(result_item: Dict[str, Any]) -> Optional[str]:
    v = result_item.get("value", {})
    choices = v.get("choices")
    if isinstance(choices, list) and len(choices) > 0:
        return str(choices[0])
    return None


def _parse_leading_int(s: Optional[str]) -> Optional[int]:
    if s is None:
        return None
    m = re.match(r"\s*(\d+)", s)
    return int(m.group(1)) if m else None


def parse_single_annotation(ann: Dict[str, Any]) -> Dict[str, Any]:
    if ann.get("was_cancelled", False):
        return {"valid": False}

    extracted = {
        "valid": True,
        "completed_by": ann.get("completed_by"),
        "grade": None,  # 0..4
        "confidence_grade": None,  # 0..4
    }

    by_from_name: Dict[str, Dict[str, Any]] = {}
    for item in ann.get("result", []):
        fn = item.get("from_name")
        if isinstance(fn, str):
            by_from_name[fn] = item

    # grade (e.g. "0 - Clean", "1 - Microfouling")
    if "grade" in by_from_name:
        extracted["grade"] = _parse_leading_int(_get_choice(by_from_name["grade"]))

    # confidence for grade (your export uses "confidence")
    for key in ["confidence", "grade_confidence", "confidence_grade"]:
        if key in by_from_name:
            extracted["confidence_grade"] = _parse_leading_int(_get_choice(by_from_name[key]))
            break

    return extracted


def parse_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    image_filename = entry.get("data", {}).get("image")
    parsed_anns: List[Dict[str, Any]] = []

    for ann in entry.get("annotations", []):
        pa = parse_single_annotation(ann)
        if pa.get("valid"):
            parsed_anns.append(pa)

    return {"image": image_filename, "annotations": parsed_anns}


def soft_label_from_annotations(anns, num_classes: int = NUM_CLASSES):
    """
    Soft label distribution over classes 0..4.
    Confidence-weighted, but robust to your true confidence scale.

    Weighting:
      - if confidence is present: w = conf / max_conf_in_image
      - else w = 1.0
    """
    counts = np.zeros(num_classes, dtype=np.float32)

    # find max confidence present in this image (to normalize)
    confs = [
        a.get("confidence_grade")
        for a in anns
        if isinstance(a.get("confidence_grade"), int) and a.get("confidence_grade") >= 0
    ]
    max_conf = max(confs) if confs else None

    for a in anns:
        g = a.get("grade")
        c = a.get("confidence_grade")

        if g is None or not (0 <= g < num_classes):
            continue

        if isinstance(c, int) and c >= 0 and max_conf and max_conf > 0:
            w = float(c) / float(max_conf)  # normalized to [0,1]
            if w <= 0:
                w = 0.1  # avoid 0-weight if someone uses 0
        else:
            w = 1.0

        counts[g] += w

    if counts.sum() == 0:
        return np.ones(num_classes, dtype=np.float32) / num_classes

    return counts / counts.sum()
