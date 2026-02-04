from pathlib import Path
from typing import Any, Dict, List

from src.data.labelstudio_parser import load_labelstudio_json, parse_single_annotation, parse_entry


def build_image_to_annotations(label_json_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Returns mapping:
      filename -> list of parsed annotator dicts (grade, confidence_grade, completed_by, ...)
    """
    entries = load_labelstudio_json(label_json_path)
    idx: Dict[str, List[Dict[str, Any]]] = {}

    for e in entries:
        pe = parse_entry(e)
        fname = pe.get("image")
        if not fname:
            continue

        # parse_entry already returns parsed annotations in our earlier code;
        # but to be safe, we rebuild them here by reusing parse_single_annotation on raw
        parsed = []
        for ann in e.get("annotations", []):
            pa = parse_single_annotation(ann)
            if pa.get("valid"):
                parsed.append(pa)

        idx[fname] = parsed

    return idx
