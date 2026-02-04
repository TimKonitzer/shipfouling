import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


NUM_CLASSES = 5


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
    """
    Returns (grade, confidence_grade, completed_by) or None if invalid.
    """
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
    if conf is not None and not (0 <= conf <= 5):
        conf = None

    completed_by = ann.get("completed_by")
    if not isinstance(completed_by, int):
        completed_by = None

    return grade, conf, completed_by


def entropy(probs: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(probs, eps, 1.0)
    return float(-(p * np.log2(p)).sum())


def soft_dist(votes: List[Tuple[int, Optional[int]]]) -> np.ndarray:
    counts = np.zeros(NUM_CLASSES, dtype=np.float64)
    confs = [c for (_g, c) in votes if isinstance(c, int)]
    max_conf = max(confs) if confs else None

    for g, c in votes:
        if c is not None and max_conf and max_conf > 0:
            w = float(c) / float(max_conf)
            if w <= 0:
                w = 0.1
        else:
            w = 1.0
        counts[g] += w

    if counts.sum() == 0:
        return np.ones(NUM_CLASSES) / NUM_CLASSES
    return counts / counts.sum()



def main():
    project_root = Path(__file__).resolve().parent.parent
    label_path = project_root / "data" / "label.json"

    with label_path.open("r", encoding="utf-8") as f:
        entries = json.load(f)

    if not isinstance(entries, list):
        raise TypeError(f"Expected list in label.json, got {type(entries)}")

    # --- Global counters ---
    raw_vote_counts = Counter()            # counts all annotator votes
    conf_weighted_counts = np.zeros(NUM_CLASSES, dtype=np.float64)  # weighted vote mass per class

    annotations_per_image = []
    image_entropy = []
    image_majority_agreement = []  # fraction of votes that equal majority class per image

    annotator_counts = Counter()            # number of annotations per annotator
    annotator_grade_counts = defaultdict(Counter)  # per annotator class distribution

    num_images = 0
    num_images_with_any_vote = 0
    num_total_annotations = 0
    num_valid_annotations = 0

    for e in entries:
        num_images += 1
        image_name = (e.get("data", {}) or {}).get("image", None)

        votes: List[Tuple[int, Optional[int]]] = []
        vote_annotators: List[Optional[int]] = []

        anns = e.get("annotations", [])
        if not isinstance(anns, list):
            continue

        num_total_annotations += len(anns)

        for ann in anns:
            parsed = parse_annotation(ann)
            if parsed is None:
                continue
            grade, conf, completed_by = parsed
            num_valid_annotations += 1
            votes.append((grade, conf))
            vote_annotators.append(completed_by)

            raw_vote_counts[grade] += 1
            w = (conf + 1) / 6.0 if conf is not None else 1.0
            conf_weighted_counts[grade] += w

            if completed_by is not None:
                annotator_counts[completed_by] += 1
                annotator_grade_counts[completed_by][grade] += 1

        annotations_per_image.append(len(votes))

        if len(votes) == 0:
            continue

        num_images_with_any_vote += 1

        # per-image soft distribution + entropy
        p = soft_dist(votes)
        image_entropy.append(entropy(p))

        # per-image agreement with majority
        hard_votes = [g for g, _c in votes]
        maj = Counter(hard_votes).most_common(1)[0][0]
        agree = sum(1 for g in hard_votes if g == maj) / len(hard_votes)
        image_majority_agreement.append(agree)

    # --- Print summary ---
    print("\n=== LABEL ANALYSIS SUMMARY ===")
    print(f"Images in JSON:               {num_images}")
    print(f"Images with >=1 valid vote:   {num_images_with_any_vote}")
    print(f"Total annotations (raw):      {num_total_annotations}")
    print(f"Valid grade annotations:      {num_valid_annotations}")

    if annotations_per_image:
        arr = np.array(annotations_per_image, dtype=np.float64)
        print("\n--- Annotations per image ---")
        print(f"mean: {arr.mean():.2f} | median: {np.median(arr):.2f} | min: {arr.min():.0f} | max: {arr.max():.0f}")
        print("counts (how many images have k votes):")
        c = Counter(int(x) for x in annotations_per_image)
        for k in sorted(c.keys()):
            print(f"  {k}: {c[k]}")

    # Raw class distribution (all votes)
    print("\n--- Class distribution (raw votes) ---")
    total_votes = sum(raw_vote_counts.values())
    for cls in range(NUM_CLASSES):
        cnt = raw_vote_counts.get(cls, 0)
        pct = (cnt / total_votes * 100) if total_votes else 0.0
        print(f"  {cls}: {cnt:6d}  ({pct:5.2f}%)")
    if total_votes:
        most_common_cls, mc_cnt = raw_vote_counts.most_common(1)[0]
        print(f"Most frequent class (raw): {most_common_cls} ({mc_cnt} votes)")

    # Confidence-weighted distribution
    print("\n--- Class distribution (confidence-weighted) ---")
    total_w = conf_weighted_counts.sum()
    for cls in range(NUM_CLASSES):
        w = conf_weighted_counts[cls]
        pct = (w / total_w * 100) if total_w else 0.0
        print(f"  {cls}: {w:8.2f}  ({pct:5.2f}%)")

    # Agreement / uncertainty
    if image_entropy:
        ent = np.array(image_entropy, dtype=np.float64)
        agr = np.array(image_majority_agreement, dtype=np.float64)

        # Max entropy for 6 classes is log2(6)
        max_ent = math.log2(NUM_CLASSES)

        print("\n--- Annotator disagreement / uncertainty (per image) ---")
        print(f"Entropy (0=complete agreement, {max_ent:.2f}=max uncertainty)")
        print(f"  mean entropy:   {ent.mean():.3f}")
        print(f"  median entropy: {np.median(ent):.3f}")
        print(f"  90th percentile:{np.quantile(ent, 0.9):.3f}")

        print("\nMajority agreement (fraction of votes matching per-image majority):")
        print(f"  mean:   {agr.mean():.3f}")
        print(f"  median: {np.median(agr):.3f}")
        print(f"  10th percentile:{np.quantile(agr, 0.1):.3f}")

        # How many images are "high disagreement"?
        # Example threshold: agreement < 0.67 (e.g., 2/3) OR entropy > 1.0
        high_dis = np.mean(agr < 2/3) * 100
        print(f"\nImages with majority-agreement < 0.67: {high_dis:.2f}%")

    # Annotator stats (optional)
    if annotator_counts:
        print("\n--- Annotator activity (top 10) ---")
        for aid, n in annotator_counts.most_common(10):
            print(f"  annotator {aid}: {n} labels")

        print("\n--- Annotator class tendencies (top 5 annotators) ---")
        top = [aid for aid, _n in annotator_counts.most_common(5)]
        for aid in top:
            total = sum(annotator_grade_counts[aid].values())
            dist = [annotator_grade_counts[aid].get(c, 0) / total * 100 if total else 0 for c in range(NUM_CLASSES)]
            dist_str = " ".join(f"{c}:{dist[c]:5.1f}%" for c in range(NUM_CLASSES))
            print(f"  annotator {aid} ({total} labels): {dist_str}")

    # --- Disagreement vs per-image majority by annotator ---
    annotator_total = Counter()
    annotator_disagree = Counter()

    for e in entries:
        anns = e.get("annotations", [])
        votes = []
        ann_by_annotator = []

        for ann in anns:
            parsed = parse_annotation(ann)
            if parsed is None:
                continue
            grade, conf, completed_by = parsed
            votes.append(grade)
            ann_by_annotator.append((completed_by, grade))

        if len(votes) == 0:
            continue

        maj = Counter(votes).most_common(1)[0][0]

        for aid, g in ann_by_annotator:
            if aid is None:
                continue
            annotator_total[aid] += 1
            if g != maj:
                annotator_disagree[aid] += 1

    print("\n--- Annotator disagreement vs per-image majority ---")
    rows = []
    for aid, tot in annotator_total.items():
        dis = annotator_disagree[aid]
        rate = dis / tot if tot else 0.0
        rows.append((rate, dis, tot, aid))

    # sort by disagreement rate descending
    rows_desc = sorted(rows, key=lambda x: x[0], reverse=True)
    # sort by disagreement rate ascending
    rows_asc = sorted(rows, key=lambda x: x[0])

    k = min(10, len(rows))

    print(f"top {k} most-disagreeing annotators:")
    for rate, dis, tot, aid in rows_desc[:k]:
        print(f"  annotator {aid}: disagree {dis}/{tot} = {rate * 100:.1f}%")

    print(f"top {k} most-agreeing annotators:")
    for rate, dis, tot, aid in rows_asc[:k]:
        print(f"  annotator {aid}: disagree {dis}/{tot} = {rate * 100:.1f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
