import json
from pathlib import Path

label_path = Path("data/label.json")
images_dir = Path("data/images")

with open(label_path, "r") as f:
    data = json.load(f)

total = len(data)
exists = 0
size_zero = 0
missing = 0

for entry in data:
    fname = entry.get("data", {}).get("image")
    if not fname:
        continue
    img_path = images_dir / fname
    if img_path.exists():
        if img_path.stat().st_size > 0:
            exists += 1
        else:
            size_zero += 1
    else:
        missing += 1

print(f"Total entries in JSON: {total}")
print(f"Valid images (existent + size>0): {exists}")
print(f"Corrupted images (existent + size=0): {size_zero}")
print(f"Missing images: {missing}")
