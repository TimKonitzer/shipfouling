import json
from pathlib import Path

def main():
    label_path = Path("../data/label.json")

    with label_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # label.json kann entweder eine Liste oder ein Dict sein.
    if isinstance(data, list):
        print("JSON ist eine Liste. Erster Eintrag:")
        print(json.dumps(data[0], indent=2, ensure_ascii=False))
    elif isinstance(data, dict):
        first_key = next(iter(data))
        print(f"JSON ist ein Dict. Erster Key: {first_key}")
        print("Erster Eintrag (value):")
        print(json.dumps(data[first_key], indent=2, ensure_ascii=False))
    else:
        raise TypeError(f"Unerwarteter JSON-Typ: {type(data)}")

if __name__ == "__main__":
    main()
