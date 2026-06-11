from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

REQUIRED_CLASSES = ["crack", "spalling", "corrosion", "pothole", "paint_degradation"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def read_names(data: dict) -> list[str]:
    names = data.get("names", [])
    if isinstance(names, dict):
        ordered: list[str] = []
        for key, value in sorted(names.items(), key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else 9999):
            if not str(key).isdigit():
                continue
            idx = int(key)
            while len(ordered) <= idx:
                ordered.append("")
            ordered[idx] = str(value)
        return ordered
    return [str(n) for n in names]


def image_label_dirs(root: Path, split: str) -> tuple[Path, Path]:
    if (root / split / "images").exists():
        return root / split / "images", root / split / "labels"
    return root / "images" / split, root / "labels" / split


def count_labels_sampled(dataset: Path, names: list[str], max_per_class_split: int) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "valid", "test"):
        img_dir, lbl_dir = image_label_dirs(dataset, split)
        if not lbl_dir.exists():
            continue
        label_files = list(lbl_dir.glob("*.txt"))
        random.shuffle(label_files)

        split_key = "val" if split in ("val", "valid") else split
        if split_key not in counts:
            counts[split_key] = {n: 0 for n in REQUIRED_CLASSES}

        per_class_seen: dict[str, int] = {n: 0 for n in REQUIRED_CLASSES}
        for lf in label_files:
            for line in lf.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                try:
                    cid = int(float(parts[0]))
                except ValueError:
                    continue
                if 0 <= cid < len(names):
                    cname = names[cid]
                    unified = next((rc for rc in REQUIRED_CLASSES if rc == cname), None)
                    if unified and per_class_seen.get(unified, 0) < max_per_class_split:
                        counts[split_key][unified] = counts[split_key].get(unified, 0) + 1
                        per_class_seen[unified] = per_class_seen.get(unified, 0) + 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Sampled scenario coverage audit for stage 1 dataset.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-per-class-split", type=int, default=300)
    parser.add_argument("--min-train-samples", type=int, default=120)
    parser.add_argument("--min-val-samples", type=int, default=25)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    data_yaml = dataset / "data.yaml"
    if not data_yaml.exists():
        result = {"error": f"data.yaml not found at {data_yaml}", "gate": {"ok": False}}
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return  # exit 0 — let caller decide what to do

    names = read_names(load_yaml(data_yaml))
    counts = count_labels_sampled(dataset, names, args.max_per_class_split)

    train = counts.get("train", {})
    val = counts.get("val", {})

    train_ok = {n: train.get(n, 0) >= args.min_train_samples for n in REQUIRED_CLASSES}
    val_ok = {n: val.get(n, 0) >= args.min_val_samples for n in REQUIRED_CLASSES}
    gate_ok = all(train_ok.values()) and all(val_ok.values())

    result = {
        "dataset": str(dataset),
        "max_per_class_split": args.max_per_class_split,
        "min_train_samples": args.min_train_samples,
        "min_val_samples": args.min_val_samples,
        "sampled_counts": counts,
        "gate": {
            "ok": gate_ok,
            "train_ok": train_ok,
            "val_ok": val_ok,
            "train_failing": [n for n, ok in train_ok.items() if not ok],
            "val_failing": [n for n, ok in val_ok.items() if not ok],
        },
    }

    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    # Always exit 0 — weak class decisions are made by the caller, not this audit.


if __name__ == "__main__":
    main()
