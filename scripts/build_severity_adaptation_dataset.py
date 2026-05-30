from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEVERITIES = ("minor", "moderate", "critical")
SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def ensure_structure(root: Path) -> None:
    for split in ("train", "val", "test"):
        for severity in SEVERITIES:
            (root / split / severity).mkdir(parents=True, exist_ok=True)


def copy_classification_dataset(source: Path, output: Path, prefix: str, repeat_train: int = 1) -> Counter:
    counts: Counter = Counter()
    if not source.exists():
        return counts
    for source_split, output_split in SPLIT_ALIASES.items():
        for severity in SEVERITIES:
            src_dir = source / source_split / severity
            if not src_dir.exists():
                continue
            dst_dir = output / output_split / severity
            copies = repeat_train if output_split == "train" else 1
            for image_path in src_dir.iterdir():
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                    continue
                for copy_index in range(max(copies, 1)):
                    suffix = f"_r{copy_index}" if copies > 1 else ""
                    output_name = f"{prefix}_{output_split}_{severity}_{image_path.stem}{suffix}{image_path.suffix.lower()}"
                    shutil.copy2(image_path, dst_dir / output_name)
                    counts[f"{prefix}/{output_split}/{severity}"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a severity classifier adaptation dataset.")
    parser.add_argument("--base", default="severity_dataset", help="Base severity classification dataset")
    parser.add_argument(
        "--target",
        default="domain_adaptation/severity_extra",
        help="Extra manually sorted severity crops in train|valid|val|test/minor|moderate|critical folders",
    )
    parser.add_argument("--output", default="severity_dataset_domain_adapted")
    parser.add_argument("--target-repeat", type=int, default=3)
    args = parser.parse_args()

    output = Path(args.output)
    clean_dir(output)
    ensure_structure(output)

    counts = Counter()
    counts.update(copy_classification_dataset(Path(args.base), output, "base", 1))
    counts.update(copy_classification_dataset(Path(args.target), output, "target", max(args.target_repeat, 1)))

    summary = {
        "base": args.base,
        "target": args.target,
        "output": args.output,
        "counts": dict(counts),
    }
    (output / "severity_adaptation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
