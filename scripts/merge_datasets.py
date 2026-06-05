from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCE_TO_OUTPUT_SPLIT = {
    "train": "train",
    "valid": "val",
    "val": "val",
    "test": "test",
}


def normalize_label(value: str) -> str:
    if value is None:
        return ""
    value = str(value).strip().lower()
    value = value.replace("_", " ").replace("-", " ")
    return " ".join(value.split())


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip())
    return value.strip("_").lower() or "dataset"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def read_names(data: dict) -> list[str]:
    names = data.get("names", [])
    if isinstance(names, dict):
        ordered: list[str] = []
        for key, value in sorted(
            names.items(),
            key=lambda item: (0, int(item[0])) if str(item[0]).isdigit() else (1, str(item[0])),
        ):
            if not str(key).isdigit():
                continue
            index = int(key)
            while len(ordered) <= index:
                ordered.append("")
            ordered[index] = str(value)
        return ordered
    return [str(name) for name in names]


def build_synonym_map(config: dict) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for unified, names in (config.get("synonyms") or {}).items():
        mapping[normalize_label(unified)] = unified
        for name in names or []:
            key = normalize_label(name)
            if key:
                mapping[key] = unified
    return mapping


def resolve_dataset_root(dataset: dict) -> Path:
    root = Path(dataset["path"])
    if not root.is_absolute():
        root = Path.cwd() / root
    return root.resolve()


def collect_samples(dataset: dict, config: dict, synonym_map: dict[str, str], errors: Counter) -> list[dict]:
    root = resolve_dataset_root(dataset)
    data_path = root / "data.yaml"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data.yaml in {root}")

    data = load_yaml(data_path)
    names = read_names(data)
    samples = []
    keep_empty = bool(config.get("keep_empty"))
    dataset_key = sanitize_name(dataset.get("name", root.name))

    for source_split, output_split in SOURCE_TO_OUTPUT_SPLIT.items():
        images_dir = root / source_split / "images"
        labels_dir = root / source_split / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            continue

        for image_path in sorted(images_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue

            label_path = labels_dir / f"{image_path.stem}.txt"
            mapped_lines = []
            if label_path.exists():
                with label_path.open("r", encoding="utf-8") as handle:
                    for raw_line in handle:
                        line = raw_line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) < 5:
                            errors[f"{dataset_key}:malformed_label"] += 1
                            continue
                        try:
                            class_id = int(float(parts[0]))
                        except ValueError:
                            errors[f"{dataset_key}:bad_class_id"] += 1
                            continue
                        if class_id < 0 or class_id >= len(names):
                            errors[f"{dataset_key}:class_id_out_of_range:{class_id}"] += 1
                            continue

                        original_label = normalize_label(names[class_id])
                        unified = synonym_map.get(original_label)
                        if not unified:
                            errors[f"{dataset_key}:unmapped:{original_label or '<blank>'}"] += 1
                            continue

                        unified_id = config["classes"].index(unified)
                        mapped_lines.append(" ".join([str(unified_id)] + parts[1:]))

            if mapped_lines or keep_empty:
                samples.append(
                    {
                        "dataset_key": dataset_key,
                        "image_path": image_path,
                        "labels": mapped_lines,
                        "source_split": source_split,
                        "output_split": output_split,
                    }
                )
    return samples


def dataset_metadata(dataset: dict) -> dict:
    root = resolve_dataset_root(dataset)
    data_path = root / "data.yaml"
    names: list[str] = []
    if data_path.exists():
        names = read_names(load_yaml(data_path))

    image_counts = {}
    label_file_counts = {}
    for source_split in SOURCE_TO_OUTPUT_SPLIT:
        images_dir = root / source_split / "images"
        labels_dir = root / source_split / "labels"
        image_counts[source_split] = len([path for path in images_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS]) if images_dir.exists() else 0
        label_file_counts[source_split] = len(list(labels_dir.glob("*.txt"))) if labels_dir.exists() else 0

    return {
        "name": dataset.get("name", root.name),
        "path": str(root),
        "class_names": names,
        "source_image_files": image_counts,
        "source_label_files": label_file_counts,
    }


def summarize_samples(samples: list[dict], classes: list[str]) -> dict:
    split_counts = Counter()
    label_counts = Counter({name: 0 for name in classes})
    for sample in samples:
        split_counts[sample["output_split"]] += 1
        for line in sample["labels"]:
            class_id = int(line.split()[0])
            if 0 <= class_id < len(classes):
                label_counts[classes[class_id]] += 1
    return {
        "mapped_images": len(samples),
        "mapped_images_by_split": dict(split_counts),
        "mapped_labels": dict(label_counts),
    }


def split_samples(samples: list[dict], split_cfg: dict, seed: int) -> dict[str, list[dict]]:
    if not samples:
        return {"train": [], "val": [], "test": []}

    train_ratio = float(split_cfg.get("train", 0.8))
    val_ratio = float(split_cfg.get("val", 0.1))
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)

    train_count = int(len(shuffled) * train_ratio)
    val_count = int(len(shuffled) * val_ratio)
    return {
        "train": shuffled[:train_count],
        "val": shuffled[train_count : train_count + val_count],
        "test": shuffled[train_count + val_count :],
    }


def preserve_samples(samples: list[dict]) -> dict[str, list[dict]]:
    output = {"train": [], "val": [], "test": []}
    for sample in samples:
        output[sample["output_split"]].append(sample)
    return output


def write_output(samples_by_split: dict[str, list[dict]], output_dir: Path, classes: list[str]) -> dict:
    label_counts = Counter({name: 0 for name in classes})
    image_counts = Counter()

    for split, samples in samples_by_split.items():
        images_out = output_dir / "images" / split
        labels_out = output_dir / "labels" / split
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        for sample in samples:
            image_path = sample["image_path"]
            dataset_key = sample["dataset_key"]
            out_name = f"{dataset_key}__{sample['source_split']}__{image_path.name}"
            shutil.copy2(image_path, images_out / out_name)

            label_path = labels_out / f"{Path(out_name).stem}.txt"
            with label_path.open("w", encoding="utf-8") as handle:
                for line in sample["labels"]:
                    handle.write(f"{line}\n")
                    class_id = int(line.split()[0])
                    label_counts[classes[class_id]] += 1
            image_counts[split] += 1

    return {"images": dict(image_counts), "labels": dict(label_counts)}


def write_data_yaml(output_dir: Path, classes: list[str]) -> None:
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": classes,
    }
    with (output_dir / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def print_error_summary(errors: Counter) -> None:
    if not errors:
        print("No label mapping warnings.")
        return
    print("\nWARNING: Some labels could not be merged. Top unmapped/error counts:")
    for label, count in errors.most_common(30):
        print(f"  {label}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge YOLO datasets into a unified five-class set.")
    parser.add_argument("--config", default="configs/merge_config.yaml", help="Path to merge config")
    parser.add_argument("--force", action="store_true", help="Overwrite output directory")
    parser.add_argument(
        "--preserve-splits",
        action="store_true",
        help="Keep each source train/valid/test split mapped into the merged train/val/test folders.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    output_dir = Path(config.get("output_dir", "merged_dataset"))
    if not output_dir.is_absolute():
        output_dir = (Path.cwd() / output_dir).resolve()

    if output_dir.exists() and any(output_dir.iterdir()):
        if args.force:
            shutil.rmtree(output_dir)
        else:
            raise SystemExit(f"Output directory not empty: {output_dir}. Use --force to overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)

    synonym_map = build_synonym_map(config)
    errors: Counter = Counter()
    all_samples: list[dict] = []
    source_audit: list[dict] = []

    for dataset in config.get("datasets", []):
        audit = dataset_metadata(dataset)
        samples = collect_samples(dataset, config, synonym_map, errors)
        audit.update(summarize_samples(samples, config["classes"]))
        source_audit.append(audit)
        all_samples.extend(samples)

    if args.preserve_splits:
        samples_by_split = preserve_samples(all_samples)
    else:
        samples_by_split = split_samples(all_samples, config.get("split", {}), int(config.get("seed", 42)))

    counts = write_output(samples_by_split, output_dir, config["classes"])
    write_data_yaml(output_dir, config["classes"])

    summary = {
        "preserve_splits": bool(args.preserve_splits),
        "total_images": sum(len(items) for items in samples_by_split.values()),
        "images_by_split": {split: len(items) for split, items in samples_by_split.items()},
        "counts": counts,
        "source_audit": source_audit,
        "mapping_warnings": dict(errors),
    }
    (output_dir / "merge_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print_error_summary(errors)


if __name__ == "__main__":
    main()
