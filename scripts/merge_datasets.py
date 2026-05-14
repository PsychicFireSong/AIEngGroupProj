import argparse
import random
import re
import shutil
from pathlib import Path

import yaml

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_label(value: str) -> str:
    if value is None:
        return ""
    value = value.strip().lower()
    value = value.replace("_", " ").replace("-", " ")
    value = " ".join(value.split())
    return value


def sanitize_name(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip())
    value = value.strip("_")
    return value.lower() or "dataset"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_synonym_map(config: dict) -> dict:
    mapping = {}
    for unified, names in (config.get("synonyms") or {}).items():
        for name in names or []:
            key = normalize_label(name)
            if key:
                mapping[key] = unified
    return mapping


def collect_samples(dataset: dict, config: dict, synonym_map: dict) -> list:
    root = Path(dataset["path"]).resolve()
    data_path = root / "data.yaml"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data.yaml in {root}")

    data = load_yaml(data_path)
    names = data.get("names", [])
    name_lookup = [normalize_label(name) for name in names]

    samples = []
    keep_empty = bool(config.get("keep_empty"))
    dataset_key = sanitize_name(dataset.get("name", root.name))

    for split in ("train", "valid", "test"):
        images_dir = root / split / "images"
        labels_dir = root / split / "labels"
        if not images_dir.exists() or not labels_dir.exists():
            continue

        for image_path in images_dir.iterdir():
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue

            label_path = labels_dir / f"{image_path.stem}.txt"
            mapped_lines = []
            if label_path.exists():
                with label_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        try:
                            class_id = int(parts[0])
                        except ValueError:
                            continue
                        if class_id < 0 or class_id >= len(name_lookup):
                            continue
                        original_label = name_lookup[class_id]
                        unified = synonym_map.get(original_label)
                        if not unified:
                            continue
                        unified_id = config["classes"].index(unified)
                        mapped_lines.append(" ".join([str(unified_id)] + parts[1:]))

            if mapped_lines or keep_empty:
                samples.append(
                    {
                        "dataset_key": dataset_key,
                        "image_path": image_path,
                        "labels": mapped_lines,
                    }
                )

    return samples


def split_samples(samples: list, split_cfg: dict, seed: int) -> dict:
    total = len(samples)
    if total == 0:
        return {"train": [], "val": [], "test": []}

    train_ratio = float(split_cfg.get("train", 0.8))
    val_ratio = float(split_cfg.get("val", 0.1))

    rng = random.Random(seed)
    rng.shuffle(samples)

    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)

    train_samples = samples[:train_count]
    val_samples = samples[train_count : train_count + val_count]
    test_samples = samples[train_count + val_count :]

    return {"train": train_samples, "val": val_samples, "test": test_samples}


def write_output(samples_by_split: dict, output_dir: Path, classes: list) -> dict:
    counts = {name: 0 for name in classes}

    for split, samples in samples_by_split.items():
        images_out = output_dir / "images" / split
        labels_out = output_dir / "labels" / split
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)

        for sample in samples:
            image_path = sample["image_path"]
            dataset_key = sample["dataset_key"]
            out_name = f"{dataset_key}__{image_path.name}"

            shutil.copy2(image_path, images_out / out_name)

            label_path = labels_out / f"{Path(out_name).stem}.txt"
            with label_path.open("w", encoding="utf-8") as handle:
                for line in sample["labels"]:
                    handle.write(f"{line}\n")
                    class_id = int(line.split()[0])
                    counts[classes[class_id]] += 1

    return counts


def write_data_yaml(output_dir: Path, classes: list) -> None:
    data = {
        "path": str(output_dir.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": classes,
    }
    with (output_dir / "data.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge YOLO datasets into a unified set.")
    parser.add_argument("--config", default="configs/merge_config.yaml", help="Path to merge config")
    parser.add_argument("--force", action="store_true", help="Overwrite output directory")
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
    all_samples = []

    for dataset in config.get("datasets", []):
        samples = collect_samples(dataset, config, synonym_map)
        all_samples.extend(samples)

    split_cfg = config.get("split", {})
    samples_by_split = split_samples(all_samples, split_cfg, int(config.get("seed", 42)))

    class_counts = write_output(samples_by_split, output_dir, config["classes"])
    write_data_yaml(output_dir, config["classes"])

    total_images = sum(len(items) for items in samples_by_split.values())
    print(f"Total images: {total_images}")
    print("Images by split:")
    for split, items in samples_by_split.items():
        print(f"  {split}: {len(items)}")
    print("Label counts:")
    for name, count in class_counts.items():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
