from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "test": "test"}
REQUIRED_NAMES = ["crack", "spalling", "corrosion", "pothole", "paint_degradation"]


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_yaml(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def image_label_dirs(root: Path, split: str) -> tuple[Path, Path]:
    direct_images = root / split / "images"
    direct_labels = root / split / "labels"
    if direct_images.exists():
        return direct_images, direct_labels
    return root / "images" / split, root / "labels" / split


def output_dirs(root: Path, split: str) -> tuple[Path, Path]:
    return root / "images" / split, root / "labels" / split


def ensure_output(root: Path) -> None:
    for split in ("train", "val", "test"):
        images_dir, labels_dir = output_dirs(root, split)
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)


def read_names(data_yaml: Path) -> list[str]:
    data = load_yaml(data_yaml)
    names = data.get("names", [])
    if isinstance(names, dict):
        ordered: list[str] = []
        for key, value in sorted(names.items(), key=lambda item: int(item[0]) if str(item[0]).isdigit() else 9999):
            if not str(key).isdigit():
                continue
            index = int(key)
            while len(ordered) <= index:
                ordered.append("")
            ordered[index] = str(value)
        return ordered
    return [str(name) for name in names]


def read_label_counts(label_path: Path, names: list[str]) -> Counter:
    counts: Counter = Counter()
    if not label_path.exists():
        return counts
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        if 0 <= class_id < len(names):
            counts[names[class_id]] += 1
    return counts


def collect_samples(root: Path, kind: str, names: list[str]) -> list[dict]:
    samples: list[dict] = []
    if not root.exists():
        return samples
    for source_split, output_split in SPLIT_ALIASES.items():
        images_dir, labels_dir = image_label_dirs(root, source_split)
        if not images_dir.exists():
            continue
        for image_path in sorted(images_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            label_path = labels_dir / f"{image_path.stem}.txt"
            class_counts = read_label_counts(label_path, names)
            samples.append(
                {
                    "id": f"{kind}:{output_split}:{image_path.name}",
                    "kind": kind,
                    "split": output_split,
                    "image_path": image_path,
                    "label_path": label_path,
                    "class_counts": dict(class_counts),
                }
            )
    return samples


def sample_fingerprint(samples: list[dict], params: dict) -> str:
    digest = hashlib.sha256(json.dumps(params, sort_keys=True).encode("utf-8"))
    for sample in sorted(samples, key=lambda item: item["id"]):
        image_path = Path(sample["image_path"])
        label_path = Path(sample["label_path"])
        digest.update(str(image_path).encode("utf-8"))
        digest.update(str(image_path.stat().st_size if image_path.exists() else 0).encode("utf-8"))
        digest.update(str(label_path.stat().st_size if label_path.exists() else 0).encode("utf-8"))
    return digest.hexdigest()


def copy_sample(sample: dict, output: Path, prefix: str, copy_index: int, counts: Counter, source_counts: Counter) -> None:
    split = sample["split"]
    images_dir, labels_dir = output_dirs(output, split)
    image_path = Path(sample["image_path"])
    label_path = Path(sample["label_path"])
    suffix = f"__r{copy_index:02d}" if copy_index else ""
    output_stem = f"{prefix}__{image_path.stem}{suffix}"
    output_image = images_dir / f"{output_stem}{image_path.suffix.lower()}"
    output_label = labels_dir / f"{output_stem}.txt"
    shutil.copy2(image_path, output_image)
    if label_path.exists():
        shutil.copy2(label_path, output_label)
    else:
        output_label.write_text("", encoding="utf-8")
    for class_name, value in sample["class_counts"].items():
        counts[(split, class_name)] += int(value)
    source_counts[f"{sample['kind']}/{split}"] += 1


def group_by_split(samples: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[sample["split"]].append(sample)
    return grouped


def select_target_train(samples: list[dict], names: list[str], per_class_goal: int, max_images_per_class: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    candidates_by_class: dict[str, list[dict]] = {name: [] for name in names}
    for sample in samples:
        if sample["split"] != "train":
            continue
        for class_name, value in sample["class_counts"].items():
            if value > 0 and class_name in candidates_by_class:
                candidates_by_class[class_name].append(sample)
    for class_samples in candidates_by_class.values():
        rng.shuffle(class_samples)

    selected: dict[str, dict] = {}
    added_boxes: Counter = Counter()
    cursors = Counter()
    while True:
        progressed = False
        for class_name in names:
            if added_boxes[class_name] >= per_class_goal:
                continue
            candidates = candidates_by_class.get(class_name, [])
            if not candidates:
                continue
            images_for_class = sum(1 for item in selected.values() if item["class_counts"].get(class_name, 0) > 0)
            if images_for_class >= max_images_per_class:
                continue
            start_cursor = cursors[class_name]
            attempts = 0
            while attempts < len(candidates):
                candidate = candidates[cursors[class_name] % len(candidates)]
                cursors[class_name] += 1
                attempts += 1
                if candidate["id"] in selected:
                    continue
                selected[candidate["id"]] = candidate
                for other_class, value in candidate["class_counts"].items():
                    added_boxes[other_class] += int(value)
                progressed = True
                break
            if cursors[class_name] == start_cursor and attempts == 0:
                continue
        if not progressed:
            break
        if all(added_boxes[name] >= per_class_goal or not candidates_by_class.get(name) for name in names):
            break
    return list(selected.values())


def current_split_counts(samples: list[dict], names: list[str], split: str) -> Counter:
    counts: Counter = Counter()
    for sample in samples:
        if sample["split"] != split:
            continue
        for class_name, value in sample["class_counts"].items():
            if class_name in names:
                counts[class_name] += int(value)
    return counts


def reserve_target_train_for_validation(
    base_samples: list[dict],
    target_samples: list[dict],
    names: list[str],
    min_val_boxes: int,
    reserve_images_per_class: int,
    seed: int,
) -> tuple[list[dict], set[str]]:
    """Move train images into val when val coverage is missing, without duplicating them into train."""
    rng = random.Random(seed)
    val_counts = current_split_counts(base_samples + target_samples, names, "val")
    train_candidates_by_class: dict[str, list[dict]] = {name: [] for name in names}
    # Prefer target-domain validation coverage, then fall back to baseline train images.
    for sample in [*target_samples, *base_samples]:
        if sample["split"] != "train":
            continue
        for class_name, value in sample["class_counts"].items():
            if value > 0 and class_name in train_candidates_by_class:
                train_candidates_by_class[class_name].append(sample)
    for candidates in train_candidates_by_class.values():
        rng.shuffle(candidates)

    reserved: dict[str, dict] = {}
    reserved_ids: set[str] = set()
    for class_name in names:
        if val_counts[class_name] >= min_val_boxes:
            continue
        picked_for_class = 0
        for sample in train_candidates_by_class.get(class_name, []):
            if sample["id"] in reserved_ids:
                continue
            copied = dict(sample)
            copied["split"] = "val"
            copied["id"] = f"reserved_val:{sample['id']}"
            reserved[copied["id"]] = copied
            reserved_ids.add(sample["id"])
            for other_class, value in sample["class_counts"].items():
                val_counts[other_class] += int(value)
            picked_for_class += 1
            if val_counts[class_name] >= min_val_boxes or picked_for_class >= reserve_images_per_class:
                break
    return list(reserved.values()), reserved_ids


def oversample_weak_classes(
    train_samples: list[dict],
    names: list[str],
    current_counts: Counter,
    output: Path,
    source_counts: Counter,
    balance_goal_boxes: int,
    max_repeat_per_image: int,
    seed: int,
) -> Counter:
    rng = random.Random(seed)
    repeats_by_sample: Counter = Counter()
    added: Counter = Counter()
    by_class: dict[str, list[dict]] = {name: [] for name in names}
    for sample in train_samples:
        if sample["split"] != "train":
            continue
        for class_name, value in sample["class_counts"].items():
            if value > 0 and class_name in by_class:
                by_class[class_name].append(sample)
    for items in by_class.values():
        rng.shuffle(items)

    progressed = True
    while progressed:
        progressed = False
        for class_name in names:
            if current_counts[("train", class_name)] >= balance_goal_boxes:
                continue
            candidates = by_class.get(class_name, [])
            if not candidates:
                continue
            for sample in candidates:
                if repeats_by_sample[sample["id"]] >= max_repeat_per_image:
                    continue
                repeats_by_sample[sample["id"]] += 1
                copy_sample(
                    sample,
                    output,
                    f"balance_{class_name}_{sample['kind']}",
                    repeats_by_sample[sample["id"]],
                    current_counts,
                    source_counts,
                )
                for other_class, value in sample["class_counts"].items():
                    added[other_class] += int(value)
                progressed = True
                break
    return added


def copy_hard_negatives(samples: list[dict], output: Path, train_limit: int, val_limit: int, seed: int, source_counts: Counter) -> None:
    rng = random.Random(seed)
    by_split = group_by_split(samples)
    for split, limit in {"train": train_limit, "val": val_limit, "test": val_limit}.items():
        split_samples = [sample for sample in by_split.get(split, []) if not sample["class_counts"]]
        rng.shuffle(split_samples)
        for index, sample in enumerate(split_samples[: max(limit, 0)]):
            copy_sample(sample, output, f"hard_negative_{split}", index, Counter(), source_counts)


def class_counts_by_split(output: Path, names: list[str]) -> dict[str, dict[str, int]]:
    counts = {split: {name: 0 for name in names} for split in ("train", "val", "test")}
    empty_labels = Counter()
    for split in ("train", "val", "test"):
        labels_dir = output / "labels" / split
        if not labels_dir.exists():
            continue
        for label_path in labels_dir.glob("*.txt"):
            text = label_path.read_text(encoding="utf-8").strip()
            if not text:
                empty_labels[split] += 1
                continue
            for class_name, value in read_label_counts(label_path, names).items():
                counts[split][class_name] += int(value)
    return {**counts, "_empty_label_images": dict(empty_labels)}


def guard_summary(counts: dict, names: list[str], min_train: int, min_val: int, max_class_ratio: float) -> dict:
    missing_train = [name for name in names if counts["train"].get(name, 0) < min_train]
    missing_val = [name for name in names if counts["val"].get(name, 0) < min_val]
    nonzero_train = [counts["train"].get(name, 0) for name in names if counts["train"].get(name, 0) > 0]
    ratio = (max(nonzero_train) / min(nonzero_train)) if nonzero_train else float("inf")
    return {
        "ok": not missing_train and not missing_val and ratio <= max_class_ratio,
        "missing_or_too_low_train": missing_train,
        "missing_or_too_low_val": missing_val,
        "train_max_min_ratio": ratio,
        "max_allowed_train_ratio": max_class_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a class-balanced five-class YOLO detection dataset.")
    parser.add_argument("--base", default="merged_dataset")
    parser.add_argument("--target", default="domain_adaptation/target_yolo")
    parser.add_argument("--hard-negatives", default="domain_adaptation/hard_negatives")
    parser.add_argument("--output", default="merged_dataset_domain_balanced")
    parser.add_argument(
        "--allow-target-only-base",
        action="store_true",
        help="Allow building from the target-domain dataset alone when the baseline merged dataset is unavailable or rejected.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-box-goal-per-class", type=int, default=4500)
    parser.add_argument("--max-base-images-per-class", type=int, default=2400)
    parser.add_argument("--preserve-all-base-train", action="store_true")
    parser.add_argument("--target-box-goal-per-class", type=int, default=2200)
    parser.add_argument("--max-target-images-per-class", type=int, default=1400)
    parser.add_argument("--balance-goal-boxes", type=int, default=1800)
    parser.add_argument("--max-repeat-per-image", type=int, default=5)
    parser.add_argument("--val-reserve-images-per-class", type=int, default=30)
    parser.add_argument("--max-hard-negative-train", type=int, default=350)
    parser.add_argument("--max-hard-negative-val", type=int, default=80)
    parser.add_argument("--min-train-boxes-per-class", type=int, default=1)
    parser.add_argument("--min-val-boxes-per-class", type=int, default=1)
    parser.add_argument("--max-class-ratio", type=float, default=25.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base = Path(args.base)
    target = Path(args.target)
    hard_negatives = Path(args.hard_negatives)
    output = Path(args.output)
    base_data_yaml = base / "data.yaml"
    target_data_yaml = target / "data.yaml"
    using_target_only_base = False
    if base_data_yaml.exists():
        names = read_names(base_data_yaml)
    elif args.allow_target_only_base and target_data_yaml.exists():
        names = read_names(target_data_yaml)
        using_target_only_base = True
        print(f"Baseline data.yaml is unavailable, using target-only balanced dataset mode: {target}")
    else:
        raise FileNotFoundError(f"Missing base data.yaml: {base_data_yaml}")
    if names != REQUIRED_NAMES:
        raise ValueError(f"Expected names {REQUIRED_NAMES}, got {names}")
    if target_data_yaml.exists():
        target_names = read_names(target_data_yaml)
        if target_names != names:
            raise ValueError(f"Target names must match {names}, got {target_names}")

    base_samples = [] if using_target_only_base else collect_samples(base, "base", names)
    target_samples = collect_samples(target, "target", names)
    negative_samples = collect_samples(hard_negatives, "hard_negative", names)
    params = vars(args)
    fingerprint = sample_fingerprint(base_samples + target_samples + negative_samples, params)
    summary_path = output / "class_balanced_summary.json"
    if summary_path.exists() and not args.force:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("fingerprint") == fingerprint and summary.get("guard", {}).get("ok"):
            print(json.dumps(summary, indent=2))
            return

    clean_dir(output)
    ensure_output(output)
    source_counts: Counter = Counter()
    train_counts: Counter = Counter()

    reserved_val_samples, reserved_train_ids = reserve_target_train_for_validation(
        base_samples,
        target_samples,
        names,
        args.min_val_boxes_per_class,
        args.val_reserve_images_per_class,
        args.seed,
    )

    if args.preserve_all_base_train:
        selected_base_train = [
            sample
            for sample in base_samples
            if sample["split"] == "train" and sample["id"] not in reserved_train_ids
        ]
    else:
        selected_base_train = select_target_train(
            [
                sample
                for sample in base_samples
                if sample["split"] == "train" and sample["id"] not in reserved_train_ids
            ],
            names,
            max(args.base_box_goal_per_class, 0),
            max(args.max_base_images_per_class, 0),
            args.seed,
        )
    selected_base_train_ids = {sample["id"] for sample in selected_base_train}

    for sample in base_samples:
        if sample["split"] == "train" and sample["id"] not in selected_base_train_ids:
            continue
        copy_sample(sample, output, "base", 0, train_counts, source_counts)

    train_eligible_target = [
        sample
        for sample in target_samples
        if sample["split"] == "train" and sample["id"] not in reserved_train_ids
    ]
    selected_target = select_target_train(
        train_eligible_target,
        names,
        max(args.target_box_goal_per_class, 0),
        max(args.max_target_images_per_class, 0),
        args.seed,
    )
    selected_target_ids = {sample["id"] for sample in selected_target}
    for sample in reserved_val_samples:
        copy_sample(sample, output, "reserved_val", 0, train_counts, source_counts)
    for sample in selected_target:
        copy_sample(sample, output, "target", 0, train_counts, source_counts)
    for sample in target_samples:
        if sample["split"] in {"val", "test"}:
            copy_sample(sample, output, "target", 0, train_counts, source_counts)

    copy_hard_negatives(
        negative_samples,
        output,
        args.max_hard_negative_train,
        args.max_hard_negative_val,
        args.seed,
        source_counts,
    )

    train_pool = selected_base_train + selected_target
    oversample_added = oversample_weak_classes(
        train_pool,
        names,
        train_counts,
        output,
        source_counts,
        max(args.balance_goal_boxes, 0),
        max(args.max_repeat_per_image, 0),
        args.seed,
    )

    save_yaml(
        output / "data.yaml",
        {
            "path": str(output.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": names,
        },
    )
    counts = class_counts_by_split(output, names)
    guard = guard_summary(
        counts,
        names,
        args.min_train_boxes_per_class,
        args.min_val_boxes_per_class,
        args.max_class_ratio,
    )
    summary = {
        "base": str(base),
        "target": str(target),
        "hard_negatives": str(hard_negatives),
        "output": str(output),
        "using_target_only_base": using_target_only_base,
        "fingerprint": fingerprint,
        "parameters": params,
        "names": names,
        "source_image_counts": dict(source_counts),
        "base_train_images_available": sum(1 for sample in base_samples if sample["split"] == "train"),
        "base_train_images_selected": len(selected_base_train_ids),
        "target_train_images_available": sum(1 for sample in target_samples if sample["split"] == "train"),
        "train_images_reserved_for_val": len(reserved_train_ids),
        "target_train_images_selected": len(selected_target_ids),
        "oversample_added_boxes": dict(oversample_added),
        "class_counts_by_split": counts,
        "guard": guard,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not guard["ok"]:
        raise SystemExit(
            "Class-balanced dataset guard failed. Fix source coverage before training: "
            + json.dumps(guard, sort_keys=True)
        )


if __name__ == "__main__":
    main()
