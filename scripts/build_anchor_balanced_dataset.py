from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from build_balanced_detection_dataset import (
    REQUIRED_NAMES,
    class_counts_by_split,
    clean_dir,
    collect_samples,
    copy_hard_negatives,
    copy_sample,
    ensure_output,
    group_by_split,
    oversample_weak_classes,
    read_names,
    save_yaml,
)


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCE_DOMAIN_PREFIXES = (
    ("pothole_detection_yolov8_baseline_mirror", "road_pothole_scene"),
    ("pothole_detection_yolov8_tplpk", "road_pothole_scene"),
    ("defects_in_facade_building", "facade_wall_scene"),
    ("facade_defects_detection", "facade_wall_scene"),
    ("facade_building_defect", "facade_wall_scene"),
    ("building_defect_v2", "mixed_building_scene"),
    ("buildingdamage_spalling", "structural_spalling_scene"),
    ("building_damage_insurance_wall_defects", "wall_maintenance_scene"),
    ("building_defect_on_walls", "wall_maintenance_scene"),
    ("wall_surface_crack_mold_peeling_seepage", "wall_maintenance_scene"),
    ("concrete_crack_broad", "closeup_concrete_crack"),
    ("wall_cracks", "wall_crack_scene"),
)


def sample_fingerprint(samples: list[dict], params: dict) -> str:
    digest = hashlib.sha256(json.dumps(params, sort_keys=True).encode("utf-8"))
    for sample in sorted(samples, key=lambda item: item["id"]):
        image_path = Path(sample["image_path"])
        label_path = Path(sample["label_path"])
        digest.update(sample["id"].encode("utf-8"))
        digest.update(str(image_path).encode("utf-8"))
        digest.update(str(image_path.stat().st_size if image_path.exists() else 0).encode("utf-8"))
        digest.update(str(label_path.stat().st_size if label_path.exists() else 0).encode("utf-8"))
    return digest.hexdigest()


def count_boxes(samples: list[dict], names: list[str], split: str | None = None) -> Counter:
    counts: Counter = Counter()
    for sample in samples:
        if split is not None and sample["split"] != split:
            continue
        for class_name, value in sample["class_counts"].items():
            if class_name in names:
                counts[class_name] += int(value)
    return counts


def source_domain(sample: dict) -> str:
    if sample.get("kind") == "base":
        return "baseline_anchor"
    if sample.get("kind") == "hard_negative":
        return "hard_negative"
    image_name = Path(sample["image_path"]).name.lower()
    for prefix, domain in SOURCE_DOMAIN_PREFIXES:
        if image_name.startswith(f"{prefix}_"):
            return domain
    return "target_unknown"


def domain_round_robin(samples: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        grouped[source_domain(sample)].append(sample)
    for items in grouped.values():
        rng.shuffle(items)
    domains = sorted(grouped)
    rng.shuffle(domains)

    ordered: list[dict] = []
    while domains:
        next_domains: list[str] = []
        for domain in domains:
            items = grouped[domain]
            if items:
                ordered.append(items.pop())
            if items:
                next_domains.append(domain)
        domains = next_domains
    return ordered


def domain_image_counts(samples: list[dict]) -> dict[str, int]:
    counts: Counter = Counter()
    for sample in samples:
        counts[f"{sample['kind']}/{sample['split']}/{source_domain(sample)}"] += 1
    return dict(counts)


def domain_box_counts(samples: list[dict], names: list[str]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for sample in samples:
        domain = source_domain(sample)
        for class_name, value in sample["class_counts"].items():
            if class_name in names:
                counts[domain][class_name] += int(value)
    return {
        domain: {name: int(class_counts.get(name, 0)) for name in names}
        for domain, class_counts in sorted(counts.items())
    }


def parse_label_quality(label_path: Path, names: list[str], max_box_area: float) -> tuple[Counter, Counter, int]:
    counts: Counter = Counter()
    errors: Counter = Counter()
    total_lines = 0
    if not label_path.exists():
        return counts, Counter({"missing_label": 1}), total_lines
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        total_lines += 1
        parts = line.split()
        if len(parts) != 5:
            errors["non_box_or_malformed_label"] += 1
            continue
        try:
            class_id = int(float(parts[0]))
            x, y, width, height = [float(item) for item in parts[1:]]
        except ValueError:
            errors["non_numeric_label"] += 1
            continue
        if not 0 <= class_id < len(names):
            errors["class_id_out_of_range"] += 1
            continue
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            errors["box_out_of_range"] += 1
            continue
        if x - width / 2 < -0.03 or x + width / 2 > 1.03 or y - height / 2 < -0.03 or y + height / 2 > 1.03:
            errors["box_extends_too_far"] += 1
            continue
        if width * height > max_box_area:
            errors["box_too_large"] += 1
            continue
        counts[names[class_id]] += 1
    return counts, errors, total_lines


def assess_sample_quality(sample: dict, names: list[str], args: argparse.Namespace) -> tuple[bool, str, Counter, dict]:
    image_path = Path(sample["image_path"])
    label_path = Path(sample["label_path"])
    kind = sample["kind"]
    if not image_path.exists() or image_path.suffix.lower() not in IMAGE_EXTS:
        return False, "missing_or_unsupported_image", Counter(), {}
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return False, "unreadable_image", Counter(), {}
    height, width = image.shape[:2]
    if min(width, height) < args.min_image_side:
        return False, "image_too_small", Counter(), {"width": width, "height": height}

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if kind != "base" and blur_variance < args.min_blur_variance:
        return False, "target_or_negative_too_blurry", Counter(), {
            "width": width,
            "height": height,
            "blur_variance": round(blur_variance, 3),
        }

    label_counts, label_errors, total_label_lines = parse_label_quality(label_path, names, args.max_box_area)
    invalid_count = sum(label_errors.values())
    invalid_fraction = invalid_count / max(total_label_lines, 1)
    if kind == "hard_negative":
        if sum(label_counts.values()) > 0:
            return False, "hard_negative_has_positive_labels", label_counts, {}
        return True, "kept", Counter(), {
            "width": width,
            "height": height,
            "blur_variance": round(blur_variance, 3),
            "boxes": 0,
        }

    if total_label_lines == 0 or sum(label_counts.values()) == 0:
        return False, "positive_image_without_valid_labels", label_counts, dict(label_errors)
    if invalid_fraction > args.max_invalid_label_fraction:
        return False, "too_many_invalid_labels", label_counts, dict(label_errors)
    if sum(label_counts.values()) > args.max_boxes_per_image:
        return False, "too_many_boxes_in_image", label_counts, {"boxes": sum(label_counts.values())}
    if kind == "target" and sum(1 for value in label_counts.values() if value > 0) > args.max_target_classes_per_image:
        return False, "target_image_too_many_classes", label_counts, dict(label_counts)
    return True, "kept", label_counts, {
        "width": width,
        "height": height,
        "blur_variance": round(blur_variance, 3),
        "boxes": sum(label_counts.values()),
        "classes": sum(1 for value in label_counts.values() if value > 0),
    }


def filter_samples_by_quality(samples: list[dict], names: list[str], args: argparse.Namespace) -> tuple[list[dict], dict]:
    kept: list[dict] = []
    rejected_reasons: Counter = Counter()
    rejected_by_kind: Counter = Counter()
    kept_by_kind: Counter = Counter()
    before_counts = count_boxes(samples, names)
    after_counts: Counter = Counter()
    metrics = {
        "min_blur_variance_kept": None,
        "max_boxes_kept": 0,
    }
    for sample in samples:
        ok, reason, clean_counts, sample_metrics = assess_sample_quality(sample, names, args)
        if not ok:
            rejected_reasons[f"{sample['kind']}:{reason}"] += 1
            rejected_by_kind[sample["kind"]] += 1
            continue
        copied = dict(sample)
        copied["class_counts"] = dict(clean_counts)
        kept.append(copied)
        kept_by_kind[sample["kind"]] += 1
        after_counts.update(clean_counts)
        blur = sample_metrics.get("blur_variance")
        if blur is not None:
            metrics["min_blur_variance_kept"] = blur if metrics["min_blur_variance_kept"] is None else min(metrics["min_blur_variance_kept"], blur)
        metrics["max_boxes_kept"] = max(metrics["max_boxes_kept"], int(sample_metrics.get("boxes", 0) or 0))
    raw_total = len(samples)
    kept_total = len(kept)
    summary = {
        "raw_samples": raw_total,
        "kept_samples": kept_total,
        "rejected_samples": raw_total - kept_total,
        "keep_rate": kept_total / raw_total if raw_total else 1.0,
        "kept_by_kind": dict(kept_by_kind),
        "rejected_by_kind": dict(rejected_by_kind),
        "rejected_reasons": dict(rejected_reasons),
        "box_counts_before_filter": dict(before_counts),
        "box_counts_after_filter": dict(after_counts),
        "metrics": metrics,
    }
    return kept, summary


def guard_baseline_anchor(base_samples: list[dict], names: list[str], min_train_boxes: int, min_val_boxes: int) -> dict:
    train_counts = count_boxes(base_samples, names, "train")
    val_counts = count_boxes(base_samples, names, "val")
    return {
        "ok": all(train_counts[name] >= min_train_boxes for name in names)
        and all(val_counts[name] >= min_val_boxes for name in names),
        "train_counts": dict(train_counts),
        "val_counts": dict(val_counts),
        "missing_train": [name for name in names if train_counts[name] < min_train_boxes],
        "missing_val": [name for name in names if val_counts[name] < min_val_boxes],
        "min_train_boxes": min_train_boxes,
        "min_val_boxes": min_val_boxes,
    }


def target_quality_guard(
    target_samples: list[dict],
    names: list[str],
    min_boxes_per_class: int,
    keep_rate: float,
    min_keep_rate: float,
) -> dict:
    total_target = len(target_samples)
    counts = count_boxes(target_samples, names)
    missing = [name for name in names if counts.get(name, 0) < min_boxes_per_class]
    return {
        "ok": total_target > 0 and not missing and keep_rate >= min_keep_rate,
        "target_box_counts": dict(counts),
        "missing_or_low_target_classes": missing,
        "min_quality_target_boxes_per_class": min_boxes_per_class,
        "target_keep_rate": keep_rate,
        "min_target_keep_rate": min_keep_rate,
    }


def class_candidates(samples: list[dict], names: list[str], split: str) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {name: [] for name in names}
    for sample in samples:
        if sample["split"] != split:
            continue
        for class_name, value in sample["class_counts"].items():
            if class_name in buckets and value > 0:
                buckets[class_name].append(sample)
    return buckets


def select_target_validation(
    target_samples: list[dict],
    names: list[str],
    images_per_class: int,
    seed: int,
) -> tuple[list[dict], set[str]]:
    selected: dict[str, dict] = {}
    reserved_train_ids: set[str] = set()

    # Prefer real validation/test target splits, then reserve train images if needed.
    ordered_splits = ["val", "test", "train"]
    for class_name in names:
        picked = 0
        for split in ordered_splits:
            candidates = [
                sample
                for sample in target_samples
                if sample["split"] == split
                and sample["class_counts"].get(class_name, 0) > 0
                and sample["id"] not in selected
            ]
            candidates = domain_round_robin(candidates, seed + len(selected))
            for sample in candidates:
                copied = dict(sample)
                if split == "train":
                    copied["id"] = f"reserved_target_val:{sample['id']}"
                    copied["split"] = "val"
                    reserved_train_ids.add(sample["id"])
                selected[copied["id"]] = copied
                picked += 1
                if picked >= images_per_class:
                    break
            if picked >= images_per_class:
                break
    return list(selected.values()), reserved_train_ids


def select_target_train(
    target_samples: list[dict],
    names: list[str],
    reserved_train_ids: set[str],
    box_goal_per_class: int,
    max_images_per_class: int,
    max_total_images: int,
    seed: int,
) -> list[dict]:
    buckets = class_candidates(target_samples, names, "train")
    for values in buckets.values():
        values[:] = domain_round_robin(values, seed)

    selected: dict[str, dict] = {}
    selected_images_by_class: Counter = Counter()
    selected_boxes_by_class: Counter = Counter()
    cursors: Counter = Counter()

    while len(selected) < max_total_images:
        progressed = False
        for class_name in names:
            if selected_boxes_by_class[class_name] >= box_goal_per_class:
                continue
            if selected_images_by_class[class_name] >= max_images_per_class:
                continue
            candidates = buckets.get(class_name, [])
            if not candidates:
                continue
            attempts = 0
            while attempts < len(candidates):
                sample = candidates[cursors[class_name] % len(candidates)]
                cursors[class_name] += 1
                attempts += 1
                if sample["id"] in selected or sample["id"] in reserved_train_ids:
                    continue
                selected[sample["id"]] = sample
                for other_class, value in sample["class_counts"].items():
                    if other_class in names:
                        selected_boxes_by_class[other_class] += int(value)
                        if int(value) > 0:
                            selected_images_by_class[other_class] += 1
                progressed = True
                break
        if not progressed:
            break
        if all(
            selected_boxes_by_class[name] >= box_goal_per_class or not buckets.get(name)
            for name in names
        ):
            break
    return list(selected.values())


def split_source_counts(samples: list[dict]) -> dict[str, int]:
    counts: Counter = Counter()
    for sample in samples:
        counts[f"{sample['kind']}/{sample['split']}"] += 1
    return dict(counts)


def add_sample_counts(samples: list[dict], counts: Counter) -> None:
    for sample in samples:
        split = sample["split"]
        for class_name, value in sample["class_counts"].items():
            counts[(split, class_name)] += int(value)


def counts_by_split_from_counter(counts: Counter, names: list[str]) -> dict[str, dict[str, int]]:
    return {
        split: {name: int(counts.get((split, name), 0)) for name in names}
        for split in ("train", "val", "test")
    }


def simulate_oversample_weak_classes(
    train_samples: list[dict],
    names: list[str],
    current_counts: Counter,
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
                for other_class, value in sample["class_counts"].items():
                    current_counts[("train", other_class)] += int(value)
                    added[other_class] += int(value)
                progressed = True
                break
    return added


def final_guard(counts: dict, names: list[str], min_train: int, min_val: int, max_ratio: float) -> dict:
    train_nonzero = [counts["train"].get(name, 0) for name in names if counts["train"].get(name, 0) > 0]
    ratio = max(train_nonzero) / min(train_nonzero) if train_nonzero else float("inf")
    return {
        "ok": all(counts["train"].get(name, 0) >= min_train for name in names)
        and all(counts["val"].get(name, 0) >= min_val for name in names)
        and ratio <= max_ratio,
        "missing_or_too_low_train": [
            name for name in names if counts["train"].get(name, 0) < min_train
        ],
        "missing_or_too_low_val": [
            name for name in names if counts["val"].get(name, 0) < min_val
        ],
        "train_max_min_ratio": ratio,
        "max_allowed_train_ratio": max_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a safer anchor-balanced YOLO dataset for revised production retraining.")
    parser.add_argument("--base", default="merged_dataset")
    parser.add_argument("--target", default="domain_adaptation/target_yolo")
    parser.add_argument("--hard-negatives", default="domain_adaptation/hard_negatives")
    parser.add_argument("--output", default="merged_dataset_anchor_balanced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-box-goal-per-class", type=int, default=1500)
    parser.add_argument("--max-target-images-per-class", type=int, default=900)
    parser.add_argument("--max-target-train-images", type=int, default=3600)
    parser.add_argument("--target-val-images-per-class", type=int, default=35)
    parser.add_argument("--oversample-target-boxes", type=int, default=6500)
    parser.add_argument("--max-repeat-per-image", type=int, default=4)
    parser.add_argument("--max-hard-negative-train", type=int, default=450)
    parser.add_argument("--max-hard-negative-val", type=int, default=100)
    parser.add_argument("--min-anchor-train-boxes-per-class", type=int, default=100)
    parser.add_argument("--min-anchor-val-boxes-per-class", type=int, default=10)
    parser.add_argument("--min-final-train-boxes-per-class", type=int, default=1200)
    parser.add_argument("--min-final-val-boxes-per-class", type=int, default=30)
    parser.add_argument("--max-final-class-ratio", type=float, default=6.0)
    parser.add_argument("--min-image-side", type=int, default=96)
    parser.add_argument("--min-blur-variance", type=float, default=4.0)
    parser.add_argument("--max-box-area", type=float, default=0.92)
    parser.add_argument("--max-boxes-per-image", type=int, default=90)
    parser.add_argument("--max-target-classes-per-image", type=int, default=3)
    parser.add_argument("--max-invalid-label-fraction", type=float, default=0.0)
    parser.add_argument("--min-quality-target-boxes-per-class", type=int, default=25)
    parser.add_argument("--min-target-keep-rate", type=float, default=0.75)
    parser.add_argument("--audit-only", action="store_true", help="Run quality and balance checks without copying images.")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    base = Path(args.base)
    target = Path(args.target)
    hard_negatives = Path(args.hard_negatives)
    output = Path(args.output)
    base_data_yaml = base / "data.yaml"
    if not base_data_yaml.exists():
        raise FileNotFoundError(
            f"Missing baseline anchor data.yaml: {base_data_yaml}. "
            "The revised path intentionally refuses target-only fine-tuning."
        )
    names = read_names(base_data_yaml)
    if names != REQUIRED_NAMES:
        raise ValueError(f"Expected baseline class names {REQUIRED_NAMES}, got {names}")

    raw_base_samples = collect_samples(base, "base", names)
    raw_target_samples = collect_samples(target, "target", names) if (target / "data.yaml").exists() else []
    raw_negative_samples = collect_samples(hard_negatives, "hard_negative", names) if hard_negatives.exists() else []

    base_samples, base_quality = filter_samples_by_quality(raw_base_samples, names, args)
    target_samples, target_quality = filter_samples_by_quality(raw_target_samples, names, args)
    negative_samples, negative_quality = filter_samples_by_quality(raw_negative_samples, names, args)

    anchor_guard = guard_baseline_anchor(
        base_samples,
        names,
        args.min_anchor_train_boxes_per_class,
        args.min_anchor_val_boxes_per_class,
    )
    if not anchor_guard["ok"]:
        raise SystemExit(
            "Baseline anchor guard failed. Do not fine-tune from a partial baseline dataset: "
            + json.dumps(anchor_guard, sort_keys=True)
        )

    params = vars(args)
    target_guard = target_quality_guard(
        target_samples,
        names,
        args.min_quality_target_boxes_per_class,
        target_quality["keep_rate"],
        args.min_target_keep_rate,
    )
    if not target_guard["ok"]:
        raise SystemExit(
            "Target-domain quality guard failed before training. "
            "This prevents noisy or under-covered target data from damaging the baseline classes: "
            + json.dumps(target_guard, sort_keys=True)
        )

    target_val, reserved_train_ids = select_target_validation(
        target_samples,
        names,
        args.target_val_images_per_class,
        args.seed,
    )

    target_train = select_target_train(
        target_samples,
        names,
        reserved_train_ids,
        args.target_box_goal_per_class,
        args.max_target_images_per_class,
        args.max_target_train_images,
        args.seed,
    )

    simulated_counts: Counter = Counter()
    add_sample_counts(base_samples, simulated_counts)
    add_sample_counts(target_val, simulated_counts)
    add_sample_counts(target_train, simulated_counts)
    simulated_train_counts_before_oversample = {
        name: simulated_counts.get(("train", name), 0)
        for name in names
    }
    largest_train_class = max(simulated_train_counts_before_oversample.values() or [0])
    dynamic_balance_goal = max(
        args.oversample_target_boxes,
        math.ceil(largest_train_class / max(args.max_final_class_ratio, 1.0)),
        args.min_final_train_boxes_per_class,
    )
    train_pool = [sample for sample in base_samples if sample["split"] == "train"] + target_train
    simulated_oversample_added = simulate_oversample_weak_classes(
        train_pool,
        names,
        simulated_counts,
        dynamic_balance_goal,
        args.max_repeat_per_image,
        args.seed,
    )
    simulated_counts_by_split = counts_by_split_from_counter(simulated_counts, names)
    simulated_guard = final_guard(
        simulated_counts_by_split,
        names,
        args.min_final_train_boxes_per_class,
        args.min_final_val_boxes_per_class,
        args.max_final_class_ratio,
    )

    fingerprint = sample_fingerprint(raw_base_samples + raw_target_samples + raw_negative_samples, params)
    summary_path = output / "anchor_balanced_summary.json"
    if args.audit_only:
        audit_summary = {
            "audit_only": True,
            "base": str(base),
            "target": str(target),
            "hard_negatives": str(hard_negatives),
            "output": str(output),
            "fingerprint": fingerprint,
            "names": names,
            "parameters": params,
            "anchor_guard": anchor_guard,
            "target_quality_guard": target_guard,
            "quality_filter": {
                "base": base_quality,
                "target": target_quality,
                "hard_negative": negative_quality,
            },
            "base_source_counts": split_source_counts(base_samples),
            "target_source_counts": split_source_counts(target_samples),
            "target_domain_image_counts": domain_image_counts(target_samples),
            "target_domain_box_counts": domain_box_counts(target_samples, names),
            "target_val_selected": len(target_val),
            "target_train_selected": len(target_train),
            "target_train_selected_boxes": dict(count_boxes(target_train, names)),
            "target_val_selected_domain_counts": domain_image_counts(target_val),
            "target_train_selected_domain_counts": domain_image_counts(target_train),
            "target_train_selected_domain_box_counts": domain_box_counts(target_train, names),
            "train_counts_before_oversample": simulated_train_counts_before_oversample,
            "dynamic_balance_goal": dynamic_balance_goal,
            "oversample_added_boxes": dict(simulated_oversample_added),
            "class_counts_by_split": simulated_counts_by_split,
            "guard": simulated_guard,
        }
        output.mkdir(parents=True, exist_ok=True)
        audit_path = output / "anchor_balanced_audit_summary.json"
        audit_path.write_text(json.dumps(audit_summary, indent=2), encoding="utf-8")
        print(json.dumps(audit_summary, indent=2))
        if not simulated_guard["ok"]:
            raise SystemExit(
                "Anchor-balanced audit failed. Fix dataset coverage before creating the full dataset: "
                + json.dumps(simulated_guard, sort_keys=True)
            )
        return

    if summary_path.exists() and not args.force:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("fingerprint") == fingerprint and summary.get("guard", {}).get("ok"):
            print(json.dumps(summary, indent=2))
            return

    clean_dir(output)
    ensure_output(output)
    source_counts: Counter = Counter()
    copied_counts: Counter = Counter()

    for sample in base_samples:
        copy_sample(sample, output, "base_anchor", 0, copied_counts, source_counts)

    for sample in target_val:
        copy_sample(sample, output, "target_val", 0, copied_counts, source_counts)

    for sample in target_train:
        copy_sample(sample, output, "target_train", 0, copied_counts, source_counts)

    copy_hard_negatives(
        negative_samples,
        output,
        args.max_hard_negative_train,
        args.max_hard_negative_val,
        args.seed,
        source_counts,
    )

    train_counts_before_oversample = {
        name: copied_counts.get(("train", name), 0)
        for name in names
    }
    oversample_added = oversample_weak_classes(
        train_pool,
        names,
        copied_counts,
        output,
        source_counts,
        dynamic_balance_goal,
        args.max_repeat_per_image,
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
    guard = final_guard(
        counts,
        names,
        args.min_final_train_boxes_per_class,
        args.min_final_val_boxes_per_class,
        args.max_final_class_ratio,
    )

    summary = {
        "base": str(base),
        "target": str(target),
        "hard_negatives": str(hard_negatives),
        "output": str(output),
        "fingerprint": fingerprint,
        "names": names,
        "parameters": params,
        "anchor_guard": anchor_guard,
        "target_quality_guard": target_guard,
        "quality_filter": {
            "base": base_quality,
            "target": target_quality,
            "hard_negative": negative_quality,
        },
        "source_image_counts": dict(source_counts),
        "base_source_counts": split_source_counts(base_samples),
        "target_source_counts": split_source_counts(target_samples),
        "target_domain_image_counts": domain_image_counts(target_samples),
        "target_domain_box_counts": domain_box_counts(target_samples, names),
        "target_val_selected": len(target_val),
        "target_train_selected": len(target_train),
        "target_train_selected_boxes": dict(count_boxes(target_train, names)),
        "target_val_selected_domain_counts": domain_image_counts(target_val),
        "target_train_selected_domain_counts": domain_image_counts(target_train),
        "target_train_selected_domain_box_counts": domain_box_counts(target_train, names),
        "train_counts_before_oversample": train_counts_before_oversample,
        "dynamic_balance_goal": dynamic_balance_goal,
        "oversample_added_boxes": dict(oversample_added),
        "class_counts_by_split": counts,
        "guard": guard,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not guard["ok"]:
        raise SystemExit(
            "Anchor-balanced dataset guard failed. Fix dataset coverage before GPU training: "
            + json.dumps(guard, sort_keys=True)
        )


if __name__ == "__main__":
    main()
