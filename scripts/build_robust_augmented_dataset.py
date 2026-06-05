from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def save_yaml(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


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


def clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_token(value: str, max_chars: int = 24) -> str:
    token = "".join(char.lower() if char.isalnum() else "_" for char in str(value))
    token = "_".join(part for part in token.split("_") if part)
    return (token or "item")[:max_chars]


def generated_image_name(kind: str, variant: str, index: int, source_path: Path) -> str:
    suffix = source_path.suffix.lower() if source_path.suffix.lower() in IMAGE_EXTS else ".jpg"
    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:12]
    return f"{safe_token(kind, 12)}_{safe_token(variant, 18)}_{index:06d}_{digest}{suffix}"


def copy_dataset(src: Path, dst: Path) -> None:
    clean_dir(dst)
    for split in ("train", "val", "test"):
        for kind in ("images", "labels"):
            src_dir = src / kind / split
            dst_dir = dst / kind / split
            dst_dir.mkdir(parents=True, exist_ok=True)
            if not src_dir.exists():
                continue
            for path in src_dir.iterdir():
                if path.is_file():
                    shutil.copy2(path, dst_dir / path.name)


def parse_label(path: Path, names: list[str]) -> list[dict]:
    labels: list[dict] = []
    if not path.exists():
        return labels
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            class_id = int(float(parts[0]))
            x, y, w, h = [float(item) for item in parts[1:]]
        except ValueError:
            continue
        if 0 <= class_id < len(names) and 0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1:
            labels.append({"class_id": class_id, "class_name": names[class_id], "x": x, "y": y, "w": w, "h": h})
    return labels


def class_counts_for_label(path: Path, names: list[str]) -> Counter:
    counts: Counter = Counter()
    for label in parse_label(path, names):
        counts[label["class_name"]] += 1
    return counts


def dataset_counts(root: Path, names: list[str]) -> dict[str, dict[str, int]]:
    counts = {split: {name: 0 for name in names} for split in ("train", "val", "test")}
    empty = Counter()
    for split in counts:
        labels_dir = root / "labels" / split
        if not labels_dir.exists():
            continue
        for label_path in labels_dir.glob("*.txt"):
            per_file = class_counts_for_label(label_path, names)
            if not per_file:
                empty[split] += 1
            for class_name, value in per_file.items():
                counts[split][class_name] += int(value)
    return {**counts, "_empty_label_images": dict(empty)}


def collect_train_samples(root: Path, names: list[str]) -> list[dict]:
    samples: list[dict] = []
    images_dir = root / "images" / "train"
    labels_dir = root / "labels" / "train"
    if not images_dir.exists():
        return samples
    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        counts = class_counts_for_label(label_path, names)
        samples.append({"image_path": image_path, "label_path": label_path, "class_counts": dict(counts)})
    return samples


def adjust_brightness(image: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def gamma_correct(image: np.ndarray, gamma: float) -> np.ndarray:
    inv = 1.0 / max(gamma, 1e-6)
    table = np.array([(i / 255.0) ** inv * 255 for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)


def add_noise(image: np.ndarray, rng: random.Random, sigma: float) -> np.ndarray:
    noise = rng.normalvariate(0, sigma)
    # Use numpy RNG seeded from Python RNG for deterministic full-frame noise.
    np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
    noisy = image.astype(np.float32) + np_rng.normal(noise, sigma, image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def downsample_restore(image: np.ndarray, factor: float) -> np.ndarray:
    height, width = image.shape[:2]
    small_w = max(8, int(width * factor))
    small_h = max(8, int(height * factor))
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (width, height), interpolation=cv2.INTER_LINEAR)


def clahe_luminance(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    enhanced = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge((enhanced, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


def unsharp_mask(image: np.ndarray, amount: float = 0.9) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), 1.2)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def edge_emphasis(image: np.ndarray) -> np.ndarray:
    enhanced = clahe_luminance(image, 2.4)
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 55, 145)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    # Keep this subtle: the goal is edge robustness, not training on artificial line art.
    return cv2.addWeighted(unsharp_mask(enhanced, 0.55), 0.86, edges_bgr, 0.14, 0)


def grayscale_structure(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(image, 0.35, gray_bgr, 0.65, 0)


def perspective_view(image: np.ndarray, rng: random.Random) -> np.ndarray:
    height, width = image.shape[:2]
    max_dx = max(2, int(width * 0.045))
    max_dy = max(2, int(height * 0.045))
    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    dest = np.float32(
        [
            [rng.randint(0, max_dx), rng.randint(0, max_dy)],
            [width - 1 - rng.randint(0, max_dx), rng.randint(0, max_dy)],
            [width - 1 - rng.randint(0, max_dx), height - 1 - rng.randint(0, max_dy)],
            [rng.randint(0, max_dx), height - 1 - rng.randint(0, max_dy)],
        ]
    )
    matrix = cv2.getPerspectiveTransform(source, dest)
    return cv2.warpPerspective(image, matrix, (width, height), borderMode=cv2.BORDER_REFLECT_101)


def surface_color_shift(image: np.ndarray, rng: random.Random) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + rng.randint(-8, 8)) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] + rng.randint(-28, 24), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] + rng.randint(-18, 18), 0, 255)
    shifted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return cv2.addWeighted(shifted, 0.82, clahe_luminance(image, 1.5), 0.18, 0)


def label_boxes_px(labels: list[dict], width: int, height: int) -> list[tuple[int, int, int, int]]:
    boxes = []
    for label in labels:
        x, y, w, h = label["x"], label["y"], label["w"], label["h"]
        x1 = max(0, int((x - w / 2) * width))
        y1 = max(0, int((y - h / 2) * height))
        x2 = min(width, int((x + w / 2) * width))
        y2 = min(height, int((y + h / 2) * height))
        if x2 > x1 and y2 > y1:
            boxes.append((x1, y1, x2, y2))
    return boxes


def overlap_ratio(rect: tuple[int, int, int, int], box: tuple[int, int, int, int]) -> float:
    rx1, ry1, rx2, ry2 = rect
    bx1, by1, bx2, by2 = box
    ix1, iy1 = max(rx1, bx1), max(ry1, by1)
    ix2, iy2 = min(rx2, bx2), min(ry2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / area


def occlude(image: np.ndarray, labels: list[dict], rng: random.Random, max_box_overlap: float) -> np.ndarray:
    output = image.copy()
    height, width = image.shape[:2]
    boxes = label_boxes_px(labels, width, height)
    count = rng.randint(1, 3)
    for _ in range(count):
        accepted = None
        for _attempt in range(30):
            rect_w = rng.randint(max(8, width // 18), max(10, width // 5))
            rect_h = rng.randint(max(8, height // 18), max(10, height // 5))
            x1 = rng.randint(0, max(0, width - rect_w))
            y1 = rng.randint(0, max(0, height - rect_h))
            rect = (x1, y1, x1 + rect_w, y1 + rect_h)
            if not boxes or max(overlap_ratio(rect, box) for box in boxes) <= max_box_overlap:
                accepted = rect
                break
        if accepted is None:
            continue
        x1, y1, x2, y2 = accepted
        color = rng.choice([(35, 35, 35), (90, 90, 90), (160, 160, 160), (210, 210, 210)])
        cv2.rectangle(output, (x1, y1), (x2, y2), color, -1)
    return output


def apply_variant(image: np.ndarray, labels: list[dict], variant: str, rng: random.Random, max_box_overlap: float) -> np.ndarray:
    if variant == "low_light":
        return gamma_correct(adjust_brightness(image, 0.72, -24), 0.78)
    if variant == "overexposed":
        return gamma_correct(adjust_brightness(image, 1.24, 22), 1.18)
    if variant == "shadow":
        output = adjust_brightness(image, 0.86, -12)
        height, width = output.shape[:2]
        overlay = output.copy()
        pts = np.array(
            [
                [rng.randint(0, width // 2), 0],
                [width, rng.randint(0, height // 3)],
                [width, height],
                [rng.randint(0, width // 3), height],
            ],
            dtype=np.int32,
        )
        cv2.fillPoly(overlay, [pts], (20, 20, 20))
        return cv2.addWeighted(overlay, 0.22, output, 0.78, 0)
    if variant == "blur_distance":
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        return downsample_restore(blurred, 0.42)
    if variant == "noise_compression":
        noisy = add_noise(image, rng, 9.0)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), rng.randint(45, 70)]
        ok, encoded = cv2.imencode(".jpg", noisy, encode_param)
        return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else noisy
    if variant == "occlusion":
        return occlude(image, labels, rng, max_box_overlap)
    if variant == "local_contrast":
        return clahe_luminance(image, 2.6)
    if variant == "edge_emphasis":
        return edge_emphasis(image)
    if variant == "grayscale_structure":
        return grayscale_structure(image)
    if variant == "perspective_view":
        return perspective_view(image, rng)
    if variant == "surface_color_shift":
        return surface_color_shift(image, rng)
    raise ValueError(f"Unknown variant: {variant}")


def label_to_xyxy(label: dict, width: int, height: int) -> tuple[float, float, float, float]:
    x, y, w, h = label["x"], label["y"], label["w"], label["h"]
    return (
        max(0.0, (x - w / 2.0) * width),
        max(0.0, (y - h / 2.0) * height),
        min(float(width), (x + w / 2.0) * width),
        min(float(height), (y + h / 2.0) * height),
    )


def crop_window_for_label(
    label: dict,
    width: int,
    height: int,
    context_scale: float,
    rng: random.Random,
    min_crop_side: int,
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = label_to_xyxy(label, width, height)
    box_w = max(2.0, x2 - x1)
    box_h = max(2.0, y2 - y1)
    crop_w = min(float(width), max(float(min_crop_side), box_w * context_scale))
    crop_h = min(float(height), max(float(min_crop_side), box_h * context_scale))
    # Maintain enough surrounding surface so close-up crops do not become pure texture tiles.
    crop_w = max(crop_w, crop_h * 0.65)
    crop_h = max(crop_h, crop_w * 0.65)
    crop_w = min(float(width), crop_w)
    crop_h = min(float(height), crop_h)

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    jitter_x = rng.uniform(-0.10, 0.10) * crop_w
    jitter_y = rng.uniform(-0.10, 0.10) * crop_h
    left = int(round(cx + jitter_x - crop_w / 2.0))
    top = int(round(cy + jitter_y - crop_h / 2.0))
    right = left + int(round(crop_w))
    bottom = top + int(round(crop_h))

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        left -= right - width
        right = width
    if bottom > height:
        top -= bottom - height
        bottom = height
    left = max(0, left)
    top = max(0, top)
    right = min(width, right)
    bottom = min(height, bottom)
    if right - left < min_crop_side or bottom - top < min_crop_side:
        return None
    return left, top, right, bottom


def transform_labels_to_crop(
    labels: list[dict],
    crop: tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    min_visible_fraction: float,
) -> list[str]:
    left, top, right, bottom = crop
    crop_w = max(1, right - left)
    crop_h = max(1, bottom - top)
    transformed: list[str] = []
    for label in labels:
        x1, y1, x2, y2 = label_to_xyxy(label, image_width, image_height)
        ix1, iy1 = max(x1, left), max(y1, top)
        ix2, iy2 = min(x2, right), min(y2, bottom)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        original_area = max(1.0, (x2 - x1) * (y2 - y1))
        visible_area = (ix2 - ix1) * (iy2 - iy1)
        if visible_area / original_area < min_visible_fraction:
            continue
        nx = ((ix1 + ix2) / 2.0 - left) / crop_w
        ny = ((iy1 + iy2) / 2.0 - top) / crop_h
        nw = (ix2 - ix1) / crop_w
        nh = (iy2 - iy1) / crop_h
        if 0 <= nx <= 1 and 0 <= ny <= 1 and nw > 0 and nh > 0:
            transformed.append(f"{label['class_id']} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")
    return transformed


def collect_scale_space_targets(samples: list[dict], names: list[str], crops_per_class: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {name: [] for name in names}
    for sample in samples:
        labels = parse_label(Path(sample["label_path"]), names)
        for label in labels:
            class_name = names[label["class_id"]]
            buckets[class_name].append({**sample, "target_label": label, "all_labels": labels})
    for items in buckets.values():
        rng.shuffle(items)

    selected: list[dict] = []
    per_image_repeats: Counter = Counter()
    for class_name in names:
        added = 0
        for target in buckets[class_name]:
            if added >= crops_per_class:
                break
            image_key = str(target["image_path"])
            if per_image_repeats[image_key] >= 4:
                continue
            selected.append(target)
            per_image_repeats[image_key] += 1
            added += 1
    rng.shuffle(selected)
    return selected


def create_scale_space_crops(
    output_root: Path,
    samples: list[dict],
    names: list[str],
    args: argparse.Namespace,
    rng: random.Random,
) -> dict:
    if args.scale_space_crops_per_class <= 0:
        return {
            "created": 0,
            "rejected": 0,
            "class_counts": {},
            "context_counts": {},
        }
    contexts = [float(item.strip()) for item in args.scale_space_contexts.split(",") if item.strip()]
    if not contexts:
        raise ValueError("--scale-space-contexts must contain at least one numeric context scale.")

    targets = collect_scale_space_targets(samples, names, args.scale_space_crops_per_class, args.seed + 917)
    out_images = output_root / "images" / "train"
    out_labels = output_root / "labels" / "train"
    created = 0
    rejected = 0
    class_counts: Counter = Counter()
    context_counts: Counter = Counter()

    for index, target in enumerate(targets):
        image_path = Path(target["image_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            rejected += 1
            continue
        height, width = image.shape[:2]
        context = contexts[index % len(contexts)]
        crop = crop_window_for_label(
            target["target_label"],
            width,
            height,
            context,
            rng,
            args.scale_space_min_crop_side,
        )
        if crop is None:
            rejected += 1
            continue
        transformed = transform_labels_to_crop(
            target["all_labels"],
            crop,
            width,
            height,
            args.scale_space_min_visible_fraction,
        )
        if not transformed:
            rejected += 1
            continue
        left, top, right, bottom = crop
        cropped = image[top:bottom, left:right]
        if cropped.size == 0:
            rejected += 1
            continue
        class_name = names[target["target_label"]["class_id"]]
        output_name = generated_image_name("scalespace", f"c{context:.2f}_{class_name}", created, image_path)
        output_image = out_images / output_name
        output_label = out_labels / f"{Path(output_name).stem}.txt"
        cv2.imwrite(str(output_image), cropped)
        output_label.write_text("\n".join(transformed) + "\n", encoding="utf-8")
        created += 1
        context_counts[f"{context:.2f}"] += 1
        for line in transformed:
            class_id = int(line.split()[0])
            class_counts[names[class_id]] += 1

    return {
        "created": created,
        "rejected": rejected,
        "selected_targets": len(targets),
        "class_counts": dict(class_counts),
        "context_counts": dict(context_counts),
    }


def select_augmented_samples(samples: list[dict], names: list[str], per_class_goal: int, max_images_per_class: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {name: [] for name in names}
    for sample in samples:
        for class_name, count in sample["class_counts"].items():
            if count > 0 and class_name in buckets:
                buckets[class_name].append(sample)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected: list[dict] = []
    added_boxes: Counter = Counter()
    selected_images: Counter = Counter()
    repeats: Counter = Counter()
    cursors: Counter = Counter()

    while True:
        progressed = False
        for class_name in names:
            if added_boxes[class_name] >= per_class_goal or selected_images[class_name] >= max_images_per_class:
                continue
            candidates = buckets.get(class_name, [])
            if not candidates:
                continue
            for _ in range(len(candidates)):
                sample = candidates[cursors[class_name] % len(candidates)]
                cursors[class_name] += 1
                key = str(sample["image_path"])
                if repeats[key] >= 2:
                    continue
                repeats[key] += 1
                selected.append(sample)
                for other_class, value in sample["class_counts"].items():
                    if other_class in names:
                        added_boxes[other_class] += int(value)
                        if int(value) > 0:
                            selected_images[other_class] += 1
                progressed = True
                break
        if not progressed:
            break
        if all(added_boxes[name] >= per_class_goal or not buckets.get(name) for name in names):
            break
    return selected


def augment_dataset(input_root: Path, output_root: Path, args: argparse.Namespace) -> dict:
    data_yaml = input_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing data.yaml: {data_yaml}")
    names = read_names(data_yaml)
    copy_dataset(input_root, output_root)

    train_samples = collect_train_samples(output_root, names)
    positive_samples = [sample for sample in train_samples if sum(sample["class_counts"].values()) > 0]
    negative_samples = [sample for sample in train_samples if not sample["class_counts"]]
    selected = select_augmented_samples(
        positive_samples,
        names,
        args.aug_box_goal_per_class,
        args.max_aug_images_per_class,
        args.seed,
    )
    rng = random.Random(args.seed)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    variant_counts: Counter = Counter()
    added_counts: Counter = Counter()
    created = 0
    rejected = 0

    out_images = output_root / "images" / "train"
    out_labels = output_root / "labels" / "train"
    for index, sample in enumerate(selected):
        image_path = Path(sample["image_path"])
        label_path = Path(sample["label_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            rejected += 1
            continue
        labels = parse_label(label_path, names)
        variant = variants[index % len(variants)]
        augmented = apply_variant(image, labels, variant, rng, args.max_occlusion_box_overlap)
        output_name = generated_image_name("robust", variant, created, image_path)
        output_image = out_images / output_name
        output_label = out_labels / f"{Path(output_name).stem}.txt"
        cv2.imwrite(str(output_image), augmented)
        shutil.copy2(label_path, output_label)
        created += 1
        variant_counts[variant] += 1
        added_counts.update(sample["class_counts"])

    # Add photometric variants of hard negatives too, to lower false positives in odd lighting.
    rng.shuffle(negative_samples)
    for index, sample in enumerate(negative_samples[: max(0, args.max_negative_aug_images)]):
        image = cv2.imread(str(sample["image_path"]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        variant = ["low_light", "overexposed", "blur_distance", "noise_compression"][index % 4]
        augmented = apply_variant(image, [], variant, rng, args.max_occlusion_box_overlap)
        output_name = generated_image_name("robust_neg", variant, index, Path(sample["image_path"]))
        output_image = out_images / output_name
        output_label = out_labels / f"{Path(output_name).stem}.txt"
        cv2.imwrite(str(output_image), augmented)
        output_label.write_text("", encoding="utf-8")
        created += 1
        variant_counts[f"negative_{variant}"] += 1

    scale_space_summary = create_scale_space_crops(output_root, positive_samples, names, args, rng)

    save_yaml(
        output_root / "data.yaml",
        {
            "path": str(output_root.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": names,
        },
    )
    counts = dataset_counts(output_root, names)
    train_values = [counts["train"].get(name, 0) for name in names if counts["train"].get(name, 0) > 0]
    ratio = max(train_values) / min(train_values) if train_values else math.inf
    guard = {
        "ok": all(counts["train"].get(name, 0) >= args.min_train_boxes_per_class for name in names)
        and all(counts["val"].get(name, 0) >= args.min_val_boxes_per_class for name in names)
        and ratio <= args.max_train_class_ratio,
        "train_max_min_ratio": ratio,
        "max_allowed_train_ratio": args.max_train_class_ratio,
        "missing_or_low_train": [name for name in names if counts["train"].get(name, 0) < args.min_train_boxes_per_class],
        "missing_or_low_val": [name for name in names if counts["val"].get(name, 0) < args.min_val_boxes_per_class],
    }
    summary = {
        "input": str(input_root),
        "output": str(output_root),
        "names": names,
        "parameters": vars(args),
        "positive_train_samples": len(positive_samples),
        "negative_train_samples": len(negative_samples),
        "selected_positive_samples": len(selected),
        "created_augmented_images": created,
        "rejected_augmented_images": rejected,
        "variant_counts": dict(variant_counts),
        "added_box_counts": dict(added_counts),
        "scale_space": scale_space_summary,
        "class_counts_by_split": counts,
        "guard": guard,
    }
    (output_root / "robust_augmentation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Create robust defect-cue and scale-space variants from an anchor-balanced YOLO dataset.")
    parser.add_argument("--input", default="merged_dataset_anchor_balanced")
    parser.add_argument("--output", default="merged_dataset_anchor_robust")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--aug-box-goal-per-class", type=int, default=1800)
    parser.add_argument("--max-aug-images-per-class", type=int, default=650)
    parser.add_argument("--max-negative-aug-images", type=int, default=240)
    parser.add_argument("--max-occlusion-box-overlap", type=float, default=0.35)
    parser.add_argument(
        "--scale-space-crops-per-class",
        type=int,
        default=320,
        help="Create this many label-preserving ROI crops per class across close/mid/context scales.",
    )
    parser.add_argument(
        "--scale-space-contexts",
        default="1.35,2.20,3.40",
        help="Comma-separated crop context multipliers around each target box.",
    )
    parser.add_argument("--scale-space-min-crop-side", type=int, default=96)
    parser.add_argument("--scale-space-min-visible-fraction", type=float, default=0.55)
    parser.add_argument(
        "--variants",
        default=(
            "low_light,overexposed,shadow,blur_distance,noise_compression,occlusion,"
            "local_contrast,edge_emphasis,grayscale_structure,perspective_view,surface_color_shift"
        ),
    )
    parser.add_argument("--min-train-boxes-per-class", type=int, default=1200)
    parser.add_argument("--min-val-boxes-per-class", type=int, default=30)
    parser.add_argument("--max-train-class-ratio", type=float, default=6.0)
    args = parser.parse_args()

    summary = augment_dataset(Path(args.input), Path(args.output), args)
    print(json.dumps(summary, indent=2))
    if not summary["guard"]["ok"]:
        raise SystemExit("Robust augmentation guard failed: " + json.dumps(summary["guard"], sort_keys=True))


if __name__ == "__main__":
    main()
