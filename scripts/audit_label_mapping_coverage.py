from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from auto_collect_domain_sources import LABEL_SYNONYMS, TARGET_NAMES, map_source_label
from merge_datasets import build_synonym_map, load_yaml, normalize_label


SOURCE_SPLITS = ("train", "valid", "val", "test")


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


def find_data_yaml(root: Path) -> Path | None:
    if (root / "data.yaml").exists():
        return root / "data.yaml"
    matches = sorted(root.glob("**/data.yaml"))
    return matches[0] if matches else None


def label_dirs(root: Path, split: str) -> list[Path]:
    candidates = [root / split / "labels", root / "labels" / split]
    return [path for path in candidates if path.exists()]


def count_used_class_ids(dataset_root: Path) -> tuple[Counter, Counter]:
    used: Counter = Counter()
    errors: Counter = Counter()
    for split in SOURCE_SPLITS:
        for labels_dir in label_dirs(dataset_root, split):
            for label_path in labels_dir.glob("*.txt"):
                for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
                    line = raw_line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        errors[f"{label_path}:{line_number}:malformed"] += 1
                        continue
                    try:
                        class_id = int(float(parts[0]))
                    except ValueError:
                        errors[f"{label_path}:{line_number}:bad_class_id"] += 1
                        continue
                    used[class_id] += 1
    return used, errors


def map_audit_label(
    source_name: str,
    mapping: dict[str, str],
    allowed_targets: list[str],
    source_key: str = "",
    use_source_overrides: bool = False,
) -> str | None:
    if use_source_overrides:
        target, _ = map_source_label(source_name, source_key)
        if target in allowed_targets:
            return target
    normalized = normalize_label(source_name)
    target = mapping.get(normalized)
    if target in allowed_targets:
        return target
    without_numeric_prefix = re.sub(r"^\d+\s+", "", normalized).strip()
    if without_numeric_prefix and without_numeric_prefix != normalized:
        target = mapping.get(without_numeric_prefix)
        if target in allowed_targets:
            return target
    return None


def audit_source_dataset(
    dataset_root: Path,
    mapping: dict[str, str],
    allowed_targets: list[str],
    source_key: str = "",
    use_source_overrides: bool = False,
) -> dict:
    data_yaml = find_data_yaml(dataset_root)
    if not data_yaml:
        return {"path": str(dataset_root), "exists": dataset_root.exists(), "status": "missing_data_yaml"}
    data = load_yaml(data_yaml)
    names = read_names(data)
    used_ids, label_errors = count_used_class_ids(data_yaml.parent)
    mapped: dict[str, dict] = {}
    unmapped: dict[str, int] = {}
    out_of_range: dict[str, int] = {}
    for class_id, count in sorted(used_ids.items()):
        if class_id < 0 or class_id >= len(names):
            out_of_range[str(class_id)] = int(count)
            continue
        source_name = names[class_id]
        target = map_audit_label(source_name, mapping, allowed_targets, source_key or dataset_root.name, use_source_overrides)
        if target in allowed_targets:
            mapped[source_name] = {"target": target, "labels": int(count)}
        else:
            unmapped[source_name] = int(count)
    return {
        "path": str(data_yaml.parent),
        "data_yaml": str(data_yaml),
        "exists": True,
        "status": "audited",
        "class_names": names,
        "used_label_count": int(sum(used_ids.values())),
        "mapped_used_classes": mapped,
        "unmapped_used_classes": unmapped,
        "out_of_range_class_ids": out_of_range,
        "label_errors": dict(label_errors),
    }


def audit_canonical_dataset(dataset_root: Path, expected_names: list[str]) -> dict:
    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        return {"path": str(dataset_root), "exists": False, "status": "missing"}
    names = read_names(load_yaml(data_yaml))
    used_ids, label_errors = count_used_class_ids(dataset_root)
    out_of_range = {str(class_id): int(count) for class_id, count in used_ids.items() if class_id < 0 or class_id >= len(expected_names)}
    return {
        "path": str(dataset_root),
        "exists": True,
        "status": "audited",
        "class_names": names,
        "expected_names": expected_names,
        "class_order_ok": names == expected_names,
        "used_label_count": int(sum(used_ids.values())),
        "out_of_range_class_ids": out_of_range,
        "label_errors": dict(label_errors),
    }


def iter_target_raw_sources(raw_root: Path) -> list[Path]:
    if not raw_root.exists():
        return []
    roots = []
    for data_yaml in sorted(raw_root.glob("**/data.yaml")):
        roots.append(data_yaml.parent)
    return roots


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit that all used source YOLO labels map into the project class taxonomy.")
    parser.add_argument("--merge-config", default="configs/merge_config.yaml")
    parser.add_argument("--target-raw", default="domain_adaptation/auto_raw")
    parser.add_argument("--target-yolo", default="domain_adaptation/target_yolo")
    parser.add_argument("--canonical-dataset", action="append", default=[])
    parser.add_argument("--output", default="output/label_mapping_coverage_summary.json")
    parser.add_argument("--fail-on-missing-config-source", action="store_true")
    args = parser.parse_args()

    config = load_yaml(Path(args.merge_config))
    merge_map = build_synonym_map(config)
    baseline_sources = []
    target_sources = []
    errors: list[str] = []

    for dataset in config.get("datasets", []):
        root = Path(dataset["path"])
        if not root.is_absolute():
            root = Path.cwd() / root
        audit = audit_source_dataset(root, merge_map, config.get("classes", TARGET_NAMES))
        audit["name"] = dataset.get("name", root.name)
        baseline_sources.append(audit)
        if audit["status"] == "missing_data_yaml":
            if args.fail_on_missing_config_source:
                errors.append(f"Configured source missing data.yaml: {audit['name']} at {root}")
            continue
        if audit.get("unmapped_used_classes"):
            errors.append(f"Baseline source {audit['name']} has unmapped used classes: {audit['unmapped_used_classes']}")
        if audit.get("out_of_range_class_ids"):
            errors.append(f"Baseline source {audit['name']} has out-of-range class ids: {audit['out_of_range_class_ids']}")
        if audit.get("label_errors"):
            errors.append(f"Baseline source {audit['name']} has malformed labels: {list(audit['label_errors'])[:5]}")

    for root in iter_target_raw_sources(Path(args.target_raw)):
        audit = audit_source_dataset(root, LABEL_SYNONYMS, TARGET_NAMES, source_key=root.name, use_source_overrides=True)
        target_sources.append(audit)
        if audit.get("unmapped_used_classes"):
            errors.append(f"Target raw source {root} has unmapped used classes: {audit['unmapped_used_classes']}")
        if audit.get("out_of_range_class_ids"):
            errors.append(f"Target raw source {root} has out-of-range class ids: {audit['out_of_range_class_ids']}")
        if audit.get("label_errors"):
            errors.append(f"Target raw source {root} has malformed labels: {list(audit['label_errors'])[:5]}")

    canonical = {}
    for dataset_value in [args.target_yolo, *args.canonical_dataset]:
        root = Path(dataset_value)
        audit = audit_canonical_dataset(root, TARGET_NAMES)
        canonical[str(root)] = audit
        if audit.get("exists"):
            if not audit.get("class_order_ok"):
                errors.append(f"Canonical dataset {root} class order changed: {audit.get('class_names')}")
            if audit.get("out_of_range_class_ids"):
                errors.append(f"Canonical dataset {root} has out-of-range class ids: {audit['out_of_range_class_ids']}")
            if audit.get("label_errors"):
                errors.append(f"Canonical dataset {root} has malformed labels: {list(audit['label_errors'])[:5]}")

    summary = {
        "ok": not errors,
        "errors": errors,
        "expected_names": TARGET_NAMES,
        "baseline_sources": baseline_sources,
        "target_raw_sources": target_sources,
        "canonical_datasets": canonical,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if errors:
        raise SystemExit("Label mapping coverage audit failed. Add mappings instead of discarding source labels.")


if __name__ == "__main__":
    main()
