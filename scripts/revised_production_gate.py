from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def read_rows(path: Path, threshold: str) -> list[dict]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if str(row.get("threshold", "")).rstrip("0").rstrip(".") == threshold.rstrip("0").rstrip(".")
            and row.get("match") != "skipped"
        ]
    return rows


def row_status(row: dict) -> str:
    explicit = row.get("status", "")
    if explicit:
        return explicit
    expected = row.get("expected_class", "")
    top_class = row.get("top_class", "")
    if expected == "none":
        return "false_positive" if top_class else "true_negative"
    if row.get("match") == "true":
        return "match"
    if not top_class:
        return "no_detection"
    return f"wrong_{top_class}"


def is_ok(row: dict) -> bool:
    return row_status(row) in {"match", "true_negative"}


def expected_match_counter(rows: list[dict]) -> Counter:
    counts: Counter = Counter()
    for row in rows:
        expected = row["expected_class"]
        if row_status(row) == "match":
            for item in expected.split("|"):
                item = item.strip()
                if item:
                    counts[item] += 1
    return counts


def bucket_stats(counter: Counter) -> dict:
    total = int(counter.get("total", 0))
    ok = int(counter.get("ok", 0))
    return {
        **{key: int(value) for key, value in counter.items()},
        "pass_rate": ok / total if total else 0.0,
    }


def bucket_failures(buckets: dict[str, dict], min_rate: float, min_samples: int) -> dict[str, dict]:
    failures = {}
    for name, stats in buckets.items():
        total = int(stats.get("total", 0))
        pass_rate = float(stats.get("pass_rate", 0.0))
        if total >= min_samples and pass_rate < min_rate:
            failures[name] = {
                "total": total,
                "ok": int(stats.get("ok", 0)),
                "pass_rate": round(pass_rate, 4),
                "required_pass_rate": min_rate,
            }
    return failures


def summarize(rows: list[dict]) -> dict:
    statuses = Counter(row_status(row) for row in rows)
    by_expected: dict[str, Counter] = defaultdict(Counter)
    by_domain: dict[str, Counter] = defaultdict(Counter)
    confusion: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        status = row_status(row)
        expected = row["expected_class"]
        domain = row.get("domain_group") or "unknown_domain"
        pred = row.get("top_class") or "none"
        by_expected[expected]["total"] += 1
        by_expected[expected][status] += 1
        by_domain[domain]["total"] += 1
        by_domain[domain][status] += 1
        if is_ok(row):
            by_expected[expected]["ok"] += 1
            by_domain[domain]["ok"] += 1
        confusion[expected][pred] += 1
    ok_count = sum(1 for row in rows if is_ok(row))
    wrong_class = sum(value for key, value in statuses.items() if key.startswith("wrong_"))
    no_detection = statuses["no_detection"]
    false_positive = statuses["false_positive"]
    # Penalize wrong confident class assignments more than misses because they are worse in deployment triage.
    score = ok_count - (0.85 * wrong_class) - (0.55 * no_detection) - (1.25 * false_positive)
    return {
        "rows": len(rows),
        "ok": ok_count,
        "pass_rate": ok_count / len(rows) if rows else 0.0,
        "wrong_class": wrong_class,
        "no_detection": no_detection,
        "false_positive": false_positive,
        "statuses": dict(statuses),
        "by_expected": {key: bucket_stats(value) for key, value in by_expected.items()},
        "by_domain": {key: bucket_stats(value) for key, value in by_domain.items()},
        "confusion": {key: dict(value) for key, value in confusion.items()},
        "expected_class_matches": dict(expected_match_counter(rows)),
        "score": round(score, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare current and candidate production sweeps and decide promotion.")
    parser.add_argument("--current-summary", required=True)
    parser.add_argument("--candidate-summary", required=True)
    parser.add_argument("--output", default="output/revised_production_gate.json")
    parser.add_argument("--threshold", default="0.2")
    parser.add_argument("--min-evaluation-rows", type=int, default=30)
    parser.add_argument("--min-candidate-pass-rate", type=float, default=0.75)
    parser.add_argument("--min-domain-pass-rate", type=float, default=0.70)
    parser.add_argument("--min-expected-pass-rate", type=float, default=0.70)
    parser.add_argument("--min-bucket-samples", type=int, default=3)
    parser.add_argument("--min-pass-rate-delta", type=float, default=0.03)
    parser.add_argument("--allowed-class-regression", type=int, default=0)
    parser.add_argument("--must-not-regress", default="crack,spalling,corrosion,pothole,paint_degradation")
    parser.add_argument("--max-false-positive-increase", type=int, default=0)
    args = parser.parse_args()

    current_rows = read_rows(Path(args.current_summary), args.threshold)
    candidate_rows = read_rows(Path(args.candidate_summary), args.threshold)
    current = summarize(current_rows)
    candidate = summarize(candidate_rows)
    must_not_regress = [item.strip() for item in args.must_not_regress.split(",") if item.strip()]

    class_regressions = {}
    for class_name in must_not_regress:
        before = current["expected_class_matches"].get(class_name, 0)
        after = candidate["expected_class_matches"].get(class_name, 0)
        if after + args.allowed_class_regression < before:
            class_regressions[class_name] = {"current": before, "candidate": after}

    domain_failures = bucket_failures(
        candidate["by_domain"],
        args.min_domain_pass_rate,
        args.min_bucket_samples,
    )
    expected_failures = bucket_failures(
        candidate["by_expected"],
        args.min_expected_pass_rate,
        args.min_bucket_samples,
    )

    checks = {
        "same_or_more_rows": candidate["rows"] >= current["rows"] > 0,
        "enough_evaluation_rows": candidate["rows"] >= args.min_evaluation_rows,
        "absolute_candidate_pass_rate": candidate["pass_rate"] >= args.min_candidate_pass_rate,
        "domain_pass_rates_ok": not domain_failures,
        "expected_group_pass_rates_ok": not expected_failures,
        "pass_rate_improved": candidate["pass_rate"] >= current["pass_rate"] + args.min_pass_rate_delta,
        "score_improved": candidate["score"] > current["score"],
        "false_positive_controlled": candidate["false_positive"]
        <= current["false_positive"] + args.max_false_positive_increase,
        "no_key_class_regression": not class_regressions,
    }
    promoted = all(checks.values())
    decision = {
        "promote": promoted,
        "reason": "candidate_passed_revised_gate" if promoted else "candidate_failed_revised_gate",
        "threshold": args.threshold,
        "checks": checks,
        "class_regressions": class_regressions,
        "domain_failures": domain_failures,
        "expected_group_failures": expected_failures,
        "acceptance_thresholds": {
            "min_evaluation_rows": args.min_evaluation_rows,
            "min_candidate_pass_rate": args.min_candidate_pass_rate,
            "min_domain_pass_rate": args.min_domain_pass_rate,
            "min_expected_pass_rate": args.min_expected_pass_rate,
            "min_bucket_samples": args.min_bucket_samples,
        },
        "current": current,
        "candidate": candidate,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    if not promoted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
