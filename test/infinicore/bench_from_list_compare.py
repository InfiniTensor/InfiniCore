#!/usr/bin/env python3
"""Compare saved from_list benchmark results (V0 vs V1 native/ms)."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_native_ms(path: Path) -> dict[str, float]:
    results: dict[str, float] = {}
    pattern = re.compile(
        r"^(?P<name>\S+)\s+(?P<numel>\d+)\s+(?P<native>\d+\.\d+)\s+(?P<numpy>\d+\.\d+)"
    )
    for line in path.read_text().splitlines():
        match = pattern.match(line.strip())
        if match:
            results[match.group("name")] = float(match.group("native"))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare V0/V1 from_list benchmark logs")
    parser.add_argument(
        "--baseline",
        "--v0",
        type=Path,
        default=Path(__file__).with_name("bench_from_list_v1.txt"),
        dest="baseline",
        help="Baseline benchmark log (e.g. V1)",
    )
    parser.add_argument(
        "--candidate",
        "--v1",
        type=Path,
        default=Path(__file__).with_name("bench_from_list_v2.txt"),
        dest="candidate",
        help="Candidate benchmark log (e.g. V2)",
    )
    parser.add_argument(
        "--baseline-label",
        default="baseline",
        help="Label for baseline column",
    )
    parser.add_argument(
        "--candidate-label",
        default="candidate",
        help="Label for candidate column",
    )
    args = parser.parse_args()

    baseline = parse_native_ms(args.baseline)
    candidate = parse_native_ms(args.candidate)
    keys = sorted(set(baseline) & set(candidate))

    print(
        f"{'case':<28} {args.baseline_label + '(ms)':>12} "
        f"{args.candidate_label + '(ms)':>12} {'speedup':>10} {'saved':>8}"
    )
    print("-" * 74)
    speedups = []
    for key in keys:
        t0 = baseline[key]
        t1 = candidate[key]
        speedup = t0 / t1 if t1 > 0 else float("inf")
        saved = (1.0 - t1 / t0) * 100 if t0 > 0 else 0.0
        speedups.append(speedup)
        print(f"{key:<28} {t0:>12.3f} {t1:>12.3f} {speedup:>9.2f}x {saved:>7.1f}%")

    if speedups:
        avg_speedup = sum(speedups) / len(speedups)
        print("-" * 70)
        print("-" * 74)
        print(f"{'average':<28} {'':>12} {'':>12} {avg_speedup:>9.2f}x")


if __name__ == "__main__":
    main()
