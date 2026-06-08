#!/usr/bin/env python3
"""
Round-trip test: list -> from_list(dtype) -> to_list() -> compare with input.

Modes:
  quick   Run a fixed suite once (default).
  stress  Loop int32/int64 hotspot cases for a fixed duration (default 5 min).

Usage:
    PYTHONPATH=python LD_LIBRARY_PATH=$HOME/.infini/lib \\
        python test/infinicore/test_from_list_to_list_roundtrip.py

    # 5-minute int32/int64 stress (you start this manually)
    PYTHONPATH=python LD_LIBRARY_PATH=$HOME/.infini/lib \\
        python test/infinicore/test_from_list_to_list_roundtrip.py --stress

    # custom duration (seconds)
    python test/infinicore/test_from_list_to_list_roundtrip.py --stress --duration 600
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

import infinicore
from infinicore.dtype import dtype as dtype_cls
from infinicore.lib import _infinicore
from infinicore.tensor import from_list, to_list

# uint dtypes not exported at module top-level.
UINT16 = dtype_cls(_infinicore.DataType.U16)
UINT32 = dtype_cls(_infinicore.DataType.U32)
UINT64 = dtype_cls(_infinicore.DataType.U64)

_INT_DTYPES = frozenset({infinicore.int32, infinicore.int64})
_FLOAT_DTYPES = frozenset(
    {
        infinicore.float32,
        infinicore.float64,
        infinicore.float16,
        infinicore.bfloat16,
    }
)
_HALF_FLOAT_DTYPES = frozenset({infinicore.float16, infinicore.bfloat16})

DEFAULT_STRESS_DURATION_SEC = 300  # 5 minutes


@dataclass(frozen=True)
class Case:
    name: str
    data: Any
    dtype: Any


def _is_nested(data: Any) -> bool:
    return isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(
        data[0], (list, tuple)
    )


def _numel(data: Any) -> int:
    if not data:
        return 0
    if _is_nested(data):
        return len(data) * len(data[0])
    return len(data)


def _normalize_scalar(value: Any, dtype) -> Any:
    if dtype == infinicore.bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value != 0
        raise TypeError(f"unsupported scalar for bool dtype: {type(value)!r}")

    if dtype in _FLOAT_DTYPES:
        return float(value)

    if dtype == UINT64:
        return int(value)

    if dtype in (infinicore.uint8, UINT16, UINT32):
        return int(value)

    if dtype in (
        infinicore.int8,
        infinicore.int16,
        infinicore.int32,
        infinicore.int64,
    ):
        return int(value)

    raise ValueError(f"unsupported dtype for round-trip test: {dtype}")


def _normalize_list(data: Any, dtype) -> Any:
    if _is_nested(data):
        return [_normalize_list(row, dtype) for row in data]
    return [_normalize_scalar(x, dtype) for x in data]


def _compare_scalar(expected: Any, actual: Any, dtype, path: str) -> None:
    if dtype == infinicore.bool:
        if expected is not actual and expected != actual:
            raise AssertionError(f"{path}: bool mismatch {expected!r} != {actual!r}")
        return

    if dtype in _HALF_FLOAT_DTYPES:
        if not isinstance(actual, (int, float)):
            raise AssertionError(f"{path}: expected numeric, got {type(actual)!r}")
        if not math.isclose(float(expected), float(actual), rel_tol=1e-2, abs_tol=1e-2):
            raise AssertionError(
                f"{path}: half-float mismatch {expected!r} != {actual!r}"
            )
        return

    if dtype in (infinicore.float32, infinicore.float64):
        if not isinstance(actual, (int, float)):
            raise AssertionError(f"{path}: expected numeric, got {type(actual)!r}")
        if isinstance(expected, float) or isinstance(actual, float):
            if not math.isclose(float(expected), float(actual), rel_tol=1e-6, abs_tol=1e-6):
                raise AssertionError(f"{path}: float mismatch {expected!r} != {actual!r}")
        else:
            if expected != actual:
                raise AssertionError(f"{path}: int mismatch {expected!r} != {actual!r}")
        return

    if expected != actual:
        raise AssertionError(f"{path}: value mismatch {expected!r} != {actual!r}")


def _compare_lists(expected: Any, actual: Any, dtype, path: str = "root") -> None:
    # Fast path for integer hotspot dtypes.
    if dtype in _INT_DTYPES and actual == expected:
        return

    if _is_nested(expected):
        if not _is_nested(actual):
            raise AssertionError(f"{path}: expected nested list, got {actual!r}")
        if len(expected) != len(actual):
            raise AssertionError(
                f"{path}: row count mismatch {len(expected)} != {len(actual)}"
            )
        for i, (erow, arow) in enumerate(zip(expected, actual)):
            if len(erow) != len(arow):
                raise AssertionError(
                    f"{path}[{i}]: col count mismatch {len(erow)} != {len(arow)}"
                )
            for j, (ev, av) in enumerate(zip(erow, arow)):
                _compare_scalar(ev, av, dtype, f"{path}[{i}][{j}]")
        return

    if _is_nested(actual):
        raise AssertionError(f"{path}: expected flat list, got nested {actual!r}")
    if len(expected) != len(actual):
        raise AssertionError(f"{path}: length mismatch {len(expected)} != {len(actual)}")
    for i, (ev, av) in enumerate(zip(expected, actual)):
        _compare_scalar(ev, av, dtype, f"{path}[{i}]")


def roundtrip_once(case: Case, *, verbose: bool = False) -> None:
    tensor = from_list(case.data, dtype=case.dtype)
    actual = to_list(tensor)
    expected = _normalize_list(case.data, case.dtype)

    if verbose:
        print(f"  case:     {case.name}")
        print(f"  numel:    {_numel(case.data)}")
        print(f"  shape:    {list(tensor.shape)}, dtype={tensor.dtype}")
        if _numel(case.data) <= 32:
            print(f"  input:    {case.data!r}")
            print(f"  expected: {expected!r}")
            print(f"  actual:   {actual!r}")

    _compare_lists(expected, actual, case.dtype)


# ---------------------------------------------------------------------------
# Data generators (integer hotspot)
# ---------------------------------------------------------------------------

def _make_1d_range(n: int) -> list[int]:
    return list(range(n))


def _make_1d_mod(n: int, mod: int = 1000) -> list[int]:
    return [i % mod for i in range(n)]


def _make_1d_alternating(n: int) -> list[int]:
    return [(-1) ** i * (i % 97) for i in range(n)]


def _make_1d_int32_edge() -> list[int]:
    return [
        0,
        1,
        -1,
        2**31 - 1,
        -(2**31),
        2**30,
        -(2**30),
        123456789,
        -987654321,
    ]


def _make_1d_int64_edge() -> list[int]:
    return [
        0,
        1,
        -1,
        2**63 - 1,
        -(2**63),
        2**62,
        -(2**62),
        2**40,
        -(2**40),
    ]


def _make_2d_range(rows: int, cols: int) -> list[list[int]]:
    return [list(range(c * cols, (c + 1) * cols)) for c in range(rows)]


def _make_2d_ij(rows: int, cols: int) -> list[list[int]]:
    return [[r * cols + c for c in range(cols)] for r in range(rows)]


def _make_2d_mod(rows: int, cols: int, mod: int = 512) -> list[list[int]]:
    return [[(r * cols + c) % mod for c in range(cols)] for r in range(rows)]


def _add_int_hotspot_cases(
    cases: list[Case],
    *,
    prefix: str,
    dtype,
    sizes_1d: list[int],
    shapes_2d: list[tuple[int, int]],
) -> None:
    tag = "i32" if dtype == infinicore.int32 else "i64"

    for n in sizes_1d:
        cases.append(Case(f"{prefix}_1d_range_{n}_{tag}", _make_1d_range(n), dtype))
        cases.append(Case(f"{prefix}_1d_mod_{n}_{tag}", _make_1d_mod(n), dtype))
        cases.append(
            Case(f"{prefix}_1d_alt_{n}_{tag}", _make_1d_alternating(n), dtype)
        )

    if dtype == infinicore.int32:
        cases.append(Case(f"{prefix}_1d_edge_{tag}", _make_1d_int32_edge(), dtype))
    else:
        cases.append(Case(f"{prefix}_1d_edge_{tag}", _make_1d_int64_edge(), dtype))

    for rows, cols in shapes_2d:
        cases.append(
            Case(
                f"{prefix}_2d_range_{rows}x{cols}_{tag}",
                _make_2d_range(rows, cols),
                dtype,
            )
        )
        cases.append(
            Case(
                f"{prefix}_2d_ij_{rows}x{cols}_{tag}",
                _make_2d_ij(rows, cols),
                dtype,
            )
        )
        cases.append(
            Case(
                f"{prefix}_2d_mod_{rows}x{cols}_{tag}",
                _make_2d_mod(rows, cols),
                dtype,
            )
        )

    # tuple variants (same values, different container)
    cases.append(
        Case(f"{prefix}_1d_tuple_256_{tag}", tuple(range(256)), dtype)
    )
    cases.append(
        Case(
            f"{prefix}_2d_tuple_8x8_{tag}",
            tuple(tuple(range(c * 8, (c + 1) * 8)) for c in range(8)),
            dtype,
        )
    )
    cases.append(
        Case(
            f"{prefix}_2d_list_of_tuples_4x16_{tag}",
            [tuple(range(c * 16, (c + 1) * 16)) for c in range(4)],
            dtype,
        )
    )


def _build_quick_cases() -> list[Case]:
    cases: list[Case] = []

    sizes_1d = [1, 4, 16, 64, 256, 1000, 10000]
    shapes_2d = [(2, 3), (10, 10), (64, 64), (100, 10), (100, 100)]

    _add_int_hotspot_cases(
        cases,
        prefix="quick",
        dtype=infinicore.int32,
        sizes_1d=sizes_1d,
        shapes_2d=shapes_2d,
    )
    _add_int_hotspot_cases(
        cases,
        prefix="quick",
        dtype=infinicore.int64,
        sizes_1d=sizes_1d,
        shapes_2d=shapes_2d,
    )

    # Keep a small non-hotspot smoke set.
    cases.extend(
        [
            Case("quick_1d_bool", [True, False, True], infinicore.bool),
            Case("quick_1d_float_f64", [0.0, 1.5, -2.25], infinicore.float64),
            Case("quick_2d_float_f32", [[1.0, 2.0], [3.0, 4.0]], infinicore.float32),
        ]
    )
    return cases


def _build_stress_cases() -> list[Case]:
    cases: list[Case] = []

    sizes_1d = [
        1,
        4,
        16,
        32,
        64,
        128,
        256,
        512,
        1000,
        4096,
        10000,
        50000,
        100000,
    ]
    shapes_2d = [
        (1, 1),
        (4, 4),
        (16, 16),
        (32, 32),
        (64, 64),
        (100, 10),
        (100, 100),
        (100, 1000),
        (256, 256),
        (1000, 100),
    ]

    _add_int_hotspot_cases(
        cases,
        prefix="stress",
        dtype=infinicore.int32,
        sizes_1d=sizes_1d,
        shapes_2d=shapes_2d,
    )
    _add_int_hotspot_cases(
        cases,
        prefix="stress",
        dtype=infinicore.int64,
        sizes_1d=sizes_1d,
        shapes_2d=shapes_2d,
    )
    return cases


def run_quick(cases: list[Case], *, verbose: bool) -> int:
    failed = 0
    print("=" * 72)
    print(f"QUICK round-trip: list -> from_list -> to_list ({len(cases)} cases)")
    print("=" * 72)

    for case in cases:
        try:
            roundtrip_once(case, verbose=verbose)
            print(f"✓ {case.name}  (numel={_numel(case.data)})")
        except Exception as exc:
            failed += 1
            print(f"✗ {case.name}: {exc}")

    print("=" * 72)
    if failed:
        print(f"FAILED: {failed}/{len(cases)} cases")
        return 1
    print(f"PASSED: all {len(cases)} cases")
    return 0


def run_stress(
    cases: list[Case],
    *,
    duration_sec: float,
    verbose: bool,
    report_interval_sec: float = 30.0,
) -> int:
    if not cases:
        print("No stress cases configured.")
        return 1

    deadline = time.perf_counter() + duration_sec
    started = time.perf_counter()
    next_report = started + report_interval_sec

    total_iters = 0
    total_numel = 0
    failed = 0
    case_index = 0
    last_case_name = ""

    print("=" * 72)
    print("STRESS round-trip: int32/int64 hotspot")
    print(f"duration={duration_sec:.0f}s ({duration_sec / 60:.1f} min), cases={len(cases)}")
    print("Press Ctrl+C to stop early.")
    print("=" * 72)

    try:
        while time.perf_counter() < deadline:
            case = cases[case_index % len(cases)]
            case_index += 1
            last_case_name = case.name
            numel = _numel(case.data)

            try:
                roundtrip_once(case, verbose=verbose)
                total_iters += 1
                total_numel += numel
            except Exception as exc:
                failed += 1
                print(f"✗ [{total_iters + failed}] {case.name}: {exc}")

            now = time.perf_counter()
            if now >= next_report:
                elapsed = now - started
                remaining = max(0.0, deadline - now)
                elems_per_sec = total_numel / elapsed if elapsed > 0 else 0.0
                print(
                    f"[{elapsed:6.0f}s] iters={total_iters} failed={failed} "
                    f"elems={total_numel:,} ({elems_per_sec:,.0f} elem/s) "
                    f"last={last_case_name!r} remaining={remaining:.0f}s"
                )
                next_report = now + report_interval_sec

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    elapsed = time.perf_counter() - started
    elems_per_sec = total_numel / elapsed if elapsed > 0 else 0.0

    print("=" * 72)
    print("STRESS summary")
    print(f"  elapsed:      {elapsed:.1f}s")
    print(f"  iterations:   {total_iters}")
    print(f"  failures:     {failed}")
    print(f"  elements:     {total_numel:,}")
    print(f"  throughput:   {elems_per_sec:,.0f} elem/s (round-trip verified)")
    print(f"  last case:    {last_case_name}")
    print("=" * 72)

    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Round-trip test: list -> from_list -> to_list"
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Run int32/int64 hotspot stress loop (default 5 minutes)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_STRESS_DURATION_SEC,
        help=f"Stress duration in seconds (default: {DEFAULT_STRESS_DURATION_SEC})",
    )
    parser.add_argument(
        "--report-interval",
        type=float,
        default=30.0,
        help="Stress progress report interval in seconds (default: 30)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print per-iteration details (stress mode: very noisy)",
    )
    args = parser.parse_args()

    if args.stress:
        cases = _build_stress_cases()
        return run_stress(
            cases,
            duration_sec=args.duration,
            verbose=args.verbose,
            report_interval_sec=args.report_interval,
        )

    cases = _build_quick_cases()
    return run_quick(cases, verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
