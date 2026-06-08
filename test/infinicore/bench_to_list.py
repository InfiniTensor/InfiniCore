#!/usr/bin/env python3
"""Benchmark: tensor.to_numpy().tolist() vs to_list() for int32 / int64.

Compares the native C++ ``to_list`` against the numpy round-trip
(``to_numpy().tolist()``) for converting an infinicore Tensor back to a
Python list.

Usage:
    python test/infinicore/bench_to_list.py
    python test/infinicore/bench_to_list.py --sizes 1000 100000 --repeat 50
"""

from __future__ import annotations

import argparse
import statistics
import time

import infinicore
from infinicore.tensor import from_list, to_list


def _make_1d(n: int) -> list[int]:
    return list(range(n))


def _make_2d(rows: int, cols: int) -> list[list[int]]:
    return [list(range(cols)) for _ in range(rows)]


def _time_call(fn, warmup: int, repeat: int) -> list[float]:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        samples.append(t1 - t0)
    return samples


def _mean_ms(samples: list[float]) -> float:
    return statistics.mean(samples) * 1000.0


def _run_case(name: str, tensor, *, warmup: int, repeat: int, verify: bool) -> dict:
    def run_numpy():
        return tensor.to_numpy().tolist()

    def run_native():
        return to_list(tensor)

    if verify:
        a = run_numpy()
        b = run_native()
        if a != b:
            raise AssertionError(f"{name}: result mismatch between to_numpy().tolist() and to_list()")

    numpy_ms = _mean_ms(_time_call(run_numpy, warmup, repeat))
    native_ms = _mean_ms(_time_call(run_native, warmup, repeat))
    speedup = numpy_ms / native_ms if native_ms > 0 else float("inf")

    return {
        "name": name,
        "numel": tensor.numel(),
        "numpy_ms": numpy_ms,
        "native_ms": native_ms,
        "speedup": speedup,
    }


def _print_header() -> None:
    print(
        f"{'case':<28} {'numel':>10} "
        f"{'to_numpy.tolist(ms)':>20} {'to_list(ms)':>14} {'speedup':>10}"
    )
    print("-" * 86)


def _print_row(r: dict) -> None:
    tag = "faster" if r["speedup"] > 1.0 else "slower"
    print(
        f"{r['name']:<28} {r['numel']:>10d} "
        f"{r['numpy_ms']:>20.3f} {r['native_ms']:>14.3f} {r['speedup']:>9.2f}x {tag}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark to_numpy().tolist() vs to_list() for int32/int64"
    )
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1,4,16,32,64,128, 1000, 10000, 100000],
        help="1D lengths to benchmark",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=30, help="Timed iterations")
    parser.add_argument("--no-verify", action="store_true", help="Skip correctness check")
    args = parser.parse_args()

    verify = not args.no_verify
    dtypes = [("int32", infinicore.int32), ("int64", infinicore.int64)]

    print("=" * 86)
    print("to_numpy().tolist() vs to_list() benchmark (CPU)")
    print(f"warmup={args.warmup}, repeat={args.repeat}, verify={verify}")
    print("=" * 86)

    for dtype_name, dtype in dtypes:
        results = []
        for n in args.sizes:
            t = from_list(_make_1d(n), dtype=dtype)
            results.append(
                _run_case(
                    f"1d_{dtype_name}_{n}", t,
                    warmup=args.warmup, repeat=args.repeat, verify=verify,
                )
            )

        for rows, cols in [(100, 10), (100, 100), (100, 1000)]:
            t = from_list(_make_2d(rows, cols), dtype=dtype)
            results.append(
                _run_case(
                    f"2d_{dtype_name}_{rows}x{cols}", t,
                    warmup=args.warmup, repeat=args.repeat, verify=verify,
                )
            )

        print(f"\n[{dtype_name}]")
        _print_header()
        for r in results:
            _print_row(r)

    print("\n" + "=" * 86)
    print("speedup > 1.0 means to_list() is faster than to_numpy().tolist()")
    print("=" * 86)


if __name__ == "__main__":
    main()
