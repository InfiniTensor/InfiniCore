#!/usr/bin/env python3
"""
Benchmark: from_list (native C++) vs from_list_by_numpy (np.array + from_numpy).

Usage:
    python test/infinicore/bench_from_list.py
    python test/infinicore/bench_from_list.py --sizes 1000 10000 --warmup 3 --repeat 20
"""

from __future__ import annotations

import argparse
import ctypes
import statistics
import time

import numpy as np

import infinicore
from infinicore.tensor import from_list, from_list_by_numpy
from infinicore.utils import infinicore_to_numpy_dtype


def _make_1d_int(n: int) -> list[int]:
    return list(range(n))


def _make_1d_float(n: int) -> list[float]:
    return [float(i) * 0.5 for i in range(n)]


def _make_2d_int(rows: int, cols: int) -> list[list[int]]:
    return [list(range(cols)) for _ in range(rows)]


def _make_2d_float(rows: int, cols: int) -> list[list[float]]:
    return [[float(c) * 0.25 for c in range(cols)] for _ in range(rows)]


def _tensor_to_numpy(tensor: infinicore.Tensor) -> np.ndarray:
    np_dtype = infinicore_to_numpy_dtype(tensor.dtype)
    arr = np.empty(tensor.shape, dtype=np_dtype)
    cpu = infinicore.device("cpu", 0)
    buf = infinicore.from_blob(
        arr.ctypes.data_as(ctypes.c_void_p).value,
        list(tensor.shape),
        dtype=tensor.dtype,
        device=cpu,
    )
    src = tensor if tensor.device.type == "cpu" else tensor.to(cpu)
    buf.copy_(src.contiguous())
    return arr


def _assert_close(a: infinicore.Tensor, b: infinicore.Tensor, label: str) -> None:
    if list(a.shape) != list(b.shape):
        raise AssertionError(f"{label}: shape mismatch {a.shape} vs {b.shape}")
    if a.dtype != b.dtype:
        raise AssertionError(f"{label}: dtype mismatch {a.dtype} vs {b.dtype}")

    na = _tensor_to_numpy(a)
    nb = _tensor_to_numpy(b)
    if not np.array_equal(na, nb):
        if np.issubdtype(na.dtype, np.floating):
            if not np.allclose(na, nb, rtol=1e-5, atol=1e-5, equal_nan=True):
                raise AssertionError(f"{label}: floating data mismatch")
        else:
            raise AssertionError(f"{label}: data mismatch")


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


def _summarize(samples: list[float]) -> dict[str, float]:
    return {
        "min_ms": min(samples) * 1000,
        "mean_ms": statistics.mean(samples) * 1000,
        "median_ms": statistics.median(samples) * 1000,
        "max_ms": max(samples) * 1000,
        "stdev_ms": statistics.stdev(samples) * 1000 if len(samples) > 1 else 0.0,
    }


def _run_case(
    name: str,
    data,
    *,
    dtype=None,
    warmup: int,
    repeat: int,
    verify: bool,
) -> dict:
    def run_native():
        return from_list(data, dtype=dtype)

    def run_numpy():
        return from_list_by_numpy(data, dtype=dtype)

    if verify:
        t_native = run_native()
        t_numpy = run_numpy()
        _assert_close(t_native, t_numpy, name)

    native_samples = _time_call(run_native, warmup, repeat)
    numpy_samples = _time_call(run_numpy, warmup, repeat)

    native = _summarize(native_samples)
    numpy = _summarize(numpy_samples)
    speedup = numpy["mean_ms"] / native["mean_ms"] if native["mean_ms"] > 0 else float("inf")

    return {
        "name": name,
        "numel": _numel(data),
        "native": native,
        "numpy": numpy,
        "speedup_native_vs_numpy": speedup,
    }


def _numel(data) -> int:
    if not data:
        return 0
    if isinstance(data[0], (list, tuple)):
        return len(data) * len(data[0])
    return len(data)


def _print_header() -> None:
    print(
        f"{'case':<28} {'numel':>10} "
        f"{'native(ms)':>12} {'numpy(ms)':>12} {'native/numpy':>12}"
    )
    print("-" * 78)


def _print_row(result: dict) -> None:
    native_mean = result["native"]["mean_ms"]
    numpy_mean = result["numpy"]["mean_ms"]
    ratio = result["speedup_native_vs_numpy"]
    tag = "faster" if ratio > 1.0 else "slower"
    print(
        f"{result['name']:<28} {result['numel']:>10d} "
        f"{native_mean:>12.3f} {numpy_mean:>12.3f} {ratio:>11.2f}x {tag}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark from_list vs from_list_by_numpy")
    parser.add_argument(
        "--sizes",
        type=int,
        nargs="+",
        default=[1,4,8,45,72,100, 1000, 10000, 100000],
        help="1D lengths to benchmark",
    )
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=30, help="Timed iterations")
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip correctness check between both paths",
    )
    args = parser.parse_args()

    verify = not args.no_verify
    results = []

    print("=" * 78)
    print("from_list vs from_list_by_numpy benchmark (CPU)")
    print(f"warmup={args.warmup}, repeat={args.repeat}, verify={verify}")
    print("=" * 78)

    print("\n[1D int, dtype=int64]")
    _print_header()
    for n in args.sizes:
        data = _make_1d_int(n)
        r = _run_case(
            f"1d_int_{n}",
            data,
            dtype=infinicore.int64,
            warmup=args.warmup,
            repeat=args.repeat,
            verify=verify,
        )
        results.append(r)
        _print_row(r)

    print("\n[1D float, dtype=float64]")
    _print_header()
    for n in args.sizes:
        data = _make_1d_float(n)
        r = _run_case(
            f"1d_float_{n}",
            data,
            dtype=infinicore.float64,
            warmup=args.warmup,
            repeat=args.repeat,
            verify=verify,
        )
        results.append(r)
        _print_row(r)

    print("\n[1D int, dtype=float32]")
    _print_header()
    for n in args.sizes:
        data = _make_1d_int(n)
        r = _run_case(
            f"1d_int_f32_{n}",
            data,
            dtype=infinicore.float32,
            warmup=args.warmup,
            repeat=args.repeat,
            verify=verify,
        )
        results.append(r)
        _print_row(r)

    print("\n[2D int 100 x cols, dtype=int64]")
    _print_header()
    for cols in [10, 100, 1000]:
        rows = 1000
        data = _make_2d_int(rows, cols)
        r = _run_case(
            f"2d_int_{rows}x{cols}",
            data,
            dtype=infinicore.int64,
            warmup=args.warmup,
            repeat=args.repeat,
            verify=verify,
        )
        results.append(r)
        _print_row(r)

    print("\n[2D float 64 x 64, dtype=float64]")
    _print_header()
    data = _make_2d_float(64, 64)
    r = _run_case(
        "2d_float_64x64",
        data,
        dtype=infinicore.float64,
        warmup=args.warmup,
        repeat=args.repeat,
        verify=verify,
    )
    results.append(r)
    _print_row(r)

    native_wins = sum(1 for r in results if r["speedup_native_vs_numpy"] > 1.0)
    print("\n" + "=" * 78)
    print(f"Summary: from_list faster in {native_wins}/{len(results)} cases (by mean time)")
    print("ratio > 1.0 means from_list is faster than from_list_by_numpy")
    print("=" * 78)


if __name__ == "__main__":
    main()
