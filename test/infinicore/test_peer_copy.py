#!/usr/bin/env python3
"""Focused validation for cross-device InfiniCore tensor copies."""

import argparse
import itertools
import subprocess
import sys
import time

import torch

import infinicore


CPU = infinicore.device("cpu", 0)
DTYPES = (torch.float32, torch.float16, torch.bfloat16, torch.int64)


def make_data(shape, dtype, offset=0):
    count = 1
    for dim in shape:
        count *= dim
    values = (torch.arange(count, dtype=torch.int64) + offset) % 251 - 125
    return values.reshape(shape).to(dtype)


def to_torch_cpu(tensor, torch_dtype):
    cpu_tensor = tensor.to(CPU)
    result = torch.empty(tuple(tensor.shape), dtype=torch_dtype)
    infinicore.from_torch(result).copy_(cpu_tensor)
    return result


def assert_exact(actual, expected, label):
    if not torch.equal(actual, expected):
        mismatch = (actual != expected).reshape(-1).nonzero()
        first = int(mismatch[0]) if mismatch.numel() else -1
        raise AssertionError(
            f"{label}: data mismatch at flat index {first}; "
            f"actual={actual.reshape(-1)[first] if first >= 0 else 'n/a'} "
            f"expected={expected.reshape(-1)[first] if first >= 0 else 'n/a'}"
        )


def check_pair(src_id, dst_id, shape, dtype):
    src_device = infinicore.device("cuda", src_id)
    dst_device = infinicore.device("cuda", dst_id)
    expected = make_data(shape, dtype, offset=src_id * 17 + dst_id)
    src = infinicore.from_torch(expected).to(src_device)

    moved = src.to(dst_device)
    assert moved.device == dst_device
    if moved.numel() > 0:
        assert moved.data_ptr() != src.data_ptr()
    assert_exact(
        to_torch_cpu(moved, dtype),
        expected,
        f"to cuda:{src_id}->cuda:{dst_id} {shape} {dtype}",
    )

    copied = infinicore.empty(shape, dtype=src.dtype, device=dst_device)
    copied.copy_(src)
    assert_exact(
        to_torch_cpu(copied, dtype),
        expected,
        f"copy_ cuda:{src_id}->cuda:{dst_id} {shape} {dtype}",
    )

    same = src.to(src_device)
    assert same.data_ptr() == src.data_ptr()
    assert_exact(to_torch_cpu(src, dtype), expected, "source was modified")


def check_noncontiguous(src_id, dst_id):
    src_device = infinicore.device("cuda", src_id)
    dst_device = infinicore.device("cuda", dst_id)
    expected_base = make_data((7, 5), torch.float32)
    expected = expected_base.permute(1, 0)
    src_base = infinicore.from_torch(expected_base).to(src_device)
    src = src_base.permute([1, 0])
    assert not src.is_contiguous()

    moved = src.to(dst_device)
    assert moved.is_contiguous()
    assert_exact(to_torch_cpu(moved, torch.float32), expected, "noncontiguous source")

    dst_base = infinicore.empty(
        [7, 5], dtype=src.dtype, device=dst_device
    )
    dst = dst_base.permute([1, 0])
    assert not dst.is_contiguous()
    dst.copy_(src)
    assert_exact(
        to_torch_cpu(dst_base, torch.float32),
        expected_base,
        "noncontiguous destination",
    )


def expect_runtime_error(action, text):
    try:
        action()
    except RuntimeError as exc:
        if text not in str(exc):
            raise AssertionError(f"expected error containing {text!r}, got: {exc}") from exc
    else:
        raise AssertionError(f"expected RuntimeError containing {text!r}")


def check_negative_paths():
    src = infinicore.from_torch(torch.arange(8, dtype=torch.float32)).to(
        infinicore.device("cuda", 0)
    )
    bad_dtype = infinicore.empty(
        [8], dtype=infinicore.float16, device=infinicore.device("cuda", 1)
    )
    expect_runtime_error(lambda: bad_dtype.copy_(src), "different dtypes")

    bad_shape = infinicore.empty(
        [4, 2], dtype=src.dtype, device=infinicore.device("cuda", 1)
    )
    expect_runtime_error(lambda: bad_shape.copy_(src), "different shape")


def check_allocator_reuse(src_id, dst_id, iterations):
    src_device = infinicore.device("cuda", src_id)
    dst_device = infinicore.device("cuda", dst_id)
    shape = (257, 17)
    for iteration in range(iterations):
        expected = make_data(shape, torch.float32, offset=iteration)
        moved = infinicore.from_torch(expected).to(src_device).to(dst_device)

        clobber_data = make_data(shape, torch.float32, offset=iteration + 1000)
        clobber = infinicore.from_torch(clobber_data).to(src_device)
        assert clobber.device == src_device

        assert_exact(
            to_torch_cpu(moved, torch.float32),
            expected,
            f"allocator reuse iteration {iteration}",
        )


def worker(src_id, dst_id, iterations):
    src_device = infinicore.device("cuda", src_id)
    dst_device = infinicore.device("cuda", dst_id)
    expected = make_data((1024, 256), torch.float32, offset=src_id)
    src = infinicore.from_torch(expected).to(src_device)
    result = None
    for _ in range(iterations):
        result = src.to(dst_device)
    assert result is not None
    assert_exact(to_torch_cpu(result, torch.float32), expected, "concurrent worker")


def run_concurrent_ring(device_count, iterations):
    processes = []
    for src_id in range(device_count):
        dst_id = (src_id + 1) % device_count
        processes.append(
            subprocess.Popen(
                [
                    sys.executable,
                    __file__,
                    "--worker",
                    str(src_id),
                    str(dst_id),
                    "--iterations",
                    str(iterations),
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        )
    failures = []
    for process in processes:
        output, _ = process.communicate(timeout=300)
        if process.returncode != 0:
            failures.append((process.returncode, output))
    if failures:
        raise AssertionError(f"concurrent ring failed: {failures}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-pairs", action="store_true")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--concurrent", action="store_true")
    parser.add_argument("--worker", nargs=2, type=int, metavar=("SRC", "DST"))
    parser.add_argument("--large", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device_count = infinicore.get_device_count("cuda")
    if device_count < 2:
        raise RuntimeError(f"peer-copy test needs at least 2 GPUs, found {device_count}")

    if args.worker:
        worker(args.worker[0], args.worker[1], args.iterations)
        print(f"worker cuda:{args.worker[0]}->cuda:{args.worker[1]} PASS")
        return

    start = time.perf_counter()
    pairs = (
        list(itertools.permutations(range(device_count), 2))
        if args.all_pairs
        else [(0, 1), (1, 0), (0, device_count - 1), (device_count - 1, 0)]
    )
    shapes = ((0,), (1,), (257,), (3, 5, 7), (100, 3584))
    cases = 0
    for src_id, dst_id in pairs:
        for dtype, shape in itertools.product(DTYPES, shapes):
            check_pair(src_id, dst_id, shape, dtype)
            cases += 1

    check_noncontiguous(0, 1)
    check_noncontiguous(device_count - 1, 0)
    check_negative_paths()
    check_allocator_reuse(0, 1, args.iterations)
    check_allocator_reuse(device_count - 1, 0, args.iterations)

    if args.large:
        large_shape = (16 * 1024 * 1024,)
        for src_id, dst_id in ((0, 1), (1, 0), (0, device_count - 1), (device_count - 1, 0)):
            check_pair(src_id, dst_id, large_shape, torch.float32)

    if args.concurrent:
        run_concurrent_ring(device_count, args.iterations)

    elapsed = time.perf_counter() - start
    print(
        f"PASS: {cases} pair/dtype/shape cases, "
        f"reuse={2 * args.iterations}, concurrent={args.concurrent}, "
        f"elapsed={elapsed:.2f}s"
    )


if __name__ == "__main__":
    main()
