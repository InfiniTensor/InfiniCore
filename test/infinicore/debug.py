#!/usr/bin/env python3
"""
Tensor Debug 功能测试脚本

测试 debug 功能在不同设备和数据类型下的正确性
"""

import torch
import infinicore
import sys
import os
import numpy as np
import time

# Framework path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from framework import (
    TestConfig,
    TestRunner,
    TestCase,
    create_infinicore_tensor,
    get_args,
    get_test_devices,
    to_torch_dtype,
    InfiniDeviceNames,
    torch_device_map,
)

# ==============================================================================
# Test Setup
# ==============================================================================

# Test cases - 定义不同的测试场景
_TEST_CASES = [
    TestCase("basic_print", (2, 3)),  # 基本打印
    TestCase("binary_save", (3, 4)),  # 二进制保存
    TestCase("multidimensional", (2, 2, 3)),  # 多维张量
]

# 非连续内存布局测试用例 (is_contiguous=False)
_NON_CONTIGUOUS_TEST_CASES = [
    TestCase("non_contiguous", (3, 4)),  # 测试 transpose 等导致的非连续内存布局
]

# 大规模性能测试用例 - 一千万个数据
_LARGE_SCALE_TEST_CASES = [
    TestCase("large_scale_binary", (10000000,)),  # 1D: 一千万个元素
]

# Data types - 包含所有需要测试的数据类型
_TENSOR_DTYPES = [
    infinicore.float32,
    infinicore.float16,
    infinicore.bfloat16,
]

# Tolerance map - 用于数值验证时的容差
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-3},
    infinicore.float32: {"atol": 0, "rtol": 1e-5},
    infinicore.bfloat16: {"atol": 0, "rtol": 1e-2},
    infinicore.int32: {"atol": 0, "rtol": 0},
    infinicore.int64: {"atol": 0, "rtol": 0},
}

# ==============================================================================
# Helper Functions
# ==============================================================================


def load_binary_with_torch(filename, dtype, shape):
    """使用 torch.frombuffer 读取二进制文件"""
    torch_dtype = to_torch_dtype(dtype)
    with open(filename, "rb") as f:
        data = f.read()
    return torch.frombuffer(data, dtype=torch_dtype).reshape(shape)


# ==============================================================================
# Test Methods
# ==============================================================================


def test_basic_print(device, test_case, dtype, config):
    """测试基本的 debug 打印功能"""
    test_name, shape = test_case.args

    print(
        f"Testing Basic Print on {InfiniDeviceNames[device]} with "
        f"shape:{shape}, dtype:{dtype}"
    )

    device_str = torch_device_map[device]
    torch_dtype = to_torch_dtype(dtype)

    # 创建测试张量
    torch_tensor = torch.arange(
        1, int(np.prod(shape)) + 1, dtype=torch_dtype, device=device_str
    ).reshape(shape)

    infini_tensor = create_infinicore_tensor(torch_tensor, device_str)

    # 测试 debug 打印（不保存文件）
    infini_tensor.debug()

    print(f"✓ Basic print test passed")


def test_binary_save(device, test_case, dtype, config):
    """测试二进制格式保存"""
    test_name, shape = test_case.args

    print(
        f"Testing Binary Save on {InfiniDeviceNames[device]} with "
        f"shape:{shape}, dtype:{dtype}"
    )

    device_str = torch_device_map[device]
    torch_dtype = to_torch_dtype(dtype)

    # 创建测试张量
    torch_tensor = torch.arange(
        1, int(np.prod(shape)) + 1, dtype=torch_dtype, device=device_str
    ).reshape(shape)

    infini_tensor = create_infinicore_tensor(torch_tensor, device_str)

    # 保存为二进制文件
    bin_file = f"/tmp/debug_test_{device}_{dtype}_binary.bin"
    infini_tensor.debug(bin_file)

    # 验证文件存在
    assert os.path.exists(bin_file), f"Binary file not created: {bin_file}"

    # 验证文件大小
    expected_size = int(np.prod(shape)) * torch_tensor.element_size()
    actual_size = os.path.getsize(bin_file)
    assert (
        actual_size == expected_size
    ), f"Binary file size mismatch: {actual_size} vs {expected_size}"

    # 使用 torch.frombuffer 读取并验证
    loaded_tensor = load_binary_with_torch(bin_file, dtype, shape)

    # 将两个张量都移到 CPU 进行比较
    torch_tensor_cpu = torch_tensor.cpu()
    loaded_tensor_cpu = loaded_tensor.cpu()

    tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-5})
    assert torch.allclose(
        loaded_tensor_cpu,
        torch_tensor_cpu,
        atol=tolerance["atol"],
        rtol=tolerance["rtol"],
    ), f"Binary data mismatch"

    # 清理
    os.remove(bin_file)
    print(f"✓ Binary save test passed")


def test_multidimensional(device, test_case, dtype, config):
    """测试多维张量"""
    test_name, shape = test_case.args

    print(
        f"Testing Multidimensional on {InfiniDeviceNames[device]} with "
        f"shape:{shape}, dtype:{dtype}"
    )

    device_str = torch_device_map[device]
    torch_dtype = to_torch_dtype(dtype)

    # 创建多维张量
    torch_tensor = torch.arange(
        1, int(np.prod(shape)) + 1, dtype=torch_dtype, device=device_str
    ).reshape(shape)

    infini_tensor = create_infinicore_tensor(torch_tensor, device_str)

    # 测试打印
    infini_tensor.debug()

    # 测试保存和读取
    bin_file = f"/tmp/debug_test_multidim_{device}_{dtype}.bin"
    infini_tensor.debug(bin_file)

    assert os.path.exists(bin_file), "Multidimensional binary file not created"

    # 验证
    loaded_tensor = load_binary_with_torch(bin_file, dtype, shape)
    torch_tensor_cpu = torch_tensor.cpu()
    loaded_tensor_cpu = loaded_tensor.cpu()

    tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-5})
    assert torch.allclose(
        loaded_tensor_cpu,
        torch_tensor_cpu,
        atol=tolerance["atol"],
        rtol=tolerance["rtol"],
    ), f"Multidimensional data mismatch"

    # 清理
    os.remove(bin_file)
    print(f"✓ Multidimensional test passed")


def test_non_contiguous_stride(device, test_case, dtype, config):
    """测试非连续内存布局的情况（is_contiguous=False，例如 transpose 后的张量）"""
    test_name, shape = test_case.args

    print(f"\n{'='*70}")
    print(f"Testing Non-Contiguous Memory Layout on {InfiniDeviceNames[device]}")
    print(f"  Shape: {shape}, Dtype: {dtype}")
    print(f"{'='*70}")

    device_str = torch_device_map[device]
    torch_dtype = to_torch_dtype(dtype)

    # 创建连续张量
    print(f"\nStep 1: Creating contiguous tensor...")
    torch_tensor_orig = torch.arange(
        1, int(np.prod(shape)) + 1, dtype=torch_dtype, device=device_str
    ).reshape(shape)
    print(f"  Original shape: {torch_tensor_orig.shape}")
    print(f"  Original stride: {torch_tensor_orig.stride()}")
    print(f"  Is contiguous: {torch_tensor_orig.is_contiguous()}")
    print(f"  Data:\n{torch_tensor_orig}")

    # 进行 transpose 操作，创建非连续张量
    print(f"\nStep 2: Transposing to create non-contiguous tensor...")
    torch_tensor_t = torch_tensor_orig.t()  # transpose
    print(f"  Transposed shape: {torch_tensor_t.shape}")
    print(f"  Transposed stride: {torch_tensor_t.stride()}")
    print(f"  Is contiguous: {torch_tensor_t.is_contiguous()}")
    print(f"  Data:\n{torch_tensor_t}")

    # 创建 InfiniCore 张量（非连续）
    # 注意：from_blob 不支持 strides，所以我们使用 permute 创建非连续张量
    # permute([1, 0]) 相当于 transpose，会创建非连续的内存布局
    infini_tensor_orig = create_infinicore_tensor(torch_tensor_orig, device_str)
    infini_tensor_t = infini_tensor_orig.as_strided(
        list(torch_tensor_t.shape), list(torch_tensor_t.stride())
    )

    print(f"\nStep 3: InfiniCore tensor after permute:")
    print(f"  Shape: {infini_tensor_t.shape}")
    print(f"  Stride: {infini_tensor_t.stride()}")
    print(f"  Is contiguous: {infini_tensor_t.is_contiguous()}")

    # ===== 测试二进制格式 =====
    print(f"\n{'='*70}")
    print(f"Testing Binary Format (.bin) with Non-Contiguous Memory Layout")
    print(f"{'='*70}")
    print(f"Note: Binary format now SUPPORTS non-contiguous memory layout!")
    print(f"      It automatically detects and handles stride correctly.")

    bin_file = f"/tmp/debug_non_contiguous_{device}_{dtype}.bin"
    infini_tensor_t.debug(bin_file)

    # 验证二进制文件
    assert os.path.exists(bin_file), f"Binary file not created: {bin_file}"

    # 检查文件大小
    actual_size = os.path.getsize(bin_file)
    expected_size = int(np.prod(torch_tensor_t.shape)) * torch_tensor_t.element_size()

    print(f"\nFile size check:")
    print(
        f"  Expected: {expected_size} bytes ({int(np.prod(torch_tensor_t.shape))} elements)"
    )
    print(f"  Actual: {actual_size} bytes")

    assert (
        actual_size == expected_size
    ), f"File size mismatch: {actual_size} vs {expected_size}"
    print(f"  ✓ File size is correct")

    # 读取并验证数据
    loaded_tensor = load_binary_with_torch(bin_file, dtype, torch_tensor_t.shape)
    torch_tensor_cpu = torch_tensor_t.cpu()
    loaded_tensor_cpu = loaded_tensor.cpu()

    tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-5})

    print(f"\nData verification:")
    print(f"  Expected (first 2 rows):\n{torch_tensor_cpu[:2]}")
    print(f"  Got (first 2 rows):\n{loaded_tensor_cpu[:2]}")

    assert torch.allclose(
        loaded_tensor_cpu,
        torch_tensor_cpu,
        atol=tolerance["atol"],
        rtol=tolerance["rtol"],
    ), f"Data verification failed: loaded data doesn't match expected"

    print(f"\n✓ Binary format: Data matches perfectly!")
    print(
        f"  Binary format correctly handles non-contiguous memory layout using stride"
    )

    # 清理
    os.remove(bin_file)

    print(f"\n{'='*70}")
    print(f"Non-Contiguous Memory Layout Test Summary:")
    print(f"  ✅ Binary format (.bin): NOW supports non-contiguous memory!")
    print(
        f"  Performance: Contiguous tensors use fast path, non-contiguous use stride-based writing"
    )
    print(f"{'='*70}\n")


def test_large_scale_binary_performance(device, test_case, dtype, config):
    """测试大规模数据二进制保存性能（一千万个数据）"""
    test_name, shape = test_case.args

    num_elements = int(np.prod(shape))
    element_size_bytes = {
        infinicore.float32: 4,
        infinicore.float16: 2,
        infinicore.bfloat16: 2,
        infinicore.int32: 4,
        infinicore.int64: 8,
    }

    total_size_mb = (num_elements * element_size_bytes.get(dtype, 4)) / (1024 * 1024)

    print(f"\n{'='*70}")
    print(f"Performance Test: Large Scale Binary Save")
    print(f"  Device: {InfiniDeviceNames[device]}")
    print(f"  Shape: {shape}")
    print(f"  Elements: {num_elements:,}")
    print(f"  Dtype: {dtype}")
    print(f"  Expected file size: {total_size_mb:.2f} MB")
    print(f"{'='*70}")

    device_str = torch_device_map[device]
    torch_dtype = to_torch_dtype(dtype)

    # 创建大规模张量
    print(f"Creating tensor with {num_elements:,} elements...")
    create_start = time.time()
    torch_tensor = torch.randn(shape, dtype=torch_dtype, device=device_str)
    create_time = time.time() - create_start
    print(f"  Tensor creation time: {create_time:.4f} seconds")

    infini_tensor = create_infinicore_tensor(torch_tensor, device_str)

    # 测试保存性能
    bin_file = f"/tmp/debug_large_scale_{device}_{dtype}.bin"

    print(f"\n{'='*70}")
    print(f"[1/2] Writing Binary File")
    print(f"{'='*70}")
    print(f"File: {bin_file}")
    save_start = time.time()
    infini_tensor.debug(bin_file)
    save_time = time.time() - save_start

    # 验证文件存在
    assert os.path.exists(bin_file), f"Binary file not created: {bin_file}"

    # 获取实际文件大小
    actual_size = os.path.getsize(bin_file)
    actual_size_mb = actual_size / (1024 * 1024)

    # 计算写入吞吐量
    write_throughput_mbps = actual_size_mb / save_time if save_time > 0 else 0

    # 打印写入性能结果
    print(f"\n✓ Write Performance:")
    print(f"  File size: {actual_size_mb:.2f} MB ({actual_size:,} bytes)")
    print(f"  Write time: {save_time:.4f} seconds")
    print(f"  Write throughput: {write_throughput_mbps:.2f} MB/s")
    print(f"  Elements written/sec: {num_elements/save_time:,.0f}")

    # 测试读取性能
    print(f"\n{'='*70}")
    print(f"[2/2] Reading Binary File (for verification)")
    print(f"{'='*70}")
    read_start = time.time()
    loaded_tensor = load_binary_with_torch(bin_file, dtype, shape)
    read_time = time.time() - read_start
    read_throughput_mbps = actual_size_mb / read_time if read_time > 0 else 0

    print(f"\n✓ Read Performance:")
    print(f"  Read time: {read_time:.4f} seconds")
    print(f"  Read throughput: {read_throughput_mbps:.2f} MB/s")
    print(f"  Elements read/sec: {num_elements/read_time:,.0f}")

    # 简单验证前几个元素（不做完整验证以节省时间）
    torch_tensor_cpu = torch_tensor.cpu()
    loaded_tensor_cpu = loaded_tensor.cpu()

    sample_size = min(1000, num_elements)
    tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-5})
    assert torch.allclose(
        loaded_tensor_cpu.flatten()[:sample_size],
        torch_tensor_cpu.flatten()[:sample_size],
        atol=tolerance["atol"],
        rtol=tolerance["rtol"],
    ), f"Data verification failed (sampled first {sample_size} elements)"

    print(f"  Data verification: ✓ (sampled first {sample_size} elements)")

    # 打印性能总结
    print(f"\n{'='*70}")
    print(f"Performance Summary")
    print(f"{'='*70}")
    print(f"  Elements: {num_elements:,}")
    print(f"  File size: {actual_size_mb:.2f} MB")
    print(f"  Write time: {save_time:.4f} sec  →  {write_throughput_mbps:.2f} MB/s")
    print(f"  Read time:  {read_time:.4f} sec  →  {read_throughput_mbps:.2f} MB/s")
    print(
        f"  Speed ratio (Read/Write): {read_throughput_mbps/write_throughput_mbps:.2f}x"
    )
    print(f"{'='*70}")

    # 清理
    os.remove(bin_file)
    print(f"\n✓ Large scale performance test passed\n")


# ==============================================================================
# Main Execution Function
# ==============================================================================


def main():
    args = get_args()

    # 创建测试配置
    config = TestConfig(
        tensor_dtypes=_TENSOR_DTYPES,
        tolerance_map=_TOLERANCE_MAP,
        debug=args.debug,
        bench=False,  # debug 测试不需要性能测试
    )

    # 获取测试设备
    devices = get_test_devices(args)

    print("Starting debug tests...")

    all_passed = True

    # 为每种测试类型运行测试
    test_funcs = [
        ("Basic Print", test_basic_print, [_TEST_CASES[0]]),
        ("Binary Save", test_binary_save, [_TEST_CASES[1]]),
        ("Multidimensional", test_multidimensional, [_TEST_CASES[2]]),
    ]

    for test_name, test_func, test_cases in test_funcs:
        print(f"\n{'='*60}")
        print(f"Testing {test_name}")
        print(f"{'='*60}")

        runner = TestRunner(test_cases, config)
        passed = runner.run_tests(devices, test_func)
        all_passed = all_passed and passed

    # 运行非连续内存布局测试
    print(f"\n{'='*60}")
    print(f"Testing Non-Contiguous Memory Layout (is_contiguous=False)")
    print(f"{'='*60}")

    non_contiguous_runner = TestRunner(_NON_CONTIGUOUS_TEST_CASES, config)
    non_contiguous_passed = non_contiguous_runner.run_tests(
        devices, test_non_contiguous_stride
    )
    all_passed = all_passed and non_contiguous_passed

    # 运行大规模性能测试
    print(f"\n{'='*60}")
    print(f"Testing Large Scale Performance (10M elements)")
    print(f"{'='*60}")

    large_scale_runner = TestRunner(_LARGE_SCALE_TEST_CASES, config)
    large_scale_passed = large_scale_runner.run_tests(
        devices, test_large_scale_binary_performance
    )
    all_passed = all_passed and large_scale_passed

    # 打印总结
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")

    if all_passed:
        print("\033[92m✅ All debug tests passed!\033[0m")
    else:
        print("\033[91m❌ Some tests failed!\033[0m")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
