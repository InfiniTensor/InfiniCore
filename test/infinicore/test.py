import torch
from infinicore.lib import _infinicore
from infinicore.utils import to_torch_dtype
import numpy as np

import infinicore


def test():
    shape = [2, 3, 4]
    shape2 = [3, 4, 2]
    torch_tensor_ans = torch.rand(shape, dtype=torch.float32, device="cpu")
    torch_tensor_result = torch.zeros(shape, dtype=torch.float32, device="cpu")

    t_cpu = infinicore.from_blob(
        torch_tensor_ans.data_ptr(),
        shape,
        dtype=infinicore.float32,
        device=infinicore.device("cpu", 0),
    )

    t_gpu = t_cpu.to(infinicore.device("cuda", 0))

    t_gpu = t_gpu.permute([1, 2, 0])

    t_gpu2 = infinicore.empty(
        shape2, dtype=infinicore.float32, device=infinicore.device("cuda", 0)
    )

    t_gpu2.copy_(t_gpu)

    t_gpu2 = t_gpu2.permute([2, 0, 1]).contiguous()

    t_result = infinicore.from_blob(
        torch_tensor_result.data_ptr(),
        shape,
        dtype=infinicore.float32,
        device=infinicore.device("cpu", 0),
    )

    t_result.copy_(t_gpu2)

    assert torch.equal(torch_tensor_ans, torch_tensor_result)
    print("Test passed")


def test2():
    "测试infinicore.Tensor的from_torch, +* 运算符功能"
    shape = [1, 2, 3]

    x1_torch = torch.rand(shape, dtype=torch.float32, device="cpu")
    x2_torch = torch.rand(shape, dtype=torch.float32, device="cpu")

    x1_infini = infinicore.from_torch(x1_torch.clone())
    x2_infini = infinicore.from_torch(x2_torch.clone())

    ans1_infini = x1_infini + x2_infini
    ans2_infini = x1_infini * x2_infini

    ans1_torch_ref = x1_torch + x2_torch
    ans2_torch_ref = x1_torch * x2_torch

    print("----------------------------------------")
    torch_ans1_result = torch.zeros(shape, dtype=torch.float32, device="cpu")
    torch_ans2_result = torch.zeros(shape, dtype=torch.float32, device="cpu")
    torch_ans1 = infinicore.from_blob(
        torch_ans1_result.data_ptr(),
        shape,
        dtype=infinicore.float32,
        device=infinicore.device("cpu", 0),
    )
    torch_ans2 = infinicore.from_blob(
        torch_ans2_result.data_ptr(),
        shape,
        dtype=infinicore.float32,
        device=infinicore.device("cpu", 0),
    )
    torch_ans1.copy_(ans1_infini)
    torch_ans2.copy_(ans2_infini)

    print("----------------------------------------")
    print("abs error: ", torch.abs(ans1_torch_ref - torch_ans1_result).max())
    print("abs error: ", torch.abs(ans2_torch_ref - torch_ans2_result).max())


def test3():
    "测试infinicore.Tensor的@运算符功能（矩阵乘法）"
    shape1 = [2, 3]
    shape2 = [3, 4]

    x1_torch = torch.rand(shape1, dtype=torch.float32, device="cpu")
    x2_torch = torch.rand(shape2, dtype=torch.float32, device="cpu")

    x1_infini = infinicore.from_torch(x1_torch.clone())
    x2_infini = infinicore.from_torch(x2_torch.clone())

    ans_infini = x1_infini @ x2_infini
    ans_torch_ref = x1_torch @ x2_torch

    print("----------------------------------------")
    torch_ans_result = torch.zeros([2, 4], dtype=torch.float32, device="cpu")
    torch_ans = infinicore.from_blob(
        torch_ans_result.data_ptr(),
        [2, 4],
        dtype=infinicore.float32,
        device=infinicore.device("cpu", 0),
    )
    torch_ans.copy_(ans_infini)

    print("abs error: ", torch.abs(ans_torch_ref - torch_ans_result).max())


def test4_to():
    """
    解决在python代码中 tensor从gpu to到cpu上出错的问题.
    """
    if True:
        x = torch.rand((2, 3), dtype=torch.float32, device="cpu")
        x_infini = infinicore.from_torch(x.clone())
        print(" ---------------> test 1")
        x_infini.debug()
        x_gpu = x_infini.to(infinicore.device("cuda", 0))
        x_gpu = x_gpu.to(infinicore.device("cuda", 0))

        x_gpu.debug()
        x_cpu = x_infini.to(infinicore.device("cpu", 0))
        x_cpu = x_cpu.to(infinicore.device("cpu", 0))

        x_cpu.debug()

    if True:
        x = infinicore.empty(
            (2, 3), dtype=infinicore.float32, device=infinicore.device("cuda", 0)
        )
        x.debug()
        x.to(infinicore.device("cuda", 0))

        x_cpu = x.to(infinicore.device("cpu", 0))
        x_cpu.debug()

        x_cpu_gpu = x_cpu.to(infinicore.device("cuda", 0))
        x_cpu_gpu.debug()

        x_gpu = x.to(infinicore.device("cuda", 0))
        x_gpu.debug()

    print(" 简单的测试用例，通过!!")


def test5_bf16():
    """
    测试 from_list的bf16的数据类型.
    """
    aa = [1.1, 2.2, 3.3]
    torch_tensor = torch.tensor(aa, dtype=torch.bfloat16)
    print("torch的bf16的数据\n", torch_tensor.dtype, torch_tensor)

    infini_tensor = infinicore.from_list(aa, dtype=infinicore.bfloat16)
    print("\n\ninfini的bf16的数据类型\n", infini_tensor.dtype)

    print("----------------------------------------")
    torch_ans_result = torch.zeros(infini_tensor.shape, dtype=torch.bfloat16)
    torch_ans = infinicore.from_blob(
        torch_ans_result.data_ptr(),
        infini_tensor.shape,
        dtype=infinicore.bfloat16,
        device=infinicore.device("cpu", 0),
    )
    torch_ans.copy_(infini_tensor)

    print("误差:", torch_tensor - torch_ans_result)


def func6_initialize_device_relationship():
    from infinicore.device import _initialize_device_relationship

    all_device_types = [
        _infinicore.Device.Type.CPU,  # 0  "cpu"
        _infinicore.Device.Type.NVIDIA,  # 1  "cuda"
        _infinicore.Device.Type.CAMBRICON,  # 2  "mlu"
        _infinicore.Device.Type.ASCEND,  # 3  "npu"
        _infinicore.Device.Type.METAX,  # 4  "cuda"
        _infinicore.Device.Type.MOORE,  # 5  "musa"
        _infinicore.Device.Type.ILUVATAR,  # 6  "cuda"
        _infinicore.Device.Type.QY,  # 9  "cuda"
        _infinicore.Device.Type.KUNLUN,  # 7  "cuda"
        _infinicore.Device.Type.HYGON,  # 8  "cuda"
    ]
    if True:
        print("\n ---------- 测试 CPU")
        all_device_count = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        infinicore_2_python_dict, python_2_infinicore_dict = (
            _initialize_device_relationship(all_device_types, all_device_count)
        )
        print("infinicore_2_python_dict", infinicore_2_python_dict)
        print("python_2_infinicore_dict: ", python_2_infinicore_dict)

        print("\n ---------- 测试 CPU+NVIDIA")
        all_device_count = [1, 2, 0, 0, 0, 0, 0, 0, 0, 0]
        infinicore_2_python_dict, python_2_infinicore_dict = (
            _initialize_device_relationship(all_device_types, all_device_count)
        )
        print("infinicore_2_python_dict", infinicore_2_python_dict)
        print("python_2_infinicore_dict: ", python_2_infinicore_dict)

        print("\n ---------- 测试 CPU+NVIDIA+HYGON")
        all_device_count = [1, 2, 0, 0, 0, 0, 0, 0, 0, 2]
        infinicore_2_python_dict, python_2_infinicore_dict = (
            _initialize_device_relationship(all_device_types, all_device_count)
        )
        print("infinicore_2_python_dict", infinicore_2_python_dict)
        print("python_2_infinicore_dict: ", python_2_infinicore_dict)

        print("\n ---------- 测试 CPU+NVIDIA+HYGON+CAMBRICON+ASCEND")
        all_device_count = [1, 2, 2, 2, 0, 0, 0, 0, 0, 2]
        infinicore_2_python_dict, python_2_infinicore_dict = (
            _initialize_device_relationship(all_device_types, all_device_count)
        )
        print("infinicore_2_python_dict", infinicore_2_python_dict)
        print("python_2_infinicore_dict: ", python_2_infinicore_dict)

    if True:
        print("\n ---------- 算子测试 cpu")
        x = torch.ones((2, 3), dtype=torch.float32, device="cpu")
        x_infini = infinicore.from_torch(x)

        y = torch.ones((2, 3), dtype=torch.float32, device="cpu")
        y_infini = infinicore.from_torch(y)

        z_infini = x_infini + y_infini
        print(z_infini.device)
        z_infini.debug()

    if True:
        print("\n ---------- 算子测试 cuda")
        x = torch.ones((2, 3), dtype=torch.float32, device="cuda:0")
        x_infini = infinicore.from_torch(x)

        y = torch.ones((2, 3), dtype=torch.float32, device="cuda:0")
        y_infini = infinicore.from_torch(y)

        z_infini = x_infini + y_infini
        # print(z_infini.device)
        # z_infini.debug()

    if False:
        print("\n ---------- 算子测试 cuda", infinicore.device("cuda", 0)._underlying)
        x = infinicore.empty(
            (2, 3), dtype=infinicore.float32, device=infinicore.device("cuda")
        )
        y = infinicore.empty(
            (2, 3), dtype=infinicore.float32, device=infinicore.device("cuda")
        )
        z = x + y
        print(z.device)

    if False:
        print("\n ---------- 算子测试 MOORE")
        x = torch.ones((2, 3), dtype=torch.float32, device="musa:1")
        x_infini = infinicore.from_torch(x)

        y = torch.ones((2, 3), dtype=torch.float32, device="musa:1")
        y_infini = infinicore.from_torch(y)

        z_infini = x_infini + y_infini
        print(z_infini.device)
        z_infini.debug()


def func7_print_different_data_types():
    """Test printing for different data types."""

    # Test cases: (dtype_name, dtype_object, test_data)
    test_cases = [
        ("BOOL", infinicore.bool, [[True, False], [False, True]]),
        ("I8", infinicore.int8, [[-128, -64], [32, 127]]),
        ("I16", infinicore.int16, [[-32768, -16384], [8192, 32767]]),
        (
            "I32",
            infinicore.int32,
            [[-2147483648, -1073741824], [1073741824, 2147483647]],
        ),
        (
            "I64",
            infinicore.int64,
            [
                [-1000000000000000000, -500000000000000000],
                [500000000000000000, 1000000000000000000],
            ],
        ),
        ("U8", infinicore.uint8, [[0, 64], [192, 255]]),
        ("BF16", infinicore.bfloat16, [[1.234, 2.345], [4.567, 5.678]]),
        ("F16", infinicore.float16, [[1.234, 2.345], [4.567, 5.678]]),
        ("F32", infinicore.float32, [[1.234, 2.34], [4.569, 5.9]]),
        ("F64", infinicore.float64, [[1.23456789111, 2.3456789], [4.56789, 5.6789]]),
    ]

    for dtype_name, dtype_obj, test_data in test_cases:
        print(f"\n{'=' * 70}")
        print(f"Testing DataType::{dtype_name}")
        print(f"{'=' * 70}")

        # Create infinicore tensor
        t_infini = infinicore.from_list(
            test_data, dtype=dtype_obj, device=infinicore.device("cpu")
        )
        print("\n[Infinicore] Default print options:")
        print(t_infini)

        # Compare with PyTorch if supported
        torch_dtype = to_torch_dtype(dtype_obj)
        if torch_dtype is not None:
            t_torch = torch.tensor(test_data, dtype=torch_dtype)
            print("\n[PyTorch] Default print options:")
            print(t_torch)
        else:
            print(f"\n[PyTorch] DataType {dtype_name} not supported by PyTorch")


def func8_print_options():
    """Test global print options: precision, threshold, edgeitems, linewidth, sci_mode"""
    print(f"\n{'=' * 70}")
    print("Testing global print options configuration")
    print(f"{'=' * 70}")

    # Create test tensors of different sizes
    test_tensors = {
        "Small (3x3)": infinicore.from_list(
            [[1.211, 2.389, 3.89], [4.569, 5.689, 6.789], [7.89, 8.9, 9.0]],
            dtype=infinicore.float64,
        ),
        "Medium (8x8)": infinicore.from_list(
            np.random.randn(8, 8).tolist(), dtype=infinicore.float32
        ),
        "Large (15x15)": infinicore.from_list(
            np.random.randn(15, 15).tolist(), dtype=infinicore.float32
        ),
    }

    # Test cases: (name, options_dict)
    test_cases = [
        ("Precision: 2", {"precision": 2}),
        ("Precision: -1 (auto)", {"precision": -1}),
        ("Threshold: 50, Edgeitems: 2", {"threshold": 50, "edgeitems": 2}),
        ("Threshold: 200, Edgeitems: 1", {"threshold": 200, "edgeitems": 1}),
        ("Linewidth: 40", {"linewidth": 40}),
        ("Sci_mode: True (scientific)", {"sci_mode": True}),
        ("Sci_mode: False (normal)", {"sci_mode": False}),
        ("Combined: p=1, t=50, e=2", {"precision": 1, "threshold": 50, "edgeitems": 2}),
        (
            "Combined: p=6, t=100, e=1, sci=True",
            {"precision": 6, "threshold": 100, "edgeitems": 1, "sci_mode": True},
        ),
    ]

    for case_name, options in test_cases:
        print(f"\n{'=' * 70}")
        print(f"Test Case: {case_name}")
        print(f"  Options: {options}")
        print(f"{'=' * 70}")

        # Set print options
        infinicore.set_printoptions(**options)

        # Print all test tensors
        for tensor_name, tensor in test_tensors.items():
            print(f"\n[{tensor_name}]:")
            print(tensor)

    # Reset to defaults
    infinicore.set_printoptions(
        precision=-1, threshold=1000, edgeitems=3, linewidth=80, sci_mode=None
    )


def func9_print_temporary_options():
    """Test that temporary print options work correctly and don't affect global settings."""
    print(f"\n{'=' * 70}")
    print("Testing temporary print options (context manager)")
    print(f"{'=' * 70}")
    infinicore.set_printoptions(
        precision=4, threshold=1000, edgeitems=3, linewidth=80, sci_mode=None
    )

    # Create test tensor
    test_data = [[1.211, 2.389, 3.89], [4.569, 5.689, 6.789], [7.89, 8.9, 9.0]]
    t_small = infinicore.from_list(
        test_data, device=infinicore.device("cuda"), dtype=infinicore.float64
    )

    # Verify initial settings
    print("Tensor output:")
    print(t_small)

    # Enter context with temporary settings
    with infinicore.printoptions(
        precision=2, threshold=50, edgeitems=2, linewidth=40, sci_mode=True
    ):
        print("Tensor output (with temporary settings):")
        print(t_small)

    # Verify global settings are restored
    print("Tensor output (should match before context):")
    print(t_small)


if __name__ == "__main__":
    test()
    test2()
    test3()
    test4_to()
    test5_bf16()
    func6_initialize_device_relationship()
    func7_print_different_data_types()
    func8_print_options()
    func9_print_temporary_options()
