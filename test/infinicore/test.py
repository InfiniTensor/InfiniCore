import torch
from infinicore.lib import _infinicore

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
    from infinicore._device import _initialize_device_relationship

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


if __name__ == "__main__":
    test()
    test2()
    test3()
    test4_to()
    test5_bf16()
    func6_initialize_device_relationship()
