import infinicore
import torch
import numpy as np


def test_matmul_basic():
    """测试基本的矩阵乘法"""
    print("Testing basic matmul...")

    # 创建测试数据
    a_shape = [2, 3]
    b_shape = [3, 4]
    result_shape = [2, 4]

    # 创建PyTorch张量作为参考
    torch_a = torch.rand(a_shape, dtype=torch.float32, device="cpu")
    torch_b = torch.rand(b_shape, dtype=torch.float32, device="cpu")
    torch_result = torch.matmul(torch_a, torch_b)

    # 创建infinicore张量
    infini_a = infinicore.from_blob(
        torch_a.data_ptr(),
        a_shape,
        infinicore.float32,
        infinicore.device("cpu", 0),
    )

    infini_b = infinicore.from_blob(
        torch_b.data_ptr(),
        b_shape,
        infinicore.float32,
        infinicore.device("cpu", 0),
    )

    # 测试out-of-place matmul
    infini_result = infinicore.matmul(infini_a, infini_b)

    # 验证结果
    torch_result_from_infini = torch.zeros(
        result_shape, dtype=torch.float32, device="cpu"
    )
    temp_tensor = infinicore.from_blob(
        torch_result_from_infini.data_ptr(),
        result_shape,
        infinicore.float32,
        infinicore.device("cpu", 0),
    )
    temp_tensor.copy_(infini_result)

    assert torch.allclose(
        torch_result, torch_result_from_infini, rtol=1e-5
    ), "Basic matmul test failed"
    print("✓ Basic matmul test passed")


def test_matmul_inplace():
    """测试原地矩阵乘法"""
    print("Testing in-place matmul...")

    a_shape = [2, 3]
    b_shape = [3, 4]
    result_shape = [2, 4]

    torch_a = torch.rand(a_shape, dtype=torch.float32, device="cpu")
    torch_b = torch.rand(b_shape, dtype=torch.float32, device="cpu")
    torch_result = torch.matmul(torch_a, torch_b)

    # 创建预分配的结果张量
    torch_preallocated = torch.zeros(result_shape, dtype=torch.float32, device="cpu")

    infini_a = infinicore.from_blob(
        torch_a.data_ptr(),
        a_shape,
        infinicore.float32,
        infinicore.device("cpu", 0),
    )

    infini_b = infinicore.from_blob(
        torch_b.data_ptr(),
        b_shape,
        infinicore.float32,
        infinicore.device("cpu", 0),
    )

    infini_c = infinicore.from_blob(
        torch_preallocated.data_ptr(),
        result_shape,
        infinicore.float32,
        infinicore.device("cpu", 0),
    )

    # 测试in-place matmul
    infinicore.matmul_(infini_c, infini_a, infini_b)

    assert torch.allclose(
        torch_result, torch_preallocated, rtol=1e-5
    ), "In-place matmul test failed"
    print("✓ In-place matmul test passed")


def test_matmul_gpu():
    """测试GPU上的矩阵乘法"""
    print("Testing GPU matmul...")

    if not torch.cuda.is_available():
        print("⏭️  GPU not available, skipping GPU test")
        return

    a_shape = [3, 4]
    b_shape = [4, 5]
    result_shape = [3, 5]

    # 创建CPU张量
    torch_a_cpu = torch.rand(a_shape, dtype=torch.float32, device="cuda")
    torch_b_cpu = torch.rand(b_shape, dtype=torch.float32, device="cuda")
    torch_result = torch.matmul(torch_a_cpu, torch_b_cpu)

    # 转移到GPU
    torch_a_gpu = torch_a_cpu.cuda()
    torch_b_gpu = torch_b_cpu.cuda()

    # 创建infinicore GPU张量
    infini_a_gpu = infinicore.from_blob(
        torch_a_gpu.data_ptr(),
        a_shape,
        infinicore.float32,
        infinicore.device("cuda", 0),
    )

    infini_b_gpu = infinicore.from_blob(
        torch_b_gpu.data_ptr(),
        b_shape,
        infinicore.float32,
        infinicore.device("cuda", 0),
    )

    # 在GPU上执行matmul
    infini_result = infinicore.matmul(infini_a_gpu, infini_b_gpu)

    # 将结果转移回CPU验证
    infini_result = infinicore.matmul(infini_a_gpu, infini_b_gpu)

    torch_result_from_infini = torch.zeros(
        result_shape, dtype=torch.float32, device="cuda"
    )
    temp_tensor = infinicore.from_blob(
        torch_result_from_infini.data_ptr(),
        result_shape,
        infinicore.float32,
        infinicore.device("cuda", 0),
    )
    temp_tensor.copy_(infini_result)

    assert torch.allclose(
        torch_result, torch_result_from_infini, rtol=1e-5
    ), "GPU matmul test failed"
    print("✓ GPU matmul test passed")


def test_matmul_batch():
    """测试批量矩阵乘法"""
    print("Testing batch matmul...")

    batch_size = 2
    a_shape = [batch_size, 3, 4]
    b_shape = [batch_size, 4, 5]
    result_shape = [batch_size, 3, 5]

    torch_a = torch.rand(a_shape, dtype=torch.float32, device="cpu")
    torch_b = torch.rand(b_shape, dtype=torch.float32, device="cpu")
    torch_result = torch.bmm(torch_a, torch_b)  # 批量矩阵乘法

    infini_a = infinicore.from_blob(
        torch_a.data_ptr(),
        a_shape,
        infinicore.float32,
        infinicore.device("cpu", 0),
    )

    infini_b = infinicore.from_blob(
        torch_b.data_ptr(),
        b_shape,
        infinicore.float32,
        infinicore.device("cpu", 0),
    )

    infini_result = infinicore.matmul(infini_a, infini_b)

    torch_result_from_infini = torch.zeros(
        result_shape, dtype=torch.float32, device="cpu"
    )
    temp_tensor = infinicore.from_blob(
        torch_result_from_infini.data_ptr(),
        result_shape,
        infinicore.float32,
        infinicore.device("cpu", 0),
    )
    temp_tensor.copy_(infini_result)

    assert torch.allclose(
        torch_result, torch_result_from_infini, rtol=1e-5
    ), "Batch matmul test failed"
    print("✓ Batch matmul test passed")


def test_matmul_large():
    """测试大矩阵乘法"""
    print("Testing large matmul...")

    a_shape = [128, 256]
    b_shape = [256, 64]
    result_shape = [128, 64]

    torch_a = torch.rand(a_shape, dtype=torch.float32, device="cpu")
    torch_b = torch.rand(b_shape, dtype=torch.float32, device="cpu")
    torch_result = torch.matmul(torch_a, torch_b)

    infini_a = infinicore.from_blob(
        torch_a.data_ptr(),
        a_shape,
        infinicore.float32,
        infinicore.device("cpu", 0),
    )

    infini_b = infinicore.from_blob(
        torch_b.data_ptr(),
        b_shape,
        infinicore.float32,
        infinicore.device("cpu", 0),
    )

    infini_result = infinicore.matmul(infini_a, infini_b)

    torch_result_from_infini = torch.zeros(
        result_shape, dtype=torch.float32, device="cpu"
    )
    temp_tensor = infinicore.from_blob(
        torch_result_from_infini.data_ptr(),
        result_shape,
        infinicore.float32,
        infinicore.device("cpu", 0),
    )
    temp_tensor.copy_(infini_result)

    assert torch.allclose(
        torch_result, torch_result_from_infini, rtol=1e-5
    ), "Large matmul test failed"
    print("✓ Large matmul test passed")


def run_all_tests():
    """运行所有测试"""
    print("Starting matmul tests...\n")

    try:
        test_matmul_basic()
        test_matmul_inplace()
        test_matmul_batch()
        test_matmul_large()
        test_matmul_gpu()

        print("\n🎉 All matmul tests passed!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
