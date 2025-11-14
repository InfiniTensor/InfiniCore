#!/usr/bin/env python3
"""
测试从Python list创建infinicore.Tensor的功能
"""

import sys
import os

# 添加python目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))

import infinicore


def test_basic_1d():
    """测试一维list"""
    print("=" * 50)
    print("测试1: 一维整数list")
    print("=" * 50)
    data = [1, 2, 3, 4, 5]
    t = infinicore.tensor(data)
    print(f"输入: {data}")
    print(f"Shape: {t.shape}")
    print(f"Dtype: {t.dtype}")
    print(f"Device: {t.device}")
    print(f"Numel: {t.numel()}")
    t.debug()
    print()


def test_basic_1d_float():
    """测试一维浮点数list"""
    print("=" * 50)
    print("测试2: 一维浮点数list")
    print("=" * 50)
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    t = infinicore.tensor(data)
    print(f"输入: {data}")
    print(f"Shape: {t.shape}")
    print(f"Dtype: {t.dtype}")
    t.debug()
    print()


def test_2d():
    """测试二维list"""
    print("=" * 50)
    print("测试3: 二维list")
    print("=" * 50)
    data = [[1, 2, 3], [4, 5, 6]]
    t = infinicore.tensor(data)
    print(f"输入: {data}")
    print(f"Shape: {t.shape}")
    print(f"Dtype: {t.dtype}")
    t.debug()
    print()


def test_3d():
    """测试三维list"""
    print("=" * 50)
    print("测试4: 三维list")
    print("=" * 50)
    data = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
    t = infinicore.tensor(data)
    print(f"输入: {data}")
    print(f"Shape: {t.shape}")
    print(f"Dtype: {t.dtype}")
    t.debug()
    print()


def test_with_dtype():
    """测试指定dtype"""
    print("=" * 50)
    print("测试5: 指定dtype为float16")
    print("=" * 50)
    data = [1, 2, 3, 4, 5]
    t = infinicore.tensor(data, dtype=infinicore.float16)
    print(f"输入: {data}")
    print(f"Shape: {t.shape}")
    print(f"Dtype: {t.dtype}")
    t.debug()
    print()


def test_bool():
    """测试布尔类型"""
    print("=" * 50)
    print("测试6: 布尔类型list")
    print("=" * 50)
    data = [True, False, True, False]
    t = infinicore.tensor(data)
    print(f"输入: {data}")
    print(f"Shape: {t.shape}")
    print(f"Dtype: {t.dtype}")
    t.debug()
    print()


def test_mixed_types():
    """测试混合类型（会被转换为float）"""
    print("=" * 50)
    print("测试7: 混合整数和浮点数（自动推断为float）")
    print("=" * 50)
    data = [1, 2.0, 3, 4.5]
    t = infinicore.tensor(data)
    print(f"输入: {data}")
    print(f"Shape: {t.shape}")
    print(f"Dtype: {t.dtype}")
    t.debug()
    print()


def test_large_tensor():
    """测试较大的tensor"""
    print("=" * 50)
    print("测试8: 较大的tensor (10x10)")
    print("=" * 50)
    data = [[i * 10 + j for j in range(10)] for i in range(10)]
    t = infinicore.tensor(data)
    print(f"Shape: {t.shape}")
    print(f"Dtype: {t.dtype}")
    print(f"Numel: {t.numel()}")
    # 只打印前几个元素
    print("前几个元素:")
    t.debug()
    print()


def test_comparison_with_torch():
    """与torch.Tensor对比（如果torch可用）"""
    print("=" * 50)
    print("测试9: 与torch.Tensor对比")
    print("=" * 50)
    try:
        import torch
        data = [[1, 2, 3], [4, 5, 6]]
        
        # infinicore
        t_infini = infinicore.tensor(data)
        print("infinicore.Tensor:")
        print(f"  Shape: {t_infini.shape}")
        print(f"  Dtype: {t_infini.dtype}")
        
        # torch
        t_torch = torch.Tensor(data)
        print("torch.Tensor:")
        print(f"  Shape: {t_torch.shape}")
        print(f"  Dtype: {t_torch.dtype}")

        assert str(t_infini.dtype) == "infinicore.float32"
        assert t_torch.dtype == torch.float32
        print("✓ 功能与 dtype 对齐成功")
    except ImportError:
        print("torch未安装，跳过对比测试")
    print()


def test_error_cases():
    """测试错误情况"""
    print("=" * 50)
    print("测试10: 错误情况")
    print("=" * 50)
    
    # 空list
    try:
        t = infinicore.tensor([])
        print("✗ 空list应该报错但没有")
    except ValueError as e:
        print(f"✓ 空list正确报错: {e}")
    
    # 不一致的shape
    try:
        t = infinicore.tensor([[1, 2], [3, 4, 5]])
        print("✗ 不一致的shape应该报错但没有")
    except ValueError as e:
        print(f"✓ 不一致的shape正确报错: {e}")
    
    # 非list类型
    try:
        t = infinicore.tensor("not a list")
        print("✗ 非list类型应该报错但没有")
    except TypeError as e:
        print(f"✓ 非list类型正确报错: {e}")
    
    print()


def _run_device_case(device_str):
    print("=" * 50)
    print(f"测试11: device={device_str}")
    print("=" * 50)

    try:
        target_device = infinicore.device(device_str)
    except Exception as exc:
        print(f"{device_str} 不可用，跳过（原因: {exc})")
        return False

    data = [1.0, 2.0, 3.0]
    if device_str not in ("cpu", "cpu:0"):
        print(
            f"{device_str} 暂不支持直接从 Python list 创建 tensor；"
            "请先创建 CPU tensor 再调用 .to(device)"
        )
        return False

    try:
        t = infinicore.tensor(data)
        print(f"输入: {data}")
        print(f"Shape: {t.shape}")
        print(f"Dtype: {t.dtype}")
        print(f"Device: {t.device}")

        return True
    except Exception as exc:
        print(f"{device_str} 测试失败: {exc}")
        return False


def test_device_targets():
    """测试指定 device（如 CUDA）。"""
    device_list = os.environ.get("INFINICORE_DEVICE_TESTS")
    targets = ["cpu"]
    if device_list:
        targets.extend(
            item.strip() for item in device_list.split(",") if item.strip()
        )
    else:
        print("=" * 50)
        print("测试11: device（仅 CPU，设置 INFINICORE_DEVICE_TESTS 可测试 GPU）")
        print("=" * 50)
        print("未设置 INFINICORE_DEVICE_TESTS，默认仅验证 CPU device。")
        print()

    any_success = False
    for device_str in targets:
        success = _run_device_case(device_str)
        any_success = any_success or success
        print()

    if not any_success:
        print("未检测到可用的 device，device 测试全部跳过。")


def main():
    """运行所有测试"""
    print("\n" + "=" * 50)
    print("开始测试 infinicore.tensor() 功能")
    print("=" * 50 + "\n")
    
    try:
        test_basic_1d()
        test_basic_1d_float()
        test_2d()
        test_3d()
        test_with_dtype()
        test_bool()
        test_mixed_types()
        test_large_tensor()
        test_comparison_with_torch()
        test_error_cases()
        test_device_targets()
        
        print("=" * 50)
        print("所有测试完成！")
        print("=" * 50)
        
    except Exception as e:
        print(f"\n测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

