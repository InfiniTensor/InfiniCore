#!/usr/bin/env python3
"""
Tensor Debug 功能测试脚本

简单测试 debug 功能是否正常工作
"""

import torch
import infinicore
import sys
import os
import numpy as np


def test_basic_debug():
    """测试基本的 debug 打印功能"""
    print("\n" + "=" * 80)
    print("测试 1: 基本 debug 打印")
    print("=" * 80)
    
    device = infinicore.device("cpu", 0)
    
    # 测试 float32
    print("\n--- Float32 张量 (2x3) ---")
    torch_tensor = torch.tensor([[1.0, 2.0, 3.0],
                                  [4.0, 5.0, 6.0]], dtype=torch.float32)
    infini_tensor = infinicore.from_blob(
        torch_tensor.data_ptr(),
        list(torch_tensor.shape),
        dtype=infinicore.float32,
        device=device
    )
    infini_tensor.debug()
    print("✓ Float32 打印成功")
    
    # 测试 int32
    print("\n--- Int32 张量 (2x2) ---")
    torch_i32 = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    infini_i32 = infinicore.from_blob(
        torch_i32.data_ptr(),
        list(torch_i32.shape),
        dtype=infinicore.int32,
        device=device
    )
    infini_i32.debug()
    print("✓ Int32 打印成功")


def test_save_to_file():
    """测试保存到文件"""
    print("\n" + "=" * 80)
    print("测试 2: 保存张量到文件")
    print("=" * 80)
    
    device = infinicore.device("cpu", 0)
    
    # 创建张量
    torch_tensor = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
    print("\n原始张量:")
    print(torch_tensor)
    
    infini_tensor = infinicore.from_blob(
        torch_tensor.data_ptr(),
        list(torch_tensor.shape),
        dtype=infinicore.float32,
        device=device
    )
    
    # 保存到文件
    filename = "/tmp/tensor_debug_test.bin"
    print(f"\n保存到: {filename}")
    infini_tensor.debug(filename)
    
    # 验证文件
    if os.path.exists(filename):
        file_size = os.path.getsize(filename)
        expected_size = 12 * 4  # 12 个 float32
        assert file_size == expected_size, f"文件大小不匹配: {file_size} vs {expected_size}"
        
        # 读取验证
        loaded = np.fromfile(filename, dtype=np.float32).reshape(3, 4)
        print("\n从文件读取:")
        print(loaded)
        
        os.remove(filename)
        print("✓ 文件保存和读取成功")
    else:
        raise RuntimeError("文件未创建")


def test_multidimensional():
    """测试多维张量"""
    print("\n" + "=" * 80)
    print("测试 3: 多维张量")
    print("=" * 80)
    
    device = infinicore.device("cpu", 0)
    
    # 3D 张量
    print("\n--- 3D 张量 (2x2x3) ---")
    torch_3d = torch.arange(1, 13, dtype=torch.float32).reshape(2, 2, 3)
    print("PyTorch 张量:")
    print(torch_3d)
    
    infini_3d = infinicore.from_blob(
        torch_3d.data_ptr(),
        list(torch_3d.shape),
        dtype=infinicore.float32,
        device=device
    )
    
    print("\nInfiniCore debug 输出:")
    infini_3d.debug()
    print("✓ 3D 张量打印成功")


def test_infinicore_created():
    """测试 InfiniCore 创建的张量"""
    print("\n" + "=" * 80)
    print("测试 4: InfiniCore 创建的张量")
    print("=" * 80)
    
    device = infinicore.device("cpu", 0)
    
    # ones 张量
    print("\n--- ones 张量 (2x3) ---")
    ones_tensor = infinicore.ones([2, 3], dtype=infinicore.float32, device=device)
    ones_tensor.debug()
    print("✓ ones 张量打印成功")
    
    # zeros 张量
    print("\n--- zeros 张量 (3x2) ---")
    zeros_tensor = infinicore.zeros([3, 2], dtype=infinicore.float32, device=device)
    zeros_tensor.debug()
    print("✓ zeros 张量打印成功")


def test_different_dtypes():
    """测试不同数据类型"""
    print("\n" + "=" * 80)
    print("测试 5: 不同数据类型")
    print("=" * 80)
    
    device = infinicore.device("cpu", 0)
    
    dtypes = [
        (infinicore.float32, torch.float32, "Float32"),
        (infinicore.int32, torch.int32, "Int32"),
        (infinicore.int64, torch.int64, "Int64"),
    ]
    
    for infini_dtype, torch_dtype, name in dtypes:
        print(f"\n--- {name} ---")
        torch_tensor = torch.arange(1, 7, dtype=torch_dtype).reshape(2, 3)
        infini_tensor = infinicore.from_blob(
            torch_tensor.data_ptr(),
            list(torch_tensor.shape),
            dtype=infini_dtype,
            device=device
        )
        infini_tensor.debug()
        print(f"✓ {name} 测试通过")


def test_text_format():
    """测试文本格式保存"""
    print("\n" + "=" * 80)
    print("测试 6: 文本格式保存 (.txt)")
    print("=" * 80)
    
    device = infinicore.device("cpu", 0)
    
    # 创建张量
    torch_tensor = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
    print("\n原始张量:")
    print(torch_tensor)
    
    infini_tensor = infinicore.from_blob(
        torch_tensor.data_ptr(),
        list(torch_tensor.shape),
        dtype=infinicore.float32,
        device=device
    )
    
    # 保存为文本文件
    txt_filename = "/tmp/tensor_debug_test.txt"
    print(f"\n保存为文本格式: {txt_filename}")
    infini_tensor.debug(txt_filename)
    
    # 验证文本文件
    if os.path.exists(txt_filename):
        print("\n文本文件内容:")
        with open(txt_filename, 'r') as f:
            content = f.read()
            print(content)
        
        # 1. 验证元数据
        assert "# Tensor Debug Output" in content, "文本文件缺少标题"
        assert "# Shape: [3, 4]" in content, "文本文件缺少形状信息"
        assert "# Dtype: F32" in content, "文本文件缺少类型信息"
        print("✓ 元数据验证通过")
        
        # 2. 提取并验证数值数据
        lines = content.split('\n')
        data_lines = [line.strip() for line in lines 
                      if line.strip() and not line.startswith('#')]
        
        print(f"\n提取到 {len(data_lines)} 行数据")
        
        # 解析数值
        loaded_data = []
        for i, line in enumerate(data_lines):
            row = [float(x) for x in line.split()]
            loaded_data.append(row)
            print(f"  第 {i+1} 行: {row}")
        
        # 转换为 numpy 数组
        loaded_array = np.array(loaded_data, dtype=np.float32)
        
        # 3. 与原始数据对比
        expected = torch_tensor.numpy()
        assert loaded_array.shape == expected.shape, \
            f"形状不匹配: {loaded_array.shape} vs {expected.shape}"
        assert np.allclose(loaded_array, expected), \
            f"数值不匹配:\n加载的数据:\n{loaded_array}\n期望的数据:\n{expected}"
        
        print("✓ 数值验证通过")
        
        os.remove(txt_filename)
        print("✓ 文本格式保存测试通过")
    else:
        raise RuntimeError("文本文件未创建")


def test_binary_format():
    """测试二进制格式保存"""
    print("\n" + "=" * 80)
    print("测试 7: 二进制格式保存 (.bin)")
    print("=" * 80)
    
    device = infinicore.device("cpu", 0)
    
    # 创建张量
    torch_tensor = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
    print("\n原始张量:")
    print(torch_tensor)
    
    infini_tensor = infinicore.from_blob(
        torch_tensor.data_ptr(),
        list(torch_tensor.shape),
        dtype=infinicore.float32,
        device=device
    )
    
    # 保存为二进制文件
    bin_filename = "/tmp/tensor_debug_test.bin"
    print(f"\n保存为二进制格式: {bin_filename}")
    infini_tensor.debug(bin_filename)
    
    # 验证二进制文件
    if os.path.exists(bin_filename):
        file_size = os.path.getsize(bin_filename)
        expected_size = 12 * 4  # 12 个 float32
        assert file_size == expected_size, \
            f"二进制文件大小不匹配: {file_size} vs {expected_size}"
        
        # 读取并验证数据
        loaded = np.fromfile(bin_filename, dtype=np.float32).reshape(3, 4)
        print("\n从二进制文件读取:")
        print(loaded)
        
        # 验证数据正确性
        assert np.allclose(loaded, torch_tensor.numpy()), "数据不匹配"
        
        os.remove(bin_filename)
        print("✓ 二进制格式保存测试通过")
    else:
        raise RuntimeError("二进制文件未创建")


def test_format_comparison():
    """对比不同格式"""
    print("\n" + "=" * 80)
    print("测试 8: 对比不同格式")
    print("=" * 80)
    
    device = infinicore.device("cpu", 0)
    
    # 创建小张量用于对比
    torch_tensor = torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float32)
    print("\n原始张量:")
    print(torch_tensor)
    
    infini_tensor = infinicore.from_blob(
        torch_tensor.data_ptr(),
        list(torch_tensor.shape),
        dtype=infinicore.float32,
        device=device
    )
    
    bin_file = "/tmp/compare_test.bin"
    txt_file = "/tmp/compare_test.txt"
    
    # 保存两种格式
    print("\n保存两种格式...")
    infini_tensor.debug(bin_file)
    infini_tensor.debug(txt_file)
    
    # 对比文件大小
    bin_size = os.path.getsize(bin_file)
    txt_size = os.path.getsize(txt_file)
    
    print(f"\n文件大小对比:")
    print(f"  二进制文件: {bin_size} 字节")
    print(f"  文本文件: {txt_size} 字节")
    print(f"  文本/二进制比: {txt_size/bin_size:.2f}x")
    
    # ===== 验证二进制文件 =====
    print("\n验证二进制文件:")
    bin_data = np.fromfile(bin_file, dtype=np.float32).reshape(2, 2)
    print(f"  读取的数据:\n{bin_data}")
    assert np.allclose(bin_data, torch_tensor.numpy()), "二进制数据不匹配"
    print("  ✓ 二进制文件数值正确")
    
    # ===== 验证文本文件 =====
    print("\n验证文本文件:")
    with open(txt_file, 'r') as f:
        txt_content = f.read()
    
    # 1. 元数据验证
    assert "# Tensor Debug Output" in txt_content, "缺少标题"
    assert "# Shape: [2, 2]" in txt_content, "缺少形状信息"
    assert "# Dtype: F32" in txt_content, "缺少类型信息"
    print("  ✓ 元数据正确")
    
    # 2. 数值验证
    lines = txt_content.split('\n')
    data_lines = [line.strip() for line in lines 
                  if line.strip() and not line.startswith('#')]
    
    txt_data = []
    for line in data_lines:
        row = [float(x) for x in line.split()]
        txt_data.append(row)
    
    txt_array = np.array(txt_data, dtype=np.float32)
    print(f"  读取的数据:\n{txt_array}")
    
    assert txt_array.shape == torch_tensor.shape, \
        f"文本文件形状不匹配: {txt_array.shape} vs {torch_tensor.shape}"
    assert np.allclose(txt_array, torch_tensor.numpy()), \
        f"文本文件数值不匹配"
    print("  ✓ 文本文件数值正确")
    
    # ===== 对比两种格式的数据一致性 =====
    print("\n验证两种格式数据一致性:")
    assert np.allclose(bin_data, txt_array), \
        "二进制和文本文件的数据不一致！"
    print("  ✓ 两种格式数据完全一致")
    
    # 清理
    os.remove(bin_file)
    os.remove(txt_file)
    
    print("\n✓ 格式对比测试通过")


def main():
    """主测试函数"""
    print("\n" + "=" * 80)
    print("InfiniCore Tensor Debug 功能测试")
    print("=" * 80)
    
    try:
        test_basic_debug()
        test_save_to_file()
        test_multidimensional()
        test_infinicore_created()
        test_different_dtypes()
        test_text_format()
        test_binary_format()
        test_format_comparison()
        
        print("\n" + "=" * 80)
        print("✅ 所有测试通过!")
        print("=" * 80)
        return 0
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

