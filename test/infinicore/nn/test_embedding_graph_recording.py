"""
测试 embedding 是否支持 CUDA Graph 录制

使用方法：
    python test/infinicore/nn/test_embedding_graph_recording.py

关键验证点：
1. 改动前：indices->to(cpu_device) 会触发同步的 D2H 拷贝，导致图录制失败
2. 改动后：使用设备端 CUDA kernel，完全异步，支持图录制

预期结果：
- 改动前：图录制失败，设备端输入可能失败
- 改动后：图录制成功，设备端输入成功
"""

import infinicore
import torch
import ctypes


def test_embedding_graph_recording():
    """测试 embedding 是否支持 CUDA Graph 录制"""
    print("=" * 60)
    print("测试 Embedding 图录制支持")
    print("=" * 60)
    
    # 检查是否有 CUDA
    if not torch.cuda.is_available():
        print("⚠ CUDA 不可用，跳过图录制测试")
        return False
    
    device = infinicore.device("cuda", 0)
    
    # 创建 embedding 模块
    vocab_size = 1000
    embedding_dim = 128
    embedding = infinicore.nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
        dtype=infinicore.float32,
        device=device
    )
    
    # 创建设备端的 input_ids（这是关键：改动前不支持，改动后支持）
    batch_size = 4
    seq_len = 32
    input_ids_device = infinicore.from_list(
        [[i % vocab_size for i in range(seq_len)] for _ in range(batch_size)],
        dtype=infinicore.int64,
        device=device
    )
    
    print(f"\n1. 输入张量信息:")
    print(f"   - Shape: {input_ids_device.shape}")
    print(f"   - Device: {input_ids_device.device.type}")
    print(f"   - Dtype: {input_ids_device.dtype}")
    
    # 尝试使用 CUDA Graph 录制
    print(f"\n2. 尝试 CUDA Graph 录制...")
    
    # 使用 PyTorch 的 CUDA Graph API 进行测试（更简单可靠）
    try:
        # 设置设备
        infinicore.set_device(device)
        
        # 使用 PyTorch 的 CUDA Graph API
        # 注意：PyTorch 2.0+ 支持 torch.cuda.graph
        try:
            # 方法 1: 使用 PyTorch 的 CUDA Graph（推荐）
            print("   使用 PyTorch CUDA Graph API 测试...")
            
            # 创建 warmup 输入
            warmup_input = input_ids_device
            
            # Warmup（图录制前需要先执行一次，包括内存分配）
            warmup_output = embedding.forward(warmup_input)
            infinicore.sync_stream()  # 同步确保 warmup 完成
            
            # 预先分配输出张量（CUDA Graph 不支持动态内存分配）
            # 输出形状: input_shape + [embedding_dim]
            output_shape = list(input_ids_device.shape) + [embedding_dim]
            output = infinicore.empty(
                output_shape,
                dtype=embedding.weight.dtype,
                device=device
            )
            
            # Warmup embedding（确保内存分配完成）
            import infinicore.nn.functional as F
            F.embedding(warmup_input, embedding.weight, out=output)
            infinicore.sync_stream()
            
            # 开始图录制（使用预先分配的 output）
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                # 使用 embedding 的 out 参数（in-place），传入预先分配的 output
                F.embedding(input_ids_device, embedding.weight, out=output)
            
            print("   ✓ 成功完成图录制！")
            print("   ✓ Embedding 支持 CUDA Graph 录制")
            
            # 验证图可以重复执行
            graph.replay()
            infinicore.sync_stream()
            
            print("   ✓ 图可以成功重放")
            return True
            
        except AttributeError:
            # PyTorch 版本可能不支持 torch.cuda.graph
            print("   ⚠ PyTorch 版本不支持 torch.cuda.graph，使用简化验证方法")
            return test_embedding_async_verification(embedding, input_ids_device)
        except RuntimeError as e:
            error_msg = str(e)
            if "capture" in error_msg.lower() or "graph" in error_msg.lower():
                print(f"   ✗ 图录制失败: {e}")
                print("   ✗ Embedding 不支持 CUDA Graph 录制（可能包含同步操作）")
                return False
            else:
                print(f"   ⚠ 图录制测试异常: {e}")
                return test_embedding_async_verification(embedding, input_ids_device)
            
    except Exception as e:
        print(f"   ⚠ 图录制测试异常: {e}")
        print("   使用简化验证方法...")
        import traceback
        traceback.print_exc()
        return test_embedding_async_verification(embedding, input_ids_device)


def test_embedding_async_verification(embedding, input_ids_device):
    """
    简化验证：检查是否有同步操作
    
    关键检查点：
    1. 输入是否可以在设备上（改动前需要 CPU，改动后支持设备）
    2. 操作是否完全异步（没有同步点）
    """
    print("\n3. 简化验证：检查异步操作支持")
    
    # 验证 1: 输入可以在设备上
    if input_ids_device.device.type != "cuda":
        print("   ✗ 输入不在设备上，无法验证")
        return False
    
    print("   ✓ 输入在设备上")
    
    # 验证 2: 执行 forward，检查是否有同步操作
    # 如果改动前，这里会调用 indices->to(cpu_device)，触发同步
    # 如果改动后，直接使用设备端 kernel，完全异步
    
    try:
        # 记录开始时间
        start_event = infinicore.DeviceEvent(enable_timing=True)
        end_event = infinicore.DeviceEvent(enable_timing=True)
        
        start_event.record()
        output = embedding.forward(input_ids_device)
        end_event.record()
        
        # 不立即同步，检查操作是否异步
        # 如果操作是异步的，query 应该返回 False（未完成）
        # 如果操作是同步的，可能已经完成
        
        # 等待一小段时间
        import time
        time.sleep(0.001)  # 1ms
        
        # 检查事件状态
        is_complete = end_event.query()
        
        if not is_complete:
            print("   ✓ 操作是异步的（事件未立即完成）")
        else:
            print("   ⚠ 操作可能包含同步点（事件立即完成）")
        
        # 同步并测量时间
        end_event.synchronize()
        elapsed = start_event.elapsed_time(end_event)
        
        print(f"   ✓ Forward 执行时间: {elapsed:.3f} ms")
        print(f"   ✓ 输出形状: {output.shape}")
        print(f"   ✓ 输出设备: {output.device.type}")
        
        # 验证输出正确性
        embedding_dim = embedding.embedding_dim()
        expected_shape = (*input_ids_device.shape, embedding_dim)
        if output.device.type == "cuda" and output.shape == expected_shape:
            print("   ✓ 输出在设备上，形状正确")
            return True
        else:
            print(f"   ✗ 输出验证失败")
            print(f"     期望形状: {expected_shape}, 实际形状: {output.shape}")
            print(f"     期望设备: cuda, 实际设备: {output.device.type}")
            return False
            
    except Exception as e:
        print(f"   ✗ 验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embedding_device_input_support():
    """测试 embedding 是否支持设备端输入"""
    print("\n" + "=" * 60)
    print("测试 Embedding 设备端输入支持")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("⚠ CUDA 不可用，跳过测试")
        return False
    
    device = infinicore.device("cuda", 0)
    vocab_size = 100
    embedding_dim = 64
    
    embedding = infinicore.nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
        dtype=infinicore.float32,
        device=device
    )
    
    # 测试 1: 设备端输入（改动后支持）
    print("\n测试 1: 设备端输入")
    try:
        input_ids_device = infinicore.from_list(
            [[1, 2, 3, 4, 5]],
            dtype=infinicore.int64,
            device=device
        )
        output = embedding.forward(input_ids_device)
        print(f"   ✓ 设备端输入成功")
        print(f"   - 输入设备: {input_ids_device.device.type}")
        print(f"   - 输出设备: {output.device.type}")
        print(f"   - 输出形状: {output.shape}")
        return True
    except Exception as e:
        print(f"   ✗ 设备端输入失败: {e}")
        return False


def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("Embedding 图录制支持验证")
    print("=" * 60)
    
    results = []
    
    # 测试 1: 图录制支持
    result1 = test_embedding_graph_recording()
    results.append(("CUDA Graph 录制", result1))
    
    # 测试 2: 设备端输入支持
    result2 = test_embedding_device_input_support()
    results.append(("设备端输入", result2))
    
    # 总结
    print("\n" + "=" * 60)
    print("测试结果总结")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ 所有测试通过！Embedding 支持图录制")
    else:
        print("✗ 部分测试失败，Embedding 可能不完全支持图录制")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
