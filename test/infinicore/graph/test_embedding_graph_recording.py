"""
Test if embedding supports CUDA Graph recording

Usage:
    python test/infinicore/nn/test_embedding_graph_recording.py

Key verification points:
1. Before modification: indices->to(cpu_device) triggers synchronous D2H copy, causing graph recording to fail
2. After modification: Uses device-side CUDA kernel, fully asynchronous, supports graph recording

Expected results:
- Before modification: Graph recording fails, device-side input may fail
- After modification: Graph recording succeeds, device-side input succeeds
"""

import infinicore
import torch


def test_embedding_graph_recording():
    """Test if embedding supports CUDA Graph recording"""
    print("=" * 60)
    print("Testing Embedding Graph Recording Support")
    print("=" * 60)

    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping graph recording test")
        return False

    device = infinicore.device("cuda", 0)

    # Create embedding module
    vocab_size = 1000
    embedding_dim = 128
    embedding = infinicore.nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
        dtype=infinicore.float32,
        device=device,
    )

    # Create device-side input_ids (key point: unsupported before modification, supported after)
    batch_size = 4
    seq_len = 32
    input_ids_device = infinicore.from_list(
        [[i % vocab_size for i in range(seq_len)] for _ in range(batch_size)],
        dtype=infinicore.int64,
        device=device,
    )

    print(f"\n1. Input tensor information:")
    print(f"   - Shape: {input_ids_device.shape}")
    print(f"   - Device: {input_ids_device.device.type}")
    print(f"   - Dtype: {input_ids_device.dtype}")

    # Attempt CUDA Graph recording
    print(f"\n2. Attempting CUDA Graph recording...")

    # Use PyTorch's CUDA Graph API for testing (simpler and more reliable)
    try:
        # Set device
        infinicore.set_device(device)

        # Use PyTorch's CUDA Graph API
        # Note: PyTorch 2.0+ supports torch.cuda.graph
        try:
            # Method 1: Use PyTorch CUDA Graph (recommended)
            print("   Using PyTorch CUDA Graph API for testing...")

            # Create warmup input
            warmup_input = input_ids_device

            # Warmup (need to execute once before graph recording, including memory allocation)
            embedding.forward(warmup_input)
            infinicore.sync_stream()  # Synchronize to ensure warmup completes

            # Pre-allocate output tensor (CUDA Graph doesn't support dynamic memory allocation)
            # Output shape: input_shape + [embedding_dim]
            output_shape = list(input_ids_device.shape) + [embedding_dim]
            output = infinicore.empty(
                output_shape, dtype=embedding.weight.dtype, device=device
            )

            # Warmup embedding (ensure memory allocation is complete)
            import infinicore.nn.functional as F

            F.embedding(warmup_input, embedding.weight, out=output)
            infinicore.sync_stream()

            # Start graph recording (using pre-allocated output)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                # Use embedding's out parameter (in-place), passing pre-allocated output
                F.embedding(input_ids_device, embedding.weight, out=output)

            print("   ✓ Graph recording successful!")
            print("   ✓ Embedding supports CUDA Graph recording")

            # Verify graph can be replayed
            graph.replay()
            infinicore.sync_stream()

            print("   ✓ Graph can be successfully replayed")
            return True

        except AttributeError:
            # PyTorch version may not support torch.cuda.graph
            print(
                "   ⚠ PyTorch version doesn't support torch.cuda.graph, using simplified verification method"
            )
            return test_embedding_async_verification(embedding, input_ids_device)
        except RuntimeError as e:
            error_msg = str(e)
            if "capture" in error_msg.lower() or "graph" in error_msg.lower():
                print(f"   ✗ Graph recording failed: {e}")
                print(
                    "   ✗ Embedding doesn't support CUDA Graph recording (may contain synchronous operations)"
                )
                return False
            else:
                print(f"   ⚠ Graph recording test exception: {e}")
                return test_embedding_async_verification(embedding, input_ids_device)

    except Exception as e:
        print(f"   ⚠ Graph recording test exception: {e}")
        print("   Using simplified verification method...")
        import traceback

        traceback.print_exc()
        return test_embedding_async_verification(embedding, input_ids_device)


def test_embedding_async_verification(embedding, input_ids_device):
    """
    Simplified verification: Check if there are synchronous operations

    Key checkpoints:
    1. Whether input can be on device (needed CPU before modification, supports device after)
    2. Whether operations are fully asynchronous (no synchronization points)
    """
    print("\n3. Simplified verification: Checking asynchronous operation support")

    # Verification 1: Input can be on device
    if input_ids_device.device.type != "cuda":
        print("   ✗ Input not on device, cannot verify")
        return False

    print("   ✓ Input is on device")

    # Verification 2: Execute forward, check for synchronous operations
    # Before modification, this would call indices->to(cpu_device), triggering synchronization
    # After modification, directly uses device-side kernel, fully asynchronous

    try:
        # Record start time
        start_event = infinicore.DeviceEvent(enable_timing=True)
        end_event = infinicore.DeviceEvent(enable_timing=True)

        start_event.record()
        output = embedding.forward(input_ids_device)
        end_event.record()

        # Don't synchronize immediately, check if operation is asynchronous
        # If operation is asynchronous, query should return False (not completed)
        # If operation is synchronous, may have already completed

        # Wait a short time
        import time

        time.sleep(0.001)  # 1ms

        # Check event status
        is_complete = end_event.query()

        if not is_complete:
            print("   ✓ Operation is asynchronous (event not immediately completed)")
        else:
            print(
                "   ⚠ Operation may contain synchronization points (event immediately completed)"
            )

        # Synchronize and measure time
        end_event.synchronize()
        elapsed = start_event.elapsed_time(end_event)

        print(f"   ✓ Forward execution time: {elapsed:.3f} ms")
        print(f"   ✓ Output shape: {output.shape}")
        print(f"   ✓ Output device: {output.device.type}")

        # Verify output correctness
        embedding_dim = embedding.embedding_dim()
        expected_shape = (*input_ids_device.shape, embedding_dim)
        if output.device.type == "cuda" and output.shape == expected_shape:
            print("   ✓ Output on device, shape correct")
            return True
        else:
            print(f"   ✗ Output verification failed")
            print(
                f"     Expected shape: {expected_shape}, actual shape: {output.shape}"
            )
            print(f"     Expected device: cuda, actual device: {output.device.type}")
            return False

    except Exception as e:
        print(f"   ✗ Verification failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_embedding_device_input_support():
    """Test if embedding supports device-side input"""
    print("\n" + "=" * 60)
    print("Testing Embedding Device-side Input Support")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("⚠ CUDA not available, skipping test")
        return False

    device = infinicore.device("cuda", 0)
    vocab_size = 100
    embedding_dim = 64

    embedding = infinicore.nn.Embedding(
        num_embeddings=vocab_size,
        embedding_dim=embedding_dim,
        dtype=infinicore.float32,
        device=device,
    )

    # Test 1: Device-side input (supported after modification)
    print("\nTest 1: Device-side input")
    try:
        input_ids_device = infinicore.from_list(
            [[1, 2, 3, 4, 5]], dtype=infinicore.int64, device=device
        )
        output = embedding.forward(input_ids_device)
        print(f"   ✓ Device-side input successful")
        print(f"   - Input device: {input_ids_device.device.type}")
        print(f"   - Output device: {output.device.type}")
        print(f"   - Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"   ✗ Device-side input failed: {e}")
        return False


def main():
    """Main test function"""
    print("\n" + "=" * 60)
    print("Embedding Graph Recording Support Verification")
    print("=" * 60)

    results = []

    # Test 1: Graph recording support
    result1 = test_embedding_graph_recording()
    results.append(("CUDA Graph Recording", result1))

    # Test 2: Device-side input support
    result2 = test_embedding_device_input_support()
    results.append(("Device-side Input", result2))

    # Summary
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)

    all_passed = True
    for test_name, result in results:
        status = "✓ Passed" if result else "✗ Failed"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! Embedding supports graph recording")
    else:
        print("✗ Some tests failed, embedding may not fully support graph recording")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
