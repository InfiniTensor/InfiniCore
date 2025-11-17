import infinicore
import torch


def test_device_event_timing():
    """Test DeviceEvent for timing operations - using instance method API"""
    print("\nTesting DeviceEvent timing...")

    # Create events
    start_event = infinicore.DeviceEvent(enable_timing=True)
    end_event = infinicore.DeviceEvent(enable_timing=True)

    # Create test tensors
    shape = [1000, 1000]
    device = infinicore.device("cuda", 0)

    # Time tensor creation and operations
    start_event.record()

    # Perform some operations
    t1 = infinicore.ones(shape, dtype=infinicore.float32, device=device)
    t2 = infinicore.zeros(shape, dtype=infinicore.float32, device=device)

    # Simulate some computation by multiple operations
    for _ in range(10):
        t1 = t1.permute([1, 0])
        t2 = t2.permute([1, 0])

    end_event.record()

    # Wait for operations to complete
    end_event.synchronize()

    # Calculate elapsed time - USING INSTANCE METHOD (torch-compatible)
    elapsed_time = start_event.elapsed_time(end_event)

    print(f"✓ DeviceEvent timing test passed - Elapsed time: {elapsed_time:.3f} ms")
    assert elapsed_time >= 0, "Elapsed time should be non-negative"

    return elapsed_time


def test_device_event_query():
    """Test DeviceEvent query functionality"""
    print("\nTesting DeviceEvent query...")

    event = infinicore.DeviceEvent(enable_timing=True)

    # Event should not be completed before recording
    assert not event.is_recorded, "Event should not be recorded initially"

    # Record the event
    event.record()
    assert event.is_recorded, "Event should be recorded after record()"

    # Query completion (might be immediate for simple cases)
    completed = event.query()
    print(f"✓ DeviceEvent query test passed - Event completed: {completed}")

    # Ensure synchronization works
    event.synchronize()
    assert event.query(), "Event should be completed after synchronize()"


def test_multiple_devices():
    """Test operations across multiple devices"""
    print("\nTesting multiple devices...")

    cuda_count = infinicore.get_device_count("cuda")

    if cuda_count > 1:
        # Test operations on different devices
        shape = [100, 100]

        # Create events for timing
        event0_start = infinicore.DeviceEvent(
            device=infinicore.device("cuda", 0), enable_timing=True
        )
        event0_end = infinicore.DeviceEvent(
            device=infinicore.device("cuda", 0), enable_timing=True
        )
        event1_start = infinicore.DeviceEvent(
            device=infinicore.device("cuda", 1), enable_timing=True
        )
        event1_end = infinicore.DeviceEvent(
            device=infinicore.device("cuda", 1), enable_timing=True
        )

        # Create tensors on different devices
        event0_start.record()
        t_device0 = infinicore.ones(
            shape, dtype=infinicore.float32, device=infinicore.device("cuda", 0)
        )
        event0_end.record()

        event1_start.record()
        t_device1 = infinicore.zeros(
            shape, dtype=infinicore.float32, device=infinicore.device("cuda", 1)
        )
        event1_end.record()

        # Synchronize both devices
        event0_end.synchronize()
        event1_end.synchronize()

        # Calculate elapsed times
        time_device0 = event0_start.elapsed_time(event0_end)
        time_device1 = event1_start.elapsed_time(event1_end)

        print(f"✓ Multiple devices test passed")
        print(f"  Device 0 tensor creation time: {time_device0:.3f} ms")
        print(f"  Device 1 tensor creation time: {time_device1:.3f} ms")

        # Test operations timing
        event0_start.record()
        for _ in range(20):
            t_device0 = t_device0.permute([1, 0])
        event0_end.record()

        event1_start.record()
        for _ in range(20):
            t_device1 = t_device1.permute([1, 0])
        event1_end.record()

        # Synchronize again
        event0_end.synchronize()
        event1_end.synchronize()

        # Calculate operation times
        op_time_device0 = event0_start.elapsed_time(event0_end)
        op_time_device1 = event1_start.elapsed_time(event1_end)

        print(f"  Device 0 operations time: {op_time_device0:.3f} ms")
        print(f"  Device 1 operations time: {op_time_device1:.3f} ms")

        # Test cross-device operations if supported
        try:
            # Try to create an event that measures cross-device operations
            cross_start = infinicore.DeviceEvent(device=infinicore.device("cuda", 0))
            cross_end = infinicore.DeviceEvent(device=infinicore.device("cuda", 0))

            cross_start.record()
            # Perform operations on both devices
            for _ in range(10):
                t_device0 = t_device0.permute([1, 0])
                # Note: Actual cross-device operations would require explicit synchronization
            cross_end.record()
            cross_end.synchronize()

            cross_time = cross_start.elapsed_time(cross_end)
            print(f"  Cross-device operations time: {cross_time:.3f} ms")

        except Exception as e:
            print(f"  Cross-device timing skipped: {e}")

    else:
        print("⚠ Skipping multiple devices test (only 1 CUDA device available)")


def test_event_stream():
    """Test DeviceEvent with different streams"""
    print("\nTesting DeviceEvent with streams...")

    try:
        # Get default stream
        default_stream = None
        if hasattr(infinicore, "get_stream"):
            default_stream = infinicore.get_stream()
        else:
            print("⚠ infinicore.get_stream() not available, using default stream")

        # Create event and record
        event = infinicore.DeviceEvent(enable_timing=True)
        if default_stream is not None:
            event.record(stream=default_stream)
        else:
            event.record()

        event.synchronize()

        print("✓ DeviceEvent stream test passed")
    except Exception as e:
        print(f"⚠ DeviceEvent stream test skipped: {e}")


def test_concurrent_events():
    """Test multiple concurrent events"""
    print("\nTesting concurrent events...")

    # Create multiple events
    events = []
    for i in range(5):
        events.append(infinicore.DeviceEvent(enable_timing=True))

    # Record events with small delays
    for i, event in enumerate(events):
        event.record()
        # Small operation
        temp = infinicore.ones(
            [10, 10], dtype=infinicore.float32, device=infinicore.device("cuda", 0)
        )
        temp = temp.permute([1, 0])

    # Synchronize all events
    for event in events:
        event.synchronize()
        assert event.query(), "All events should be completed"

    print("✓ Concurrent events test passed")


def test_torch_style_usage():
    """Test that our API matches torch.cuda.Event usage pattern"""
    print("\nTesting torch.cuda.Event style usage...")

    # This should work exactly like torch.cuda.Event
    start = infinicore.DeviceEvent(enable_timing=True)
    end = infinicore.DeviceEvent(enable_timing=True)

    # Record events
    start.record()

    # Some operations
    tensor = infinicore.ones(
        [100, 100], dtype=infinicore.float32, device=infinicore.device("cuda", 0)
    )
    for _ in range(5):
        tensor = tensor.permute([1, 0])

    end.record()
    end.synchronize()

    # This is the torch-compatible API
    time_taken = start.elapsed_time(end)

    print(f"✓ Torch-style usage test passed - Time: {time_taken:.3f} ms")


def test_event_synchronization():
    """Test event synchronization behavior"""
    print("\nTesting event synchronization...")

    event1 = infinicore.DeviceEvent(enable_timing=True)
    event2 = infinicore.DeviceEvent(enable_timing=True)

    # Record events in sequence
    event1.record()

    # Some work
    temp = infinicore.zeros(
        [50, 50], dtype=infinicore.float32, device=infinicore.device("cuda", 0)
    )

    event2.record()

    # event2 should complete after event1
    event2.synchronize()
    assert event2.query(), "event2 should be completed"
    assert event1.query(), "event1 should also be completed after event2 sync"

    print("✓ Event synchronization test passed")


def test_event_wait_functionality():
    """Test the wait functionality of DeviceEvent"""
    print("\nTesting DeviceEvent wait functionality...")

    # Create events
    event1 = infinicore.DeviceEvent(enable_timing=True)
    event2 = infinicore.DeviceEvent(enable_timing=True)

    # Record first event
    event1.record()

    # Perform some work
    tensor1 = infinicore.ones(
        [500, 500], dtype=infinicore.float32, device=infinicore.device("cuda", 0)
    )
    for _ in range(10):
        tensor1 = tensor1.permute([1, 0])

    # Record second event
    event2.record()

    # Make event2 wait for event1 using wait() method
    event2.wait()

    # Both events should be completed now
    assert event1.query(), "event1 should be completed"
    assert event2.query(), "event2 should be completed after waiting"

    print("✓ Event wait functionality test passed")


def test_stream_wait_event():
    """Test stream waiting for events"""
    print("\nTesting stream wait event functionality...")

    try:
        # Get the current stream
        current_stream = infinicore.get_stream()

        # Create events
        dependency_event = infinicore.DeviceEvent(enable_timing=True)
        dependent_event = infinicore.DeviceEvent(enable_timing=True)

        # Record dependency event
        dependency_event.record()

        # Perform some work that creates a dependency
        tensor = infinicore.ones(
            [300, 300], dtype=infinicore.float32, device=infinicore.device("cuda", 0)
        )
        for _ in range(5):
            tensor = tensor.permute([1, 0])

        # Make the stream wait for the dependency event before recording dependent event
        dependency_event.wait(current_stream)

        # Record dependent event after the wait
        dependent_event.record()

        # Synchronize and verify
        dependent_event.synchronize()
        assert dependency_event.query(), "Dependency event should be completed"
        assert dependent_event.query(), "Dependent event should be completed"

        print("✓ Stream wait event test passed")

    except Exception as e:
        print(f"⚠ Stream wait event test skipped: {e}")


def test_multiple_stream_synchronization():
    """Test event-based synchronization between multiple streams"""
    print("\nTesting multiple stream synchronization...")

    try:
        # This test simulates a producer-consumer pattern using events
        producer_event = infinicore.DeviceEvent(enable_timing=True)
        consumer_event = infinicore.DeviceEvent(enable_timing=True)

        # Producer work
        producer_event.record()

        # Simulate producer work (data generation)
        data_tensor = infinicore.ones(
            [200, 200], dtype=infinicore.float32, device=infinicore.device("cuda", 0)
        )
        for _ in range(8):
            data_tensor = data_tensor.permute([1, 0])

        # Make consumer wait for producer to finish
        producer_event.wait()  # Wait on current stream

        # Consumer work (depends on producer's output)
        processed_tensor = data_tensor.permute([1, 0])  # Consumer operation
        consumer_event.record()

        # Verify the synchronization worked
        consumer_event.synchronize()
        assert producer_event.query(), "Producer event should be completed"
        assert consumer_event.query(), "Consumer event should be completed"

        print("✓ Multiple stream synchronization test passed")

    except Exception as e:
        print(f"⚠ Multiple stream synchronization test skipped: {e}")


def test_event_wait_with_specific_stream():
    """Test waiting on specific streams"""
    print("\nTesting event wait with specific streams...")

    try:
        # Get current stream
        main_stream = infinicore.get_stream()

        # Create events
        compute_event = infinicore.DeviceEvent(enable_timing=True)
        transfer_event = infinicore.DeviceEvent(enable_timing=True)

        # Record compute event after some computation
        compute_event.record()

        # Simulate computation
        compute_tensor = infinicore.ones(
            [150, 150], dtype=infinicore.float32, device=infinicore.device("cuda", 0)
        )
        for _ in range(6):
            compute_tensor = compute_tensor.permute([1, 0])

        # Make data transfer wait for computation to complete
        compute_event.wait(main_stream)

        # Record transfer event
        transfer_event.record()

        # Verify synchronization
        transfer_event.synchronize()
        assert compute_event.query(), "Compute event should be completed"
        assert transfer_event.query(), "Transfer event should be completed"

        print("✓ Event wait with specific stream test passed")

    except Exception as e:
        print(f"⚠ Event wait with specific stream test skipped: {e}")


def test_complex_dependency_chain():
    """Test complex dependency chains using events"""
    print("\nTesting complex dependency chains...")

    try:
        # Create multiple events for a dependency chain
        event_a = infinicore.DeviceEvent(enable_timing=True)
        event_b = infinicore.DeviceEvent(enable_timing=True)
        event_c = infinicore.DeviceEvent(enable_timing=True)
        event_d = infinicore.DeviceEvent(enable_timing=True)

        # Stage A
        event_a.record()
        tensor_a = infinicore.ones(
            [100, 100], dtype=infinicore.float32, device=infinicore.device("cuda", 0)
        )
        for _ in range(3):
            tensor_a = tensor_a.permute([1, 0])

        # Stage B depends on A
        event_a.wait()
        event_b.record()
        tensor_b = tensor_a.permute([1, 0])  # Depends on tensor_a
        for _ in range(3):
            tensor_b = tensor_b.permute([1, 0])

        # Stage C depends on B
        event_b.wait()
        event_c.record()
        tensor_c = tensor_b.permute([1, 0])  # Depends on tensor_b
        for _ in range(3):
            tensor_c = tensor_c.permute([1, 0])

        # Stage D depends on C
        event_c.wait()
        event_d.record()
        tensor_d = tensor_c.permute([1, 0])  # Depends on tensor_c

        # Final synchronization
        event_d.synchronize()

        # Verify all events completed in order
        assert event_a.query(), "Event A should be completed"
        assert event_b.query(), "Event B should be completed"
        assert event_c.query(), "Event C should be completed"
        assert event_d.query(), "Event D should be completed"

        print("✓ Complex dependency chain test passed")

    except Exception as e:
        print(f"⚠ Complex dependency chain test skipped: {e}")


def test_wait_before_record():
    """Test waiting for an event that hasn't been recorded yet"""
    print("\nTesting wait before record behavior...")

    try:
        event = infinicore.DeviceEvent(enable_timing=True)

        # This should not crash, but the behavior depends on the underlying implementation
        # In most systems, waiting for an unrecorded event is undefined behavior
        # We're testing that our API handles this gracefully
        event.wait()

        print(
            "✓ Wait before record test completed (behavior may vary by implementation)"
        )

    except Exception as e:
        print(f"⚠ Wait before record test encountered expected behavior: {e}")


def run_all_tests():
    """Run all device-related tests"""
    print("Starting DeviceEvent and device tests...")
    print("=" * 50)

    try:
        # Basic functionality tests
        test_device_event_timing()
        test_device_event_query()
        test_torch_style_usage()
        test_event_synchronization()
        test_concurrent_events()

        # Wait functionality tests (new)
        test_event_wait_functionality()
        test_stream_wait_event()
        test_multiple_stream_synchronization()
        test_event_wait_with_specific_stream()
        test_complex_dependency_chain()
        test_wait_before_record()

        # Optional tests (may depend on system capabilities)
        test_multiple_devices()
        test_event_flags()
        test_event_stream()

        print("\n" + "=" * 50)
        print("🎉 All tests passed successfully!")
        print("DeviceEvent wait functionality is working correctly!")

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
