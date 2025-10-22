#!/usr/bin/env python3
"""
Example script showing how to use InfiniCore memory statistics
to monitor memory usage during tensor operations.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import infinicore
    print("✓ Successfully imported infinicore")
except ImportError as e:
    print(f"✗ Failed to import infinicore: {e}")
    print("Make sure to build the project first with: xmake build _infinicore")
    sys.exit(1)

def get_memory_summary():
    """Get a summary of current memory usage."""
    try:
        device_stats = infinicore.get_device_memory_stats()
        return {
            'allocations': device_stats.allocation[0].current,
            'allocated_bytes': device_stats.allocated_bytes[0].current,
            'active_blocks': device_stats.active[0].current,
            'device_allocations': device_stats.num_device_alloc,
            'device_deallocations': device_stats.num_device_free
        }
    except Exception as e:
        print(f"Warning: Could not get memory stats: {e}")
        return None

def print_memory_summary(title, stats):
    """Print a concise memory summary."""
    if stats is None:
        print(f"{title}: Unable to get memory statistics")
        return

    print(f"{title}:")
    print(f"  Allocations: {stats['allocations']}")
    print(f"  Allocated bytes: {stats['allocated_bytes']:,} bytes ({stats['allocated_bytes'] / 1024 / 1024:.2f} MB)")
    print(f"  Active blocks: {stats['active_blocks']}")
    print(f"  Device alloc/dealloc: {stats['device_allocations']}/{stats['device_deallocations']}")

def monitor_memory_usage():
    """Monitor memory usage during tensor operations."""
    print("=== InfiniCore Memory Usage Monitor ===\n")

    # Initial memory state
    initial_stats = get_memory_summary()
    print_memory_summary("Initial Memory State", initial_stats)

    try:
        # Create some tensors to demonstrate memory usage
        print("\n1. Creating tensors...")

        # Create a large tensor
        print("   Creating 1000x1000 float32 tensor...")
        tensor1 = infinicore.empty((1000, 1000), dtype=infinicore.float32)
        stats_after_tensor1 = get_memory_summary()
        print_memory_summary("After creating tensor1", stats_after_tensor1)

        # Create another tensor
        print("\n   Creating 500x500 float32 tensor...")
        tensor2 = infinicore.empty((500, 500), dtype=infinicore.float32)
        stats_after_tensor2 = get_memory_summary()
        print_memory_summary("After creating tensor2", stats_after_tensor2)

        # Create a third tensor
        print("\n   Creating 2000x2000 float32 tensor...")
        tensor3 = infinicore.empty((2000, 2000), dtype=infinicore.float32)
        stats_after_tensor3 = get_memory_summary()
        print_memory_summary("After creating tensor3", stats_after_tensor3)

        # Delete some tensors
        print("\n2. Deleting tensors...")
        del tensor1
        stats_after_del1 = get_memory_summary()
        print_memory_summary("After deleting tensor1", stats_after_del1)

        del tensor2
        stats_after_del2 = get_memory_summary()
        print_memory_summary("After deleting tensor2", stats_after_del2)

        # Final cleanup
        print("\n3. Final cleanup...")
        del tensor3
        final_stats = get_memory_summary()
        print_memory_summary("Final Memory State", final_stats)

        # Show memory difference
        if initial_stats and final_stats:
            print(f"\nMemory Usage Summary:")
            print(f"  Net allocations: {final_stats['allocations'] - initial_stats['allocations']}")
            print(f"  Net allocated bytes: {final_stats['allocated_bytes'] - initial_stats['allocated_bytes']:,} bytes")
            print(f"  Net active blocks: {final_stats['active_blocks'] - initial_stats['active_blocks']}")

        print("\n✓ Memory monitoring completed successfully!")

    except Exception as e:
        print(f"✗ Error during memory monitoring: {e}")
        import traceback
        traceback.print_exc()

def demonstrate_stat_types():
    """Demonstrate different stat types and their usage."""
    print("\n=== Stat Types Demonstration ===\n")

    try:
        # Get device stats
        device_stats = infinicore.get_device_memory_stats()

        print("StatType.AGGREGATE statistics:")
        print(f"  Allocation count: {device_stats.allocation[0].current}")
        print(f"  Allocation peak: {device_stats.allocation[0].peak}")
        print(f"  Allocation total: {device_stats.allocation[0].allocated}")
        print(f"  Allocation freed: {device_stats.allocation[0].freed}")

        print(f"\nStatType.SMALL_POOL statistics:")
        print(f"  Allocation count: {device_stats.allocation[1].current}")
        print(f"  Allocation peak: {device_stats.allocation[1].peak}")

        print(f"\nStatType.LARGE_POOL statistics:")
        print(f"  Allocation count: {device_stats.allocation[2].current}")
        print(f"  Allocation peak: {device_stats.allocation[2].peak}")

        print("\n✓ Stat types demonstration completed!")

    except Exception as e:
        print(f"✗ Error during stat types demonstration: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    monitor_memory_usage()
    demonstrate_stat_types()
