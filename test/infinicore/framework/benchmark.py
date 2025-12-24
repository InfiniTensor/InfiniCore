"""
Benchmarking utilities for the InfiniCore testing framework
"""

import time
import torch
import infinicore
from .utils import synchronize_device


class BenchmarkUtils:
    """Utility class for benchmarking operations"""

    @staticmethod
    def profile_operation(
        desc,
        func,
        torch_device,
        num_prerun,
        num_iterations,
        host_time=True,
        device_time=True,
        total=False,
    ):
        """
        Performance profiling workflow with both host and device timing

        Args:
            desc: Operation description for display
            func: Function to profile
            torch_device: Torch device string
            num_prerun: Number of warm-up runs
            num_iterations: Number of iterations for timing
            host_time: Whether to measure host (CPU) time
            device_time: Whether to measure device time
            total: Whether to return total time instead of per-iteration time

        Returns:
            tuple: (host_time, device_time) timing results
        """
        # Timed execution
        host_elapsed = 0.0
        device_elapsed = 0.0

        if host_time:
            # Warm-up runs
            for _ in range(num_prerun):
                func()
            host_elapsed = BenchmarkUtils.timed_op_host(
                func, num_iterations, torch_device
            )

        if device_time:
            # Warm-up runs
            for _ in range(num_prerun):
                func()
            device_elapsed = BenchmarkUtils.timed_op_device(
                func, num_iterations, torch_device
            )

        # Print results
        if host_time and device_time:
            print(
                f"    {desc} time - Host: {host_elapsed / num_iterations :6f} ms, "
                f"Device: {device_elapsed / num_iterations :6f} ms"
            )
        elif host_time:
            print(f"    {desc} time - Host: {host_elapsed / num_iterations :6f} ms")
        elif device_time:
            print(f"    {desc} time - Device: {device_elapsed / num_iterations :6f} ms")

        if total:
            return host_elapsed, device_elapsed
        else:
            return host_elapsed / num_iterations, device_elapsed / num_iterations

    @staticmethod
    def timed_op_host(func, num_iterations, device):
        """
        Execute function multiple times and measure total host execution time

        Args:
            func: Function to execute
            num_iterations: Number of iterations
            device: Torch device string for synchronization

        Returns:
            float: Total host execution time in seconds
        """
        synchronize_device(device)
        start = time.time()
        for _ in range(num_iterations):
            func()
        synchronize_device(device)
        return (time.time() - start) * 1000.0

    @staticmethod
    def timed_op_device(func, num_iterations, device):
        """
        Execute function multiple times and measure device execution time using DeviceEvent pairs

        Args:
            func: Function to execute
            num_iterations: Number of iterations
            device: Torch device string for synchronization

        Returns:
            float: Total device execution time in milliseconds
        """
        # Only use DeviceEvent for GPU devices
        if device in ["cpu"]:
            return 0.0

        def _clear_cache():
            pass

        if infinicore.use_ntops:
            import triton

            cache = triton.runtime.driver.active.get_empty_cache_for_benchmark()

            def _clear_cache():
                triton.runtime.driver.active.clear_cache(cache)

        # Create pairs of DeviceEvents for each iteration
        start_events = [
            infinicore.DeviceEvent(enable_timing=True) for _ in range(num_iterations)
        ]
        end_events = [
            infinicore.DeviceEvent(enable_timing=True) for _ in range(num_iterations)
        ]

        # Execute the function multiple times with individual timing
        for i in range(num_iterations):
            _clear_cache()
            start_events[i].record()
            func()
            end_events[i].record()

        # Synchronize all end events
        for event in end_events:
            event.synchronize()

        # Calculate total elapsed time by summing individual iteration times
        total_device_time = 0.0
        for i in range(num_iterations):
            total_device_time += start_events[i].elapsed_time(end_events[i])

        return total_device_time

    @staticmethod
    def run_benchmarking(
        config,
        device_str,
        torch_implemented,
        infini_implemented,
        torch_operator,
        infini_operator,
        inputs,
        kwargs,
        infini_inputs,
        infini_kwargs,
        output_count,
        comparison_target,
        bench_mode="both",
    ):
        """
        Unified benchmarking logic with timing accumulation

        Args:
            config: Test configuration
            device_str: Torch device string
            torch_implemented: Whether PyTorch operator is implemented
            infini_implemented: Whether InfiniCore operator is implemented
            torch_operator: PyTorch operator function
            infini_operator: InfiniCore operator function
            inputs: PyTorch operator inputs
            kwargs: PyTorch operator keyword arguments
            infini_inputs: InfiniCore operator inputs
            infini_kwargs: InfiniCore operator keyword arguments
            output_count: Number of outputs
            comparison_target: Comparison target specification
            bench_mode: Benchmark mode - "host", "device", or "both"

        Returns:
            tuple: (torch_host_time, torch_device_time, infini_host_time, infini_device_time)
        """
        # Determine what to time based on bench_mode
        host_time = bench_mode in ["host", "both"]
        device_time = bench_mode in ["device", "both"]

        # Initialize timing variables
        torch_host_time = 0.0
        torch_device_time = 0.0
        infini_host_time = 0.0
        infini_device_time = 0.0

        if torch_implemented:
            if output_count > 1:
                # For multiple outputs, just call the operator
                def torch_op():
                    return torch_operator(*inputs, **kwargs)

            else:
                if comparison_target is None:
                    # Out-of-place benchmarking
                    def torch_op():
                        return torch_operator(*inputs, **kwargs)

                else:
                    # In-place benchmarking
                    def torch_op():
                        torch_operator(*inputs, **kwargs)
                        return (
                            kwargs.get("out")
                            if "out" in kwargs
                            else inputs[comparison_target]
                        )

            torch_host, torch_device = BenchmarkUtils.profile_operation(
                "PyTorch   ",
                torch_op,
                device_str,
                config.num_prerun,
                config.num_iterations,
                host_time=host_time,
                device_time=device_time,
                total=True,
            )
            torch_host_time = torch_host
            torch_device_time = torch_device

        if infini_implemented:
            if comparison_target is None:
                # Out-of-place benchmarking
                def infini_op():
                    return infini_operator(*infini_inputs, **infini_kwargs)

            else:
                # In-place benchmarking
                def infini_op():
                    infini_operator(*infini_inputs, **infini_kwargs)
                    return (
                        infini_kwargs.get("out")
                        if "out" in infini_kwargs
                        else infini_inputs[comparison_target]
                    )

            infini_host, infini_device = BenchmarkUtils.profile_operation(
                "InfiniCore",
                infini_op,
                device_str,
                config.num_prerun,
                config.num_iterations,
                host_time=host_time,
                device_time=device_time,
                total=True,
            )
            infini_host_time = infini_host
            infini_device_time = infini_device

        return torch_host_time, torch_device_time, infini_host_time, infini_device_time


class BenchmarkResult:
    """Container for benchmark results"""

    def __init__(self):
        self.torch_host_total = 0.0
        self.torch_device_total = 0.0
        self.infinicore_host_total = 0.0
        self.infinicore_device_total = 0.0
        self.per_test_case = {}

    def add_timing(
        self, test_case_name, torch_host, torch_device, infini_host, infini_device
    ):
        """Add timing for a specific test case"""
        self.per_test_case[test_case_name] = {
            "torch_host_time": torch_host,
            "torch_device_time": torch_device,
            "infini_host_time": infini_host,
            "infini_device_time": infini_device,
        }
        self.torch_host_total += torch_host
        self.torch_device_total += torch_device
        self.infinicore_host_total += infini_host
        self.infinicore_device_total += infini_device

    def get_host_speedup(self):
        """Calculate host speedup ratio"""
        if self.infinicore_host_total > 0:
            return self.torch_host_total / self.infinicore_host_total
        return float("inf")

    def get_device_speedup(self):
        """Calculate device speedup ratio"""
        if self.infinicore_device_total > 0:
            return self.torch_device_total / self.infinicore_device_total
        return float("inf")
