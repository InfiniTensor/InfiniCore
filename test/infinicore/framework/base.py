"""
Core base classes for operator testing framework.
Contains TestConfig, TestRunner, and BaseOperatorTest classes.
"""

import torch
import infinicore
import traceback
from abc import ABC, abstractmethod

from .test_case import TestCase, TestResult
from .datatypes import to_torch_dtype, to_infinicore_dtype
from .devices import InfiniDeviceNames, torch_device_map
from .tensor import TensorSpec, TensorInitializer
from .utils import (
    clone_torch_tensor,
    create_test_comparator,
    infinicore_tensor_from_torch,
)
from .benchmark import BenchmarkUtils


class TestConfig:
    """Test configuration"""

    def __init__(
        self,
        debug=False,
        bench=False,
        num_prerun=10,
        num_iterations=1000,
        verbose=False,
    ):
        self.debug = debug
        self.bench = bench
        self.num_prerun = num_prerun
        self.num_iterations = num_iterations
        self.verbose = verbose


class TestRunner:
    """Test runner"""

    def __init__(self, test_cases, test_config):
        self.test_cases = test_cases
        self.config = test_config
        self.failed_tests = []
        self.skipped_tests = []  # Track skipped tests (both operators not implemented)
        self.partial_tests = []  # Track partial tests (one operator not implemented)
        self.passed_tests = (
            []
        )  # Track passed tests (both operators implemented and passed)
        # Add benchmark timing statistics
        self.benchmark_times = {
            "torch_host_total": 0.0,
            "torch_device_total": 0.0,
            "infinicore_host_total": 0.0,
            "infinicore_device_total": 0.0,
            "per_test_case": {},  # Store timing per test case
        }
        # Store test results
        self.test_results = []

    def run_tests(self, devices, test_func, test_type="Test"):
        """
        Run tests on specified devices

        Args:
            devices: List of devices to test on
            test_func: Test function to execute
            test_type: Type of test for display purposes

        Returns:
            bool: True if no tests failed, False otherwise
        """
        for device in devices:
            print(f"\n{'='*60}")
            print(f"Testing {test_type} on {InfiniDeviceNames[device]}")
            print(f"{'='*60}")

            for test_case in self.test_cases:
                try:
                    print(f"{test_case}")

                    # Execute test and get TestResult object
                    test_result = test_func(device, test_case, self.config)
                    self.test_results.append(test_result)

                    # Handle different test statuses based on return_code
                    if test_result.return_code == 0:  # Success
                        self.passed_tests.append(
                            f"{test_case} - {InfiniDeviceNames[device]}"
                        )
                        print(f"\033[92m✓\033[0m Passed")
                    elif test_result.return_code == -1:
                        # Test failed - use the actual error message from test_result
                        fail_msg = f"{test_case} - {InfiniDeviceNames[device]} - {test_result.error_message}"
                        self.failed_tests.append(fail_msg)
                        print(f"\033[91m✗\033[0m {test_result.error_message}")
                    elif test_result.return_code == -2:  # Skipped
                        # Both operators not implemented - use actual error message
                        skip_msg = f"{test_case} - {InfiniDeviceNames[device]} - {test_result.error_message}"
                        self.skipped_tests.append(skip_msg)
                        print(f"\033[93m⚠\033[0m {test_result.error_message}")
                    elif test_result.return_code == -3:  # Partial
                        # One operator not implemented - use actual error message
                        partial_msg = f"{test_case} - {InfiniDeviceNames[device]} - {test_result.error_message}"
                        self.partial_tests.append(partial_msg)
                        print(f"\033[93m⚠\033[0m {test_result.error_message}")

                    if self.config.verbose and test_result.return_code != 0:
                        return False

                except Exception as e:
                    error_msg = f"Error: {e}"
                    print(f"\033[91m✗\033[0m {error_msg}")
                    self.failed_tests.append(error_msg)

                    # Create a failed TestResult
                    failed_result = TestResult(
                        success=False,
                        return_code=-1,
                        error_message=str(e),
                        test_case=test_case,
                        device=device,
                    )
                    self.test_results.append(failed_result)
                    # In verbose mode, print full traceback and stop execution
                    if self.config.verbose:
                        traceback.print_exc()
                        return False  # Stop test execution immediately

                    if self.config.debug:
                        raise

        return (
            len(self.failed_tests) == 0
            and len(self.skipped_tests) == 0
            and len(self.partial_tests) == 0
        )

    def print_summary(self):
        """
        Print test execution summary

        Returns:
            bool: True if no tests failed, False otherwise
        """
        total_tests = len(self.test_cases)
        passed_count = len(self.passed_tests)
        skipped_count = len(self.skipped_tests)
        partial_count = len(self.partial_tests)
        failed_count = len(self.failed_tests)

        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print(f"Total tests: {total_tests}")
        print(f"\033[92mPassed: {passed_count}\033[0m")

        result = True
        # Display failed tests
        if self.failed_tests:
            print(f"\033[91mFailed: {failed_count}\033[0m")

            # Return False only if there are actual test failures
            result = False
        else:
            # Calculate success rate based on actual executed tests
            executed_tests = passed_count + partial_count + failed_count
            if executed_tests > 0:
                success_rate = passed_count / executed_tests * 100
                print(f"Success rate: {success_rate:.1f}%")

            # If there are skipped or partial tests, show appropriate message
            if self.skipped_tests or self.partial_tests:
                print(
                    f"\n\033[93mTests completed with some implementations missing\033[0m"
                )
            else:
                print(f"\n\033[92mAll tests passed!\033[0m")

        # Print benchmark summary if benchmarking was enabled
        if self.config.bench and (
            self.benchmark_times["torch_host_total"] > 0
            or self.benchmark_times["torch_device_total"] > 0
            or self.benchmark_times["infinicore_host_total"] > 0
            or self.benchmark_times["infinicore_device_total"] > 0
        ):
            self._print_benchmark_summary()

        print(f"{'='*60}")
        return result

    def _print_benchmark_summary(self):
        """Print benchmark timing summary"""
        print(f"{'-'*60}")
        print("BENCHMARK SUMMARY")

        torch_host_total = self.benchmark_times["torch_host_total"]
        torch_device_total = self.benchmark_times["torch_device_total"]
        infinicore_host_total = self.benchmark_times["infinicore_host_total"]
        infinicore_device_total = self.benchmark_times["infinicore_device_total"]

        if torch_host_total > 0:
            print(f"PyTorch Host Total Time: {torch_host_total:.3f} ms")
        if torch_device_total > 0:
            print(f"PyTorch Device Total Time: {torch_device_total:.3f} ms")
        if infinicore_host_total > 0:
            print(f"InfiniCore Host Total Time: {infinicore_host_total:.3f} ms")
        if infinicore_device_total > 0:
            print(f"InfiniCore Device Total Time: {infinicore_device_total:.3f} ms")

        # Calculate speedups
        if torch_host_total > 0 and infinicore_host_total > 0:
            host_speedup = torch_host_total / infinicore_host_total
            print(f"Host Speedup (PyTorch/InfiniCore): {host_speedup:.2f}x")

        if torch_device_total > 0 and infinicore_device_total > 0:
            device_speedup = torch_device_total / infinicore_device_total
            print(f"Device Speedup (PyTorch/InfiniCore): {device_speedup:.2f}x")

    def get_test_results(self):
        """Get all test results"""
        return self.test_results


class BaseOperatorTest(ABC):
    """Base operator test"""

    def __init__(self, operator_name):
        self.operator_name = operator_name
        self.test_cases = self.get_test_cases()

    @abstractmethod
    def get_test_cases(self):
        """Return list of TestCase objects with complete configuration"""
        pass

    def torch_operator(self, *args, **kwargs):
        """PyTorch operator function"""
        raise NotImplementedError("torch_operator not implemented")

    def infinicore_operator(self, *args, **kwargs):
        """InfiniCore operator function"""
        raise NotImplementedError("infinicore_operator not implemented")

    def _create_tensor_from_spec(self, spec, device):
        """Helper method to create tensor from TensorSpec"""
        if isinstance(spec, TensorSpec):
            if spec.is_scalar:
                return spec.value
            else:
                return spec.create_torch_tensor(device)
        return spec

    def prepare_pytorch_inputs_and_kwargs(self, test_case, device):
        """Prepare inputs and kwargs, replacing TensorSpec objects with actual tensors
        Supports tuple inputs for operators like torch.cat and TensorSpec in kwargs
        """
        inputs = []
        kwargs = test_case.kwargs.copy()

        # Prepare input tensors - support both single TensorSpecs and tuples of TensorSpecs
        for input_spec in test_case.inputs:
            if isinstance(input_spec, (list, tuple)):
                # Handle tuple of multiple TensorSpecs (e.g., for torch.cat)
                tuple_tensors = []
                for item in input_spec:
                    if isinstance(item, (list, tuple)):
                        # Handle nested tuples
                        nested_tensors = []
                        for nested_item in item:
                            nested_tensors.append(
                                self._create_tensor_from_spec(nested_item, device)
                            )
                        tuple_tensors.append(tuple(nested_tensors))
                    else:
                        tuple_tensors.append(
                            self._create_tensor_from_spec(item, device)
                        )
                inputs.append(tuple(tuple_tensors))
            else:
                inputs.append(self._create_tensor_from_spec(input_spec, device))

        # Prepare output tensors based on output_count
        if test_case.output_count == 1:
            # Single output case
            if test_case.output_spec is not None:
                output_tensor = test_case.output_spec.create_torch_tensor(device)
                kwargs["out"] = output_tensor
        else:
            # Multiple outputs case
            if test_case.output_specs is not None:
                # Create output tuple for in-place multiple outputs
                output_tensors = tuple(
                    spec.create_torch_tensor(device) for spec in test_case.output_specs
                )
                kwargs["out"] = output_tensors

        # Handle integer indices for in-place operations
        if "out" in kwargs and isinstance(kwargs["out"], int):
            input_idx = kwargs["out"]
            if 0 <= input_idx < len(inputs) and isinstance(
                inputs[input_idx], torch.Tensor
            ):
                kwargs["out"] = inputs[input_idx]
            else:
                raise ValueError(
                    f"Invalid input index for in-place operation: {input_idx}"
                )

        for key, value in list(kwargs.items()):
            if isinstance(value, TensorSpec):
                # Replace TensorSpec with actual tensor
                kwargs[key] = self._create_tensor_from_spec(value, device)

        return inputs, kwargs

    def prepare_infinicore_list(self, input_sequence, clone=False):
        cloned_tensors = []
        infini_list = []
        for item in input_sequence:
            if isinstance(item, torch.Tensor):
                if clone:
                    cloned_item = clone_torch_tensor(item)
                    infini_item = infinicore_tensor_from_torch(cloned_item)
                    cloned_tensors.append(cloned_item)
                else:
                    infini_item = infinicore_tensor_from_torch(item)
            else:
                infini_item = item
            infini_list.append(infini_item)
        return infini_list, cloned_tensors

    def prepare_infinicore_inputs_and_kwargs(self, inputs, kwargs, comparison_target):
        cloned_tensors = []
        infini_inputs = []

        # Prepare infinicore inputs - only clone if needed for comparison
        for i, inp in enumerate(inputs):
            if isinstance(inp, torch.Tensor):
                # Clone only if this input will be used for comparison
                if comparison_target == i:
                    cloned_inp = clone_torch_tensor(inp)
                    infini_tensor = infinicore_tensor_from_torch(cloned_inp)
                    cloned_tensors.append(cloned_inp)
                else:
                    # For non-comparison inputs, we can use the original (but still need to convert)
                    infini_tensor = infinicore_tensor_from_torch(inp)
                infini_inputs.append(infini_tensor)
            elif isinstance(inp, (tuple, list)):
                infini_list, cloned_list = self.prepare_infinicore_list(
                    inp, comparison_target == i
                )
                infini_inputs.append(infini_list)
                cloned_tensors.append(cloned_list)
            else:
                infini_inputs.append(inp)

        # Prepare infinicore kwargs
        infini_kwargs = {}
        for key, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                # Check if this tensor is used for output comparison
                if key == "out" and comparison_target == "out":
                    cloned_value = clone_torch_tensor(value)
                    infini_kwargs[key] = infinicore_tensor_from_torch(cloned_value)
                    cloned_tensors.append(cloned_value)
                elif key == "out" and isinstance(comparison_target, int):
                    infini_kwargs[key] = infini_inputs[comparison_target]
                else:
                    infini_kwargs[key] = infinicore_tensor_from_torch(value)
            elif isinstance(value, (tuple, list)):
                infini_list, cloned_list = self.prepare_infinicore_list(
                    value, key == "out"
                )
                cloned_tensors.append(cloned_list)
                infini_kwargs[key] = infini_list
            else:
                infini_kwargs[key] = value

        return infini_inputs, infini_kwargs, cloned_tensors

    def run_test(self, device, test_case, config):
        """
        Unified test execution flow

        Args:
            device: Device to test on
            test_case: Test case configuration
            config: Test configuration

        Returns:
            TestResult: Test result object containing status and timing information
        """
        device_str = torch_device_map[device]

        # Initialize test result
        test_result = TestResult(
            success=False,
            return_code=-1,  # Default to failure
            test_case=test_case,
            device=device,
        )

        # Prepare inputs and kwargs with actual tensors
        inputs, kwargs = self.prepare_pytorch_inputs_and_kwargs(test_case, device)

        # Determine comparison target
        comparison_target = test_case.comparison_target

        # Create infinicore inputs (cloned to avoid in-place modifications affecting reference)
        infini_inputs, infini_kwargs, cloned_tensors = (
            self.prepare_infinicore_inputs_and_kwargs(inputs, kwargs, comparison_target)
        )

        # Check operator implementations
        torch_implemented = True
        infini_implemented = True
        torch_error_msg = ""
        infini_error_msg = ""

        try:
            torch_result = self.torch_operator(*inputs, **kwargs)
            if torch_result is None:
                torch_implemented = False
        except NotImplementedError as e:
            if config.verbose:
                traceback.print_exc()
            torch_implemented = False
            torch_result = None
            torch_error_msg = str(e)

        try:
            infini_result = self.infinicore_operator(*infini_inputs, **infini_kwargs)
            if infini_result is None:
                infini_implemented = False
        except NotImplementedError as e:
            if config.verbose:
                traceback.print_exc()
            infini_implemented = False
            infini_result = None
            infini_error_msg = str(e)

        if not torch_error_msg:
            torch_error_msg = "unimplemented test function"

        if not infini_error_msg:
            infini_error_msg = "unimplemented test function"

        # Skip if neither operator is implemented
        if not torch_implemented and not infini_implemented:
            test_result.return_code = -2  # Skipped
            # Combine both error messages
            test_result.error_message = f"Both operators failed: PyTorch - {torch_error_msg}; InfiniCore - {infini_error_msg}"
            return test_result

        # Single operator execution without comparison
        if not torch_implemented or not infini_implemented:
            test_result.return_code = -3  # Partial
            # Determine which operator is missing and create appropriate message with actual error
            if not torch_implemented:
                test_result.error_message = (
                    f"PyTorch operator failed: {torch_error_msg}"
                )
            else:
                test_result.error_message = (
                    f"InfiniCore operator failed: {infini_error_msg}"
                )

            # Run benchmarking for partial tests if enabled
            if config.bench:
                torch_host, torch_device, infini_host, infini_device = (
                    BenchmarkUtils.run_benchmarking(
                        config,
                        device_str,
                        torch_implemented,
                        infini_implemented,
                        self.torch_operator,
                        self.infinicore_operator,
                        inputs,
                        kwargs,
                        infini_inputs,
                        infini_kwargs,
                        test_case.output_count,
                        comparison_target,
                        bench_mode=config.bench,
                    )
                )
                test_result.torch_host_time = torch_host
                test_result.torch_device_time = torch_device
                test_result.infini_host_time = infini_host
                test_result.infini_device_time = infini_device
            return test_result
        # ==========================================================================
        # MULTIPLE OUTPUTS COMPARISON LOGIC
        # ==========================================================================
        if test_case.output_count > 1:
            # Handle multiple outputs comparison

            # Determine what to compare based on comparison_target
            if comparison_target is None:
                # Compare return values (out-of-place multiple outputs)
                torch_comparison = torch_result
                infini_comparison = infini_result
            elif comparison_target == "out":
                # Compare output tuple from kwargs (explicit multiple outputs)
                torch_comparison = kwargs.get("out")
                infini_comparison = infini_kwargs.get("out")
            else:
                raise ValueError(
                    f"Invalid comparison target for multiple outputs: {comparison_target}"
                )

            # Validate that we have multiple outputs to compare
            if not isinstance(torch_comparison, (tuple, list)) or not isinstance(
                infini_comparison, (tuple, list)
            ):
                raise ValueError(
                    f"Multiple outputs expected but got single result: "
                    f"torch={type(torch_comparison)}, infinicore={type(infini_comparison)}"
                )

            if len(torch_comparison) != len(infini_comparison):
                raise ValueError(
                    f"Output count mismatch: torch={len(torch_comparison)}, infinicore={len(infini_comparison)}"
                )

            if len(torch_comparison) != test_case.output_count:
                raise ValueError(
                    f"Output count mismatch: expected {test_case.output_count}, got {len(torch_comparison)}"
                )

            # Compare each output pair individually
            all_valid = True
            for i, (torch_out, infini_out) in enumerate(
                zip(torch_comparison, infini_comparison)
            ):
                atol = test_case.tolerance.get("atol", 1e-5)
                rtol = test_case.tolerance.get("rtol", 1e-3)

                compare_fn = create_test_comparator(
                    config, atol, rtol, f"{test_case.description} - output_{i}"
                )

                is_valid = compare_fn(infini_out, torch_out)
                if not is_valid:
                    print(f"❌ Output {i} comparison failed")
                    all_valid = False
                else:
                    print(f"✅ Output {i} comparison passed")

            if not all_valid:
                raise AssertionError(
                    f"Multiple outputs comparison failed for {test_case}"
                )

        # ==========================================================================
        # SINGLE OUTPUT COMPARISON LOGIC
        # ==========================================================================
        else:
            # Determine comparison targets for single output
            if comparison_target is None:
                # Compare return values (out-of-place)
                torch_comparison = torch_result
                infini_comparison = infini_result
            elif comparison_target == "out":
                # Compare output tensor from kwargs (explicit output)
                torch_comparison = kwargs.get("out")
                infini_comparison = cloned_tensors[0]
            elif isinstance(comparison_target, int):
                # Compare specific input tensor (in-place operation on input)
                if 0 <= comparison_target < len(inputs):
                    torch_comparison = inputs[comparison_target]
                    infini_comparison = cloned_tensors[0]
                else:
                    raise ValueError(
                        f"Invalid comparison target index: {comparison_target}"
                    )
            else:
                raise ValueError(f"Invalid comparison target: {comparison_target}")

            # Validate comparison targets
            if torch_comparison is None or infini_comparison is None:
                raise ValueError("Comparison targets cannot be None")

            # Perform comparison
            atol = test_case.tolerance.get("atol", 1e-5)
            rtol = test_case.tolerance.get("rtol", 1e-3)

            compare_fn = create_test_comparator(
                config, atol, rtol, test_case.description
            )

            is_valid = compare_fn(infini_comparison, torch_comparison)
            if not is_valid:
                raise AssertionError(f"Result comparison failed.")

        # ==========================================================================
        # UNIFIED BENCHMARKING LOGIC
        # ==========================================================================
        if config.bench:
            torch_host, torch_device, infini_host, infini_device = (
                BenchmarkUtils.run_benchmarking(
                    config,
                    device_str,
                    True,
                    True,
                    self.torch_operator,
                    self.infinicore_operator,
                    inputs,
                    kwargs,
                    infini_inputs,
                    infini_kwargs,
                    test_case.output_count,
                    comparison_target,
                    bench_mode=config.bench,
                )
            )
            test_result.torch_host_time = torch_host
            test_result.torch_device_time = torch_device
            test_result.infini_host_time = infini_host
            test_result.infini_device_time = infini_device

            # Store timing information in the test runner
            if hasattr(config, "_test_runner") and config._test_runner:
                # Accumulate total times
                config._test_runner.benchmark_times["torch_host_total"] += torch_host
                config._test_runner.benchmark_times[
                    "torch_device_total"
                ] += torch_device
                config._test_runner.benchmark_times[
                    "infinicore_host_total"
                ] += infini_host
                config._test_runner.benchmark_times[
                    "infinicore_device_total"
                ] += infini_device

        # Test passed successfully
        test_result.success = True
        test_result.return_code = 0
        return test_result
