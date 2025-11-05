import torch
import infinicore
import copy

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Union, Callable, Optional

from .datatypes import to_torch_dtype, to_infinicore_dtype
from .devices import InfiniDeviceNames, torch_device_map
from .tensor import TensorSpec, TensorInitializer
from .utils import (
    create_test_comparator,
    infinicore_tensor_from_torch,
    profile_operation,
    rearrange_tensor,
    synchronize_device,
)


class TestCase:
    """Test case"""

    OUT_OF_PLACE = "out_of_place"
    IN_PLACE = "in_place"
    BOTH = "both"

    # New in-place modes for specific operators
    IN_PLACE_0 = "in_place_0"  # e.g., elu(x, inplace=True) where x is input_0
    IN_PLACE_1 = "in_place_1"  # e.g., add(a, b, out=b) where b is input_1
    IN_PLACE_MULTI = "in_place_multi"  # Multiple in-place options

    def __init__(
        self, operation_mode, inputs, output=None, inplace_target=None, **kwargs
    ):
        if operation_mode not in [
            self.IN_PLACE,
            self.OUT_OF_PLACE,
            self.BOTH,
            self.IN_PLACE_0,
            self.IN_PLACE_1,
            self.IN_PLACE_MULTI,
        ]:
            raise ValueError(f"Invalid operation_mode: {operation_mode}")

        # For input-specific in-place modes, set default inplace_target if not provided
        if operation_mode == self.IN_PLACE_0 and inplace_target is None:
            inplace_target = 0  # Default to input 0
        elif operation_mode == self.IN_PLACE_1 and inplace_target is None:
            inplace_target = 1  # Default to input 1
        elif operation_mode == self.IN_PLACE_MULTI and inplace_target is None:
            inplace_target = "both"  # Default to testing both

        self.operation_mode = operation_mode
        self.inputs = []
        self.inplace_target = inplace_target  # Which input to use as in-place target

        for inp in inputs:
            if isinstance(inp, (list, tuple)):
                self.inputs.append(TensorSpec.from_tensor(inp))
            elif isinstance(inp, TensorSpec):
                self.inputs.append(inp)
            else:
                self.inputs.append(inp)

        if isinstance(output, (list, tuple)):
            self.output = TensorSpec.from_tensor(output)
        else:
            self.output = output

        self.kwargs = kwargs
        self.description = kwargs.pop("description", "")

    def __str__(self):
        mode_str = self.operation_mode.upper()
        input_strs = []
        for inp in self.inputs:
            if hasattr(inp, "is_scalar") and inp.is_scalar:
                dtype_str = f", dtype={inp.dtype}" if inp.dtype else ""
                input_strs.append(f"scalar({inp.value}{dtype_str})")
            elif hasattr(inp, "shape"):
                dtype_str = f", dtype={inp.dtype}" if inp.dtype else ""
                init_str = (
                    f", init={inp.init_mode}"
                    if inp.init_mode != TensorInitializer.RANDOM
                    else ""
                )
                # Show shape and strides for non-contiguous tensors
                if (
                    hasattr(inp, "is_contiguous")
                    and not inp.is_contiguous
                    and inp.strides
                ):
                    strides_str = f", strides={inp.strides}"
                    input_strs.append(
                        f"tensor{inp.shape}{strides_str}{dtype_str}{init_str}"
                    )
                else:
                    input_strs.append(f"tensor{inp.shape}{dtype_str}{init_str}")
            else:
                input_strs.append(str(inp))

        base_str = f"TestCase(mode={mode_str}, inputs=[{'; '.join(input_strs)}]"
        if self.output:
            dtype_str = f", dtype={self.output.dtype}" if self.output.dtype else ""
            init_str = (
                f", init={self.output.init_mode}"
                if self.output.init_mode != TensorInitializer.RANDOM
                else ""
            )
            # Show shape and strides for non-contiguous output tensors
            if (
                hasattr(self.output, "is_contiguous")
                and not self.output.is_contiguous
                and self.output.strides
            ):
                strides_str = f", strides={self.output.strides}"
                base_str += f", output=tensor{self.output.shape}{strides_str}{dtype_str}{init_str}"
            else:
                base_str += f", output=tensor{self.output.shape}{dtype_str}{init_str}"
        if self.kwargs:
            base_str += f", kwargs={self.kwargs}"
        if self.description:
            base_str += f", desc='{self.description}'"
        # Add inplace_target to string representation if present
        if self.inplace_target is not None:
            base_str += f", inplace_target={self.inplace_target}"
        base_str += ")"
        return base_str


class TestConfig:
    """Test configuration"""

    def __init__(
        self,
        tensor_dtypes,
        tolerance_map,
        debug=False,
        bench=False,
        num_prerun=10,
        num_iterations=1000,
        dtype_combinations=None,
    ):
        self.tensor_dtypes = tensor_dtypes
        self.tolerance_map = tolerance_map
        self.debug = debug
        self.bench = bench
        self.num_prerun = num_prerun
        self.num_iterations = num_iterations
        self.dtype_combinations = dtype_combinations


class TestRunner:
    """Test runner"""

    def __init__(self, test_cases, test_config):
        self.test_cases = test_cases
        self.config = test_config
        self.failed_tests = []

    def run_tests(self, devices, test_func, test_type="Test"):
        for device in devices:
            print(f"\n{'='*60}")
            print(f"Testing {test_type} on {InfiniDeviceNames[device]}")
            print(f"{'='*60}")

            tensor_dtypes = self._filter_tensor_dtypes_by_device(
                device, self.config.tensor_dtypes
            )

            for test_case in self.test_cases:
                if self.config.dtype_combinations:
                    for dtype_combo in self.config.dtype_combinations:
                        try:
                            # Print test case info first
                            combo_str = self._format_dtype_combo(dtype_combo)
                            print(f"{test_case} with {combo_str}")

                            test_func(device, test_case, dtype_combo, self.config)
                            print(f"\033[92m✓\033[0m Passed")
                        except Exception as e:
                            combo_str = self._format_dtype_combo(dtype_combo)
                            error_msg = f"Error: {e}"
                            print(f"\033[91m✗\033[0m {error_msg}")
                            self.failed_tests.append(error_msg)
                            if self.config.debug:
                                raise
                else:
                    for dtype in tensor_dtypes:
                        try:
                            # Print test case info first
                            print(f"{test_case} with {dtype}")

                            test_func(device, test_case, dtype, self.config)
                            print(f"\033[92m✓\033[0m Passed")
                        except Exception as e:
                            error_msg = f"Error: {e}"
                            print(f"\033[91m✗\033[0m {error_msg}")
                            self.failed_tests.append(error_msg)
                            if self.config.debug:
                                raise

        return len(self.failed_tests) == 0

    def _format_dtype_combo(self, dtype_combo):
        if isinstance(dtype_combo, dict):
            return f"dtypes({dtype_combo})"
        elif isinstance(dtype_combo, (list, tuple)):
            return f"dtypes{tuple(dtype_combo)}"
        else:
            return str(dtype_combo)

    def _filter_tensor_dtypes_by_device(self, device, tensor_dtypes):
        if device in ():
            return [dt for dt in tensor_dtypes if dt != infinicore.bfloat16]
        else:
            return tensor_dtypes

    def print_summary(self):
        if self.failed_tests:
            print(f"\n\033[91m{len(self.failed_tests)} tests failed:\033[0m")
            for failure in self.failed_tests:
                print(f"  - {failure}")
            return False
        else:
            print("\n\033[92mAll tests passed!\033[0m")
            return True


class BaseOperatorTest(ABC):
    """Base operator test"""

    def __init__(self, operator_name):
        self.operator_name = operator_name
        self.test_cases = self.get_test_cases()
        self.tensor_dtypes = self.get_tensor_dtypes()
        self.tolerance_map = self.get_tolerance_map()
        self.dtype_combinations = self.get_dtype_combinations()

    @abstractmethod
    def get_test_cases(self):
        """Return list of TestCase objects"""
        pass

    @abstractmethod
    def get_tensor_dtypes(self):
        """Return supported data types"""
        pass

    @abstractmethod
    def get_tolerance_map(self):
        """Return tolerance configuration"""
        pass

    def get_dtype_combinations(self):
        """Return dtype combinations for mixed dtype tests"""
        return None

    def torch_operator(self, *inputs, out=None, **kwargs):
        """Unified PyTorch operator function - can be overridden or return None"""
        raise NotImplementedError("torch_operator not implemented")

    def infinicore_operator(self, *inputs, out=None, **kwargs):
        """Unified InfiniCore operator function - can be overridden or return None"""
        raise NotImplementedError("infinicore_operator not implemented")

    def create_strided_tensor(
        self, shape, strides, dtype, device, init_mode=TensorInitializer.RANDOM
    ):
        """Create a non-contiguous tensor with specific strides"""
        spec = TensorSpec.from_strided_tensor(shape, strides, dtype, init_mode)
        return spec.create_torch_tensor(device, dtype)

    def prepare_inputs(self, test_case, device, dtype_config):
        """Prepare input data"""
        inputs = []

        for i, input_spec in enumerate(test_case.inputs):
            if isinstance(input_spec, TensorSpec):
                if input_spec.is_scalar:
                    inputs.append(input_spec.value)
                else:
                    tensor = input_spec.create_torch_tensor(device, dtype_config, i)
                    inputs.append(tensor)
            else:
                inputs.append(input_spec)

        return inputs, test_case.kwargs

    def prepare_inputs_for_inplace(
        self, test_case, device, dtype_config, make_copy=True
    ):
        """
        Prepare inputs for in-place operations with optional copying to preserve original data

        Args:
            test_case: The test case
            device: Target device
            dtype_config: Data type configuration (infinicore dtype)
            make_copy: Whether to create copies of inputs (needed for comparison)

        Returns:
            tuple: (inputs, kwargs, original_inputs_for_comparison)
        """
        inputs, kwargs = self.prepare_inputs(test_case, device, dtype_config)
        original_inputs = None

        if make_copy:
            # Create deep copies of input tensors for comparison
            original_inputs = []
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    original_inputs.append(inp.clone().detach())
                else:
                    original_inputs.append(copy.deepcopy(inp))
        else:
            original_inputs = inputs

        return inputs, kwargs, original_inputs

    def run_test(self, device, test_case, dtype_config, config):
        """Enhanced test execution flow with additional in-place modes"""

        # Handle new in-place modes
        if test_case.operation_mode in [
            TestCase.IN_PLACE_0,
            TestCase.IN_PLACE_1,
            TestCase.IN_PLACE_MULTI,
        ]:
            self._run_input_inplace_test(device, test_case, dtype_config, config)
            return

        # Original logic for OUT_OF_PLACE, IN_PLACE, BOTH
        if test_case.operation_mode == TestCase.BOTH:
            out_of_place_case = TestCase(
                TestCase.OUT_OF_PLACE,
                test_case.inputs,
                test_case.output,
                **test_case.kwargs,
            )
            self._run_single_test(
                device, out_of_place_case, dtype_config, config, "OUT_OF_PLACE"
            )

            if test_case.output is not None:
                in_place_case = TestCase(
                    TestCase.IN_PLACE,
                    test_case.inputs,
                    test_case.output,
                    **test_case.kwargs,
                )
                self._run_single_test(
                    device, in_place_case, dtype_config, config, "IN_PLACE"
                )
            return

        self._run_single_test(
            device, test_case, dtype_config, config, test_case.operation_mode.upper()
        )

    def get_output_dtype(self, test_case, dtype_config, torch_result=None):
        """Determine output dtype - returns infinicore dtype, not torch dtype"""
        if test_case.output and test_case.output.dtype is not None:
            return test_case.output.dtype
        elif isinstance(dtype_config, dict) and "output" in dtype_config:
            return dtype_config["output"]
        elif torch_result is not None:
            return to_infinicore_dtype(torch_result.dtype)
        else:
            if isinstance(dtype_config, (list, tuple)):
                return dtype_config[0]
            else:
                return dtype_config

    def run_test(self, device, test_case, dtype_config, config):
        """Enhanced test execution flow with additional in-place modes"""
        device_str = torch_device_map[device]

        # Handle new in-place modes
        if test_case.operation_mode in [
            TestCase.IN_PLACE_0,
            TestCase.IN_PLACE_1,
            TestCase.IN_PLACE_MULTI,
        ]:
            self._run_input_inplace_test(device, test_case, dtype_config, config)
            return

        # Original logic for OUT_OF_PLACE, IN_PLACE, BOTH
        if test_case.operation_mode == TestCase.BOTH:
            out_of_place_case = TestCase(
                TestCase.OUT_OF_PLACE,
                test_case.inputs,
                test_case.output,
                **test_case.kwargs,
            )
            self._run_single_test(
                device, out_of_place_case, dtype_config, config, "OUT_OF_PLACE"
            )

            if test_case.output is not None:
                in_place_case = TestCase(
                    TestCase.IN_PLACE,
                    test_case.inputs,
                    test_case.output,
                    **test_case.kwargs,
                )
                self._run_single_test(
                    device, in_place_case, dtype_config, config, "IN_PLACE"
                )
            return

        self._run_single_test(
            device, test_case, dtype_config, config, test_case.operation_mode.upper()
        )

    def _run_single_test(self, device, test_case, dtype_config, config, mode_name):
        """Run a single test with specified operation mode"""
        device_str = torch_device_map[device]

        inputs, kwargs = self.prepare_inputs(test_case, device, dtype_config)

        infini_inputs = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                infini_tensor = infinicore_tensor_from_torch(inp)
                infini_inputs.append(infini_tensor)
            else:
                infini_inputs.append(inp)

        # Check if operators are implemented
        torch_implemented = True
        infini_implemented = True

        try:
            torch_result = self.torch_operator(*inputs, **kwargs)
            if torch_result is None:
                torch_implemented = False
        except NotImplementedError:
            torch_implemented = False
            torch_result = None

        try:
            infini_result = self.infinicore_operator(*infini_inputs, **kwargs)
            if infini_result is None:
                infini_implemented = False
        except NotImplementedError:
            infini_implemented = False
            infini_result = None

        # If neither operator is implemented, skip the test
        if not torch_implemented and not infini_implemented:
            print(f"⚠ Both operators not implemented - test skipped")
            return

        # If only one operator is implemented, run it without comparison
        if not torch_implemented or not infini_implemented:
            missing_op = (
                "torch_operator" if not torch_implemented else "infinicore_operator"
            )
            print(
                f"⚠ {missing_op} not implemented - running single operator without comparison"
            )

            # Run the available operator for benchmarking if requested
            if config.bench:
                if torch_implemented:

                    def torch_op():
                        return self.torch_operator(*inputs, **kwargs)

                    print(f"  {mode_name}:")
                    profile_operation(
                        "PyTorch   ",
                        torch_op,
                        device_str,
                        config.num_prerun,
                        config.num_iterations,
                    )
                if infini_implemented:

                    def infini_op():
                        return self.infinicore_operator(*infini_inputs, **kwargs)

                    print(f"  {mode_name}:")
                    profile_operation(
                        "InfiniCore",
                        infini_op,
                        device_str,
                        config.num_prerun,
                        config.num_iterations,
                    )
            return

        # Both operators are implemented - proceed with normal comparison
        if test_case.operation_mode == TestCase.OUT_OF_PLACE:

            def torch_op():
                return self.torch_operator(*inputs, **kwargs)

            torch_result = torch_op()

            if (
                isinstance(torch_result, torch.Tensor)
                and not torch_result.is_contiguous()
            ):
                torch_result = torch_result.contiguous()

            def infini_op():
                return self.infinicore_operator(*infini_inputs, **kwargs)

            infini_result = infini_op()

            # Get comparison dtype (infinicore dtype)
            comparison_dtype = self.get_output_dtype(
                test_case, dtype_config, torch_result
            )

            compare_fn = create_test_comparator(
                config, comparison_dtype, mode_name=f"{mode_name}"
            )
            is_valid = compare_fn(infini_result, torch_result)
            assert is_valid, f"{mode_name} result comparison failed"

            if config.bench:
                print(f"  {mode_name}:")
                profile_operation(
                    "PyTorch   ",
                    torch_op,
                    device_str,
                    config.num_prerun,
                    config.num_iterations,
                )
                profile_operation(
                    "InfiniCore",
                    infini_op,
                    device_str,
                    config.num_prerun,
                    config.num_iterations,
                )

        else:
            if not test_case.output:
                raise ValueError("IN_PLACE test requires output specification")

            # Get output dtype and create output tensor
            output_dtype = self.get_output_dtype(test_case, dtype_config)
            output_shape = test_case.output.shape

            # Use TensorSpec to create output tensor with specified initialization mode
            if test_case.output.is_contiguous or test_case.output.strides is None:
                output_spec = TensorSpec.from_tensor(
                    output_shape, output_dtype, init_mode=test_case.output.init_mode
                )
            else:
                output_spec = TensorSpec.from_strided_tensor(
                    output_shape,
                    test_case.output.strides,
                    output_dtype,
                    init_mode=test_case.output.init_mode,
                )

            torch_output = output_spec.create_torch_tensor(device, output_dtype)

            # For non-contiguous tensors, we need to ensure zeros initialization
            if (
                not test_case.output.is_contiguous
                and test_case.output.strides is not None
            ):
                torch_output.zero_()

            def torch_op_inplace():
                self.torch_operator(*inputs, out=torch_output, **kwargs)

            torch_op_inplace()

            # Create infinicore output tensor
            torch_dummy = torch.zeros(
                output_shape, dtype=to_torch_dtype(output_dtype), device=device_str
            )
            if (
                not test_case.output.is_contiguous
                and not test_case.output.strides is None
            ):
                rearrange_tensor(torch_dummy, list(torch_output.stride()))
            infini_output = infinicore_tensor_from_torch(torch_dummy)

            def infini_op_inplace():
                self.infinicore_operator(*infini_inputs, out=infini_output, **kwargs)

            infini_op_inplace()

            comparison_dtype = self.get_output_dtype(
                test_case, dtype_config, torch_output
            )
            compare_fn = create_test_comparator(
                config, comparison_dtype, mode_name=f"{mode_name}"
            )
            is_valid = compare_fn(infini_output, torch_output)
            assert is_valid, f"{mode_name} result comparison failed"

            if config.bench:
                print(f"  {mode_name}:")
                profile_operation(
                    "PyTorch   ",
                    torch_op_inplace,
                    device_str,
                    config.num_prerun,
                    config.num_iterations,
                )
                profile_operation(
                    "InfiniCore",
                    infini_op_inplace,
                    device_str,
                    config.num_prerun,
                    config.num_iterations,
                )

    def _run_input_inplace_test(self, device, test_case, dtype_config, config):
        """
        Run test where an input tensor is modified in-place
        e.g., elu(x, inplace=True) or add(a, b, out=a)
        """
        device_str = torch_device_map[device]

        # Prepare inputs with copies for comparison
        inputs, kwargs, original_inputs = self.prepare_inputs_for_inplace(
            test_case, device, dtype_config, make_copy=True
        )

        # Determine which input is the in-place target
        if test_case.operation_mode == TestCase.IN_PLACE_0:
            target_index = (
                test_case.inplace_target if test_case.inplace_target is not None else 0
            )
        elif test_case.operation_mode == TestCase.IN_PLACE_1:
            target_index = (
                test_case.inplace_target if test_case.inplace_target is not None else 1
            )
        elif test_case.operation_mode == TestCase.IN_PLACE_MULTI:
            # For multi mode, test all possible in-place targets
            self._run_multi_inplace_test(device, test_case, dtype_config, config)
            return
        else:
            target_index = test_case.inplace_target

        if target_index >= len(inputs):
            raise ValueError(
                f"Invalid inplace_target {target_index} for {len(inputs)} inputs"
            )

        target_input = inputs[target_index]
        original_target = original_inputs[target_index]

        # Create infinicore inputs - ensure we use the same tensor objects
        infini_inputs = []
        infini_target = None

        for i, inp in enumerate(inputs):
            if isinstance(inp, torch.Tensor):
                infini_tensor = infinicore_tensor_from_torch(inp)
                infini_inputs.append(infini_tensor)
                if i == target_index:
                    infini_target = infini_tensor
            else:
                infini_inputs.append(inp)

        # Check if operators are implemented
        torch_implemented = True
        infini_implemented = True

        # Run torch operator with in-place on target input
        try:
            # For operators like elu(x, inplace=True)
            if "inplace" in kwargs and kwargs["inplace"]:
                torch_result = self.torch_operator(*inputs, **kwargs)
                # The target input should be modified in-place
                torch_modified = target_input
            else:
                # For operators like add(a, b, out=a)
                out_kwargs = kwargs.copy()
                out_kwargs["out"] = target_input
                torch_result = self.torch_operator(*inputs, **out_kwargs)
                torch_modified = target_input
        except NotImplementedError:
            torch_implemented = False
            torch_modified = None
        except Exception as e:
            if not torch_implemented:
                raise RuntimeError(f"Torch operator failed: {e}")

        # Run infinicore operator with in-place on target input
        try:
            if "inplace" in kwargs and kwargs["inplace"]:
                infini_result = self.infinicore_operator(*infini_inputs, **kwargs)
                infini_modified = infini_target
            else:
                out_kwargs = kwargs.copy()
                out_kwargs["out"] = infini_target
                infini_result = self.infinicore_operator(*infini_inputs, **out_kwargs)
                infini_modified = infini_target
        except NotImplementedError:
            infini_implemented = False
            infini_modified = None
        except Exception as e:
            if not infini_implemented:
                raise RuntimeError(f"Infinicore operator failed: {e}")

        # If neither operator is implemented, skip the test
        if not torch_implemented and not infini_implemented:
            print(f"⚠ Both operators not implemented - test skipped")
            return

        # If only one operator is implemented, run it without comparison
        if not torch_implemented or not infini_implemented:
            missing_op = (
                "torch_operator" if not torch_implemented else "infinicore_operator"
            )
            print(
                f"⚠ {missing_op} not implemented - running single operator without comparison"
            )

            # Run the available operator for benchmarking if requested
            if config.bench:
                if torch_implemented:

                    def torch_inplace_op():
                        # Restore original input for each benchmark iteration
                        if isinstance(target_input, torch.Tensor) and isinstance(
                            original_target, torch.Tensor
                        ):
                            target_input.copy_(original_target)

                        if "inplace" in kwargs and kwargs["inplace"]:
                            return self.torch_operator(*inputs, **kwargs)
                        else:
                            out_kwargs = kwargs.copy()
                            out_kwargs["out"] = target_input
                            return self.torch_operator(*inputs, **out_kwargs)

                    print(f"  {test_case.operation_mode.upper()}:")
                    profile_operation(
                        "PyTorch   ",
                        torch_inplace_op,
                        device_str,
                        config.num_prerun,
                        config.num_iterations,
                    )

                if infini_implemented:

                    def infini_inplace_op():
                        # For infinicore, we need to recreate the inputs for each iteration
                        bench_inputs, _, _ = self.prepare_inputs_for_inplace(
                            test_case, device, dtype_config, make_copy=False
                        )
                        infini_inputs_bench = []
                        infini_target_bench = None

                        for i, inp in enumerate(bench_inputs):
                            if isinstance(inp, torch.Tensor):
                                infini_tensor = infinicore_tensor_from_torch(inp)
                                infini_inputs_bench.append(infini_tensor)
                                if i == target_index:
                                    infini_target_bench = infini_tensor
                            else:
                                infini_inputs_bench.append(inp)

                        if "inplace" in kwargs and kwargs["inplace"]:
                            return self.infinicore_operator(
                                *infini_inputs_bench, **kwargs
                            )
                        else:
                            out_kwargs = kwargs.copy()
                            out_kwargs["out"] = infini_target_bench
                            return self.infinicore_operator(
                                *infini_inputs_bench, **out_kwargs
                            )

                    print(f"  {test_case.operation_mode.upper()}:")
                    profile_operation(
                        "InfiniCore",
                        infini_inplace_op,
                        device_str,
                        config.num_prerun,
                        config.num_iterations,
                    )
            return

        # Both operators are implemented - proceed with normal comparison
        # Compare the modified target inputs
        comparison_dtype = self.get_output_dtype(
            test_case, dtype_config, torch_modified
        )
        compare_fn = create_test_comparator(
            config, comparison_dtype, mode_name=f"{test_case.operation_mode.upper()}"
        )

        is_valid = compare_fn(infini_modified, torch_modified)
        assert is_valid, f"{test_case.operation_mode} result comparison failed"

        # Benchmark if requested
        if config.bench:

            def torch_inplace_op():
                # Restore original input for each benchmark iteration
                if isinstance(target_input, torch.Tensor) and isinstance(
                    original_target, torch.Tensor
                ):
                    target_input.copy_(original_target)

                if "inplace" in kwargs and kwargs["inplace"]:
                    return self.torch_operator(*inputs, **kwargs)
                else:
                    out_kwargs = kwargs.copy()
                    out_kwargs["out"] = target_input
                    return self.torch_operator(*inputs, **out_kwargs)

            def infini_inplace_op():
                # For infinicore, we need to recreate the inputs for each iteration
                bench_inputs, _, _ = self.prepare_inputs_for_inplace(
                    test_case, device, dtype_config, make_copy=False
                )
                infini_inputs_bench = []
                infini_target_bench = None

                for i, inp in enumerate(bench_inputs):
                    if isinstance(inp, torch.Tensor):
                        infini_tensor = infinicore_tensor_from_torch(inp)
                        infini_inputs_bench.append(infini_tensor)
                        if i == target_index:
                            infini_target_bench = infini_tensor
                    else:
                        infini_inputs_bench.append(inp)

                if "inplace" in kwargs and kwargs["inplace"]:
                    return self.infinicore_operator(*infini_inputs_bench, **kwargs)
                else:
                    out_kwargs = kwargs.copy()
                    out_kwargs["out"] = infini_target_bench
                    return self.infinicore_operator(*infini_inputs_bench, **out_kwargs)

            print(f"  {test_case.operation_mode.upper()}:")
            profile_operation(
                "PyTorch   ",
                torch_inplace_op,
                device_str,
                config.num_prerun,
                config.num_iterations,
            )
            profile_operation(
                "InfiniCore",
                infini_inplace_op,
                device_str,
                config.num_prerun,
                config.num_iterations,
            )

    def _run_multi_inplace_test(self, device, test_case, dtype_config, config):
        """
        Run test with multiple in-place options (e.g., test both add(a, b, out=a) and add(a, b, out=b))
        """
        # Test in-place on each input
        for target_index in range(len(test_case.inputs)):
            # Create a copy of the test case with specific target
            case = TestCase(
                TestCase.IN_PLACE_0,  # Use IN_PLACE_0 as base
                test_case.inputs,
                test_case.output,
                inplace_target=target_index,
                **test_case.kwargs,
            )
            print(f"Testing in-place on input {target_index}")
            self._run_input_inplace_test(device, case, dtype_config, config)
