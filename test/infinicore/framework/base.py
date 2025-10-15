import torch
import infinicore

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Union, Callable, Optional

from .datatypes import to_torch_dtype, to_infinicore_dtype
from .devices import InfiniDeviceNames, torch_device_map
from .utils import (
    create_infinicore_tensor,
    create_strided_infinicore_tensor,
    create_test_comparator,
    profile_operation,
    rearrange_tensor,
    synchronize_device,
)


class TensorSpec:
    """Enhanced tensor specification supporting various input types and per-tensor dtype"""

    def __init__(
        self,
        shape=None,
        dtype=None,
        strides=None,
        value=None,
        is_scalar=False,
        is_contiguous=True,
    ):
        self.shape = shape
        self.dtype = dtype  # Can be specific dtype or None (will use test dtype)
        self.strides = strides
        self.value = value  # For scalar values
        self.is_scalar = is_scalar
        self.is_contiguous = is_contiguous  # Whether tensor should be contiguous

    @classmethod
    def from_tensor(cls, shape, dtype=None, strides=None, is_contiguous=True):
        return cls(
            shape=shape,
            dtype=dtype,
            strides=strides,
            is_scalar=False,
            is_contiguous=is_contiguous,
        )

    @classmethod
    def from_scalar(cls, value, dtype=None):
        return cls(value=value, dtype=dtype, is_scalar=True)

    @classmethod
    def from_strided_tensor(cls, shape, strides, dtype=None):
        """Create a non-contiguous tensor specification with specific strides"""
        return cls(
            shape=shape,
            dtype=dtype,
            strides=strides,
            is_scalar=False,
            is_contiguous=False,
        )


class TestCase:
    """Enhanced test case supporting flexible input/output specifications with per-tensor dtypes"""

    def __init__(self, inputs, output=None, **kwargs):
        """
        Simplified constructor
        Args:
            inputs: List[TensorSpec] or simple shape tuples
            output: TensorSpec or shape tuple
        """
        # Normalize inputs
        self.inputs = []
        for inp in inputs:
            if isinstance(inp, (list, tuple)):
                self.inputs.append(TensorSpec.from_tensor(inp))
            elif isinstance(inp, TensorSpec):
                self.inputs.append(inp)
            else:
                self.inputs.append(inp)

        # Normalize output
        if isinstance(output, (list, tuple)):
            self.output = TensorSpec.from_tensor(output)
        else:
            self.output = output

        self.kwargs = kwargs
        self.description = kwargs.pop("description", "")

    def __str__(self):
        input_strs = []
        for inp in self.inputs:
            if hasattr(inp, "is_scalar") and inp.is_scalar:
                dtype_str = f", dtype={inp.dtype}" if inp.dtype else ""
                input_strs.append(f"scalar({inp.value}{dtype_str})")
            elif hasattr(inp, "shape"):
                dtype_str = f", dtype={inp.dtype}" if inp.dtype else ""
                if hasattr(inp, "is_contiguous") and not inp.is_contiguous:
                    input_strs.append(f"strided_tensor{inp.shape}{dtype_str}")
                else:
                    input_strs.append(f"tensor{inp.shape}{dtype_str}")
            else:
                input_strs.append(str(inp))

        base_str = f"TestCase(inputs=[{', '.join(input_strs)}]"
        if self.output:
            dtype_str = f", dtype={self.output.dtype}" if self.output.dtype else ""
            base_str += f", output=tensor{self.output.shape}{dtype_str}"
        if self.kwargs:
            base_str += f", kwargs={self.kwargs}"
        if self.description:
            base_str += f", desc='{self.description}'"
        base_str += ")"
        return base_str


class TestConfig:
    """Test configuration with support for dtype combinations"""

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
        self.dtype_combinations = (
            dtype_combinations  # List of dtype configs for each tensor
        )


class TestRunner:
    """Test runner with support for dtype combinations"""

    def __init__(self, test_cases, test_config):
        self.test_cases = test_cases
        self.config = test_config
        self.failed_tests = []  # Track failures

    def run_tests(self, devices, test_func, test_type="Test"):
        """Run tests and track failures with support for dtype combinations"""
        for device in devices:
            print(f"\n{'='*60}")
            print(f"Testing {test_type} on {InfiniDeviceNames[device]}")
            print(f"{'='*60}")

            # Filter unsupported data types
            tensor_dtypes = self._filter_tensor_dtypes_by_device(
                device, self.config.tensor_dtypes
            )

            for test_case in self.test_cases:
                # If dtype combinations are specified, use them
                if self.config.dtype_combinations:
                    for dtype_combo in self.config.dtype_combinations:
                        try:
                            test_func(device, test_case, dtype_combo, self.config)
                            combo_str = self._format_dtype_combo(dtype_combo)
                            print(f"✓ {test_case} with {combo_str} passed")
                        except Exception as e:
                            combo_str = self._format_dtype_combo(dtype_combo)
                            error_msg = f"{test_case} with {combo_str} on {InfiniDeviceNames[device]}: {e}"
                            print(f"✗ {error_msg}")
                            self.failed_tests.append(error_msg)
                            if self.config.debug:
                                raise
                else:
                    # Original behavior: single dtype for all tensors
                    for dtype in tensor_dtypes:
                        try:
                            test_func(device, test_case, dtype, self.config)
                            print(f"✓ {test_case} with {dtype} passed")
                        except Exception as e:
                            error_msg = f"{test_case} with {dtype} on {InfiniDeviceNames[device]}: {e}"
                            print(f"✗ {error_msg}")
                            self.failed_tests.append(error_msg)
                            if self.config.debug:
                                raise

        # Return whether any tests failed
        return len(self.failed_tests) == 0

    def _format_dtype_combo(self, dtype_combo):
        """Format dtype combination for display"""
        if isinstance(dtype_combo, dict):
            return f"dtypes({dtype_combo})"
        elif isinstance(dtype_combo, (list, tuple)):
            return f"dtypes{tuple(dtype_combo)}"
        else:
            return str(dtype_combo)

    def _filter_tensor_dtypes_by_device(self, device, tensor_dtypes):
        """Filter data types based on device"""
        if device in ():
            # Filter out unsupported data types on specified devices
            return [dt for dt in tensor_dtypes if dt != infinicore.bfloat16]
        else:
            return tensor_dtypes

    def print_summary(self):
        """Print test summary"""
        if self.failed_tests:
            print(f"\n\033[91m{len(self.failed_tests)} tests failed:\033[0m")
            for failure in self.failed_tests:
                print(f"  - {failure}")
            return False
        else:
            print("\n\033[92mAll tests passed!\033[0m")
            return True


class BaseOperatorTest(ABC):
    """Enhanced base operator test supporting flexible input/output and per-tensor dtypes"""

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
        """Return dtype combinations for tests with mixed dtypes.
        Override this method to specify different dtype combinations.
        Returns None if all tensors should use the same dtype.

        Dictionary based:
        _DTYPE_COMBINATIONS = [
            {
                "input_0": infinicore.float16,
                "input_1": infinicore.float32,
                "input_2": infinicore.bfloat16,
                "output": infinicore.float16,
            },
            {
                "input_0": infinicore.bfloat16,
                "input_1": infinicore.float16,
                "input_2": infinicore.float32,
                "output": infinicore.bfloat16,
            }
        ]

        or

        Position based:
        _DTYPE_COMBINATIONS = [
            (infinicore.float16, infinicore.float32, infinicore.float16),  # (in1, in2, out)
            (infinicore.bfloat16, infinicore.float16, infinicore.bfloat16),
        ]

        or TensorSpec + combinations:

        test_case = TestCase(
            inputs=[
                TensorSpec.from_tensor(shape, dtype=infinicore.float32),  # 固定dtype
                TensorSpec.from_tensor(shape),  # 使用组合配置中的dtype
            ],
            output=TensorSpec.from_tensor(shape)  # 使用组合配置中的dtype
        )

        """
        return None

    def has_out_of_place_test(self):
        """Check if out-of-place test functions are defined"""
        return hasattr(self, "torch_operator_out_of_place") and hasattr(
            self, "infinicore_operator_out_of_place"
        )

    def has_inplace_test(self):
        """Check if in-place test functions are defined"""
        return hasattr(self, "torch_operator_inplace") and hasattr(
            self, "infinicore_operator_inplace"
        )

    def create_strided_tensor(self, shape, strides, dtype, device_str):
        """Create a non-contiguous tensor with specific strides"""
        # Create a larger contiguous tensor and create a strided view
        total_size = 1
        for i in range(len(shape)):
            total_size += (shape[i] - 1) * abs(strides[i])

        # Create base contiguous tensor
        base_tensor = torch.rand(total_size, dtype=dtype, device=device_str)

        # Create strided tensor view
        strided_tensor = torch.as_strided(base_tensor, shape, strides)
        return strided_tensor

    def prepare_inputs(self, test_case, device_str, dtype_config):
        """Prepare input data - handles various input types including strided tensors and per-tensor dtypes"""
        inputs = []

        for i, input_spec in enumerate(test_case.inputs):
            if isinstance(input_spec, TensorSpec):
                if input_spec.is_scalar:
                    # Handle scalar inputs
                    inputs.append(input_spec.value)
                else:
                    # Handle tensor inputs
                    shape = input_spec.shape

                    # Determine dtype for this specific tensor
                    if input_spec.dtype is not None:
                        # Use explicitly specified dtype in TensorSpec
                        tensor_dtype = to_torch_dtype(input_spec.dtype)
                    elif (
                        isinstance(dtype_config, dict) and f"input_{i}" in dtype_config
                    ):
                        # Use dtype from combination config
                        tensor_dtype = to_torch_dtype(dtype_config[f"input_{i}"])
                    elif isinstance(dtype_config, (list, tuple)) and i < len(
                        dtype_config
                    ):
                        # Use dtype from tuple config
                        tensor_dtype = to_torch_dtype(dtype_config[i])
                    else:
                        # Fallback: use single dtype or default
                        tensor_dtype = to_torch_dtype(dtype_config)

                    if input_spec.is_contiguous or input_spec.strides is None:
                        # Create contiguous tensor
                        tensor = torch.rand(
                            shape, dtype=tensor_dtype, device=device_str
                        )
                    else:
                        # Create strided tensor
                        tensor = self.create_strided_tensor(
                            shape, input_spec.strides, tensor_dtype, device_str
                        )

                    inputs.append(tensor)
            else:
                # Handle raw values (scalars, lists, etc.)
                inputs.append(input_spec)

        return inputs, test_case.kwargs

    def get_output_dtype(self, test_case, dtype_config, torch_result=None):
        """Determine output dtype considering per-tensor configuration"""
        if test_case.output and test_case.output.dtype is not None:
            return to_torch_dtype(test_case.output.dtype)
        elif isinstance(dtype_config, dict) and "output" in dtype_config:
            return to_torch_dtype(dtype_config["output"])
        elif torch_result is not None:
            return torch_result.dtype
        else:
            # Fallback to first input dtype or single test dtype
            if isinstance(dtype_config, (list, tuple)):
                return to_torch_dtype(dtype_config[0])
            else:
                return to_torch_dtype(dtype_config)

    def run_out_of_place_test(self, device, test_case, dtype_config, config):
        """Generic out-of-place test execution flow with per-tensor dtype support"""
        if not self.has_out_of_place_test():
            raise NotImplementedError("Out-of-place test functions not defined")

        device_str = torch_device_map[device]

        # Prepare inputs with per-tensor dtype support
        inputs, kwargs = self.prepare_inputs(test_case, device_str, dtype_config)

        # PyTorch reference result
        def torch_op():
            return self.torch_operator_out_of_place(*inputs, **kwargs)

        torch_result = torch_op()

        # Ensure PyTorch result is contiguous
        if isinstance(torch_result, torch.Tensor) and not torch_result.is_contiguous():
            torch_result = torch_result.contiguous()

        # Convert tensor inputs to infinicore (skip scalars and non-tensors)
        infini_inputs = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                # For strided tensors, use strided infinicore tensor
                if not inp.is_contiguous():
                    infini_tensor = infinicore.strided_from_blob(
                        inp.data_ptr(),
                        list(inp.shape),
                        list(inp.stride()),
                        dtype=to_infinicore_dtype(inp.dtype),
                        device=infinicore.device(device_str, 0),
                    )
                else:
                    infini_tensor = create_infinicore_tensor(inp, device_str)
                infini_inputs.append(infini_tensor)
            else:
                infini_inputs.append(inp)

        # Infinicore result
        def infini_op():
            return self.infinicore_operator_out_of_place(*infini_inputs, **kwargs)

        infini_result = infini_op()

        # Determine comparison dtype (use output dtype or result dtype)
        comparison_dtype = to_infinicore_dtype(
            self.get_output_dtype(test_case, dtype_config, torch_result)
        )

        # Result comparison
        compare_fn = create_test_comparator(config, comparison_dtype)
        is_valid = compare_fn(infini_result, torch_result)
        assert is_valid, f"{self.operator_name} out-of-place test failed"

        # Performance testing
        if config.bench:
            profile_operation(
                f"PyTorch {self.operator_name} Out-of-place",
                torch_op,
                device_str,
                config.num_prerun,
                config.num_iterations,
            )
            profile_operation(
                f"Infinicore {self.operator_name} Out-of-place",
                infini_op,
                device_str,
                config.num_prerun,
                config.num_iterations,
            )

    def run_inplace_test(self, device, test_case, dtype_config, config):
        """Generic in-place operation test execution flow with per-tensor dtype support"""
        if not self.has_inplace_test():
            raise NotImplementedError("In-place test functions not defined")

        device_str = torch_device_map[device]

        # Prepare inputs with per-tensor dtype support
        inputs, kwargs = self.prepare_inputs(test_case, device_str, dtype_config)

        if not test_case.output:
            raise ValueError("In-place test requires output specification in test case")

        # Determine output dtype
        output_dtype = self.get_output_dtype(test_case, dtype_config)

        # PyTorch in-place operation
        output_shape = test_case.output.shape

        if test_case.output.is_contiguous or test_case.output.strides is None:
            # Create contiguous output tensor
            torch_preallocated = torch.zeros(
                output_shape, dtype=output_dtype, device=device_str
            )
        else:
            # Create strided output tensor
            torch_preallocated = self.create_strided_tensor(
                output_shape, test_case.output.strides, output_dtype, device_str
            )
            # Zero out the strided tensor
            torch_preallocated.zero_()

        def torch_op_inplace():
            self.torch_operator_inplace(*inputs, out=torch_preallocated, **kwargs)

        torch_op_inplace()

        # Infinicore in-place operation
        infini_inputs = []
        for inp in inputs:
            if isinstance(inp, torch.Tensor):
                if not inp.is_contiguous():
                    infini_tensor = infinicore.strided_from_blob(
                        inp.data_ptr(),
                        list(inp.shape),
                        list(inp.stride()),
                        dtype=to_infinicore_dtype(inp.dtype),
                        device=infinicore.device(device_str, 0),
                    )
                else:
                    infini_tensor = create_infinicore_tensor(inp, device_str)
                infini_inputs.append(infini_tensor)
            else:
                infini_inputs.append(inp)

        # Create infinicore output tensor
        torch_dummy = torch.zeros(output_shape, dtype=output_dtype, device=device_str)
        if test_case.output.is_contiguous or test_case.output.strides is None:
            infini_output = create_infinicore_tensor(torch_dummy, device_str)
        else:
            rearrange_tensor(torch_dummy, list(torch_preallocated.stride()))
            infini_output = create_strided_infinicore_tensor(torch_dummy, device_str)

        def infini_op_inplace():
            self.infinicore_operator_inplace(
                *infini_inputs, out=infini_output, **kwargs
            )

        infini_op_inplace()

        # Result comparison
        comparison_dtype = to_infinicore_dtype(
            self.get_output_dtype(test_case, dtype_config, torch_preallocated)
        )
        compare_fn = create_test_comparator(config, comparison_dtype)
        is_valid = compare_fn(infini_output, torch_preallocated)
        assert is_valid, f"{self.operator_name} in-place test failed"

        # Performance testing
        if config.bench:
            profile_operation(
                f"PyTorch {self.operator_name} In-place",
                torch_op_inplace,
                device_str,
                config.num_prerun,
                config.num_iterations,
            )
            profile_operation(
                f"Infinicore {self.operator_name} In-place",
                infini_op_inplace,
                device_str,
                config.num_prerun,
                config.num_iterations,
            )
