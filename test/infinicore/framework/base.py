import torch
import infinicore

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Union

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
    """Enhanced tensor specification supporting various input types"""

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
        self.dtype = dtype
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
    """Enhanced test case supporting flexible input/output specifications"""

    def __init__(self, inputs, output=None, **kwargs):
        """
        简化构造函数
        Args:
            inputs: List[TensorSpec] 或简单的形状元组
            output: TensorSpec 或形状元组
        """
        # 标准化 inputs
        self.inputs = []
        for inp in inputs:
            if isinstance(inp, (list, tuple)):
                self.inputs.append(TensorSpec.from_tensor(inp))
            elif isinstance(inp, TensorSpec):
                self.inputs.append(inp)
            else:
                self.inputs.append(inp)

        # 标准化 output
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
                input_strs.append(f"scalar({inp.value})")
            elif hasattr(inp, "shape"):
                if hasattr(inp, "is_contiguous") and not inp.is_contiguous:
                    input_strs.append(f"strided_tensor{inp.shape}")
                else:
                    input_strs.append(f"tensor{inp.shape}")
            else:
                input_strs.append(str(inp))

        base_str = f"TestCase(inputs=[{', '.join(input_strs)}]"
        if self.output:
            base_str += f", output=tensor{self.output.shape}"
        if self.kwargs:
            base_str += f", kwargs={self.kwargs}"
        if self.description:
            base_str += f", desc='{self.description}'"
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
    ):
        self.tensor_dtypes = tensor_dtypes
        self.tolerance_map = tolerance_map
        self.debug = debug
        self.bench = bench
        self.num_prerun = num_prerun
        self.num_iterations = num_iterations


class TestRunner:
    """Test runner"""

    def __init__(self, test_cases, test_config):
        self.test_cases = test_cases
        self.config = test_config
        self.failed_tests = []  # Track failures

    def run_tests(self, devices, test_func):
        """Run tests and track failures"""
        for device in devices:
            print(f"\n{'='*60}")
            print(f"Testing on {InfiniDeviceNames[device]}")
            print(f"{'='*60}")

            # Filter unsupported data types
            tensor_dtypes = self._filter_tensor_dtypes_by_device(
                device, self.config.tensor_dtypes
            )

            for test_case in self.test_cases:
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
    """Enhanced base operator test supporting flexible input/output"""

    def __init__(self, operator_name):
        self.operator_name = operator_name
        self.test_cases = self.get_test_cases()
        self.tensor_dtypes = self.get_tensor_dtypes()
        self.tolerance_map = self.get_tolerance_map()

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

    @abstractmethod
    def torch_operator(self, *inputs, **kwargs):
        """PyTorch operator implementation"""
        pass

    @abstractmethod
    def infinicore_operator(self, *inputs, **kwargs):
        """Infinicore operator implementation"""
        pass

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

    def prepare_inputs(self, test_case, device_str, dtype):
        """Prepare input data - handles various input types including strided tensors"""
        torch_dtype = to_torch_dtype(dtype)
        inputs = []

        for input_spec in test_case.inputs:
            if isinstance(input_spec, TensorSpec):
                if input_spec.is_scalar:
                    # Handle scalar inputs
                    inputs.append(input_spec.value)
                else:
                    # Handle tensor inputs
                    shape = input_spec.shape
                    tensor_dtype = (
                        torch_dtype
                        if input_spec.dtype is None
                        else to_torch_dtype(input_spec.dtype)
                    )

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

    def run_test(self, device, test_case, dtype, config):
        """Generic test execution flow with flexible inputs - output is always contiguous"""
        device_str = torch_device_map[device]

        # Prepare inputs
        inputs, kwargs = self.prepare_inputs(test_case, device_str, dtype)

        # PyTorch reference result - output is always contiguous for out-of-place
        def torch_op():
            return self.torch_operator(*inputs, **kwargs)

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

        # Infinicore result - output is always contiguous for out-of-place
        def infini_op():
            return self.infinicore_operator(*infini_inputs, **kwargs)

        infini_result = infini_op()

        # Result comparison
        compare_fn = create_test_comparator(config, dtype)
        is_valid = compare_fn(infini_result, torch_result)
        assert is_valid, f"{self.operator_name} test failed"

        # Performance testing
        if config.bench:
            profile_operation(
                f"PyTorch {self.operator_name}",
                torch_op,
                device_str,
                config.num_prerun,
                config.num_iterations,
            )
            profile_operation(
                f"Infinicore {self.operator_name}",
                infini_op,
                device_str,
                config.num_prerun,
                config.num_iterations,
            )

    def run_inplace_test(self, device, test_case, dtype, config):
        """Generic in-place operation test execution flow - supports strided output"""
        device_str = torch_device_map[device]

        # Prepare inputs and output
        inputs, kwargs = self.prepare_inputs(test_case, device_str, dtype)

        if not test_case.output:
            raise ValueError("In-place test requires output specification in test case")

        # PyTorch in-place operation
        output_shape = test_case.output.shape
        output_dtype = (
            to_torch_dtype(dtype)
            if test_case.output.dtype is None
            else to_torch_dtype(test_case.output.dtype)
        )

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
            self.torch_operator(*inputs, out=torch_preallocated, **kwargs)

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

        # # Create infinicore output tensor
        # if test_case.output.is_contiguous or test_case.output.strides is None:
        #     infini_output = infinicore.empty(
        #         output_shape, dtype=dtype, device=infinicore.device(device_str, 0)
        #     )
        # else:
        #     infini_output = infinicore.strided_empty(
        #         output_shape,
        #         test_case.output.strides,
        #         dtype=dtype,
        #         device=infinicore.device(device_str, 0),
        #     )

        torch_dummy = torch.zeros(output_shape, dtype=output_dtype, device=device_str)
        if test_case.output.is_contiguous or test_case.output.strides is None:
            infini_output = create_infinicore_tensor(torch_dummy, device_str)
        else:
            rearrange_tensor(torch_dummy, list(torch_preallocated.stride()))
            infini_output = create_strided_infinicore_tensor(torch_dummy, device_str)

        def infini_op_inplace():
            self.infinicore_operator(*infini_inputs, out=infini_output, **kwargs)

        infini_op_inplace()

        # Result comparison
        compare_fn = create_test_comparator(config, dtype)
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
