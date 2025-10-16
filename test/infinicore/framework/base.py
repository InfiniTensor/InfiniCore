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
    """Tensor specification supporting various input types and per-tensor dtype"""

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
        self.value = value
        self.is_scalar = is_scalar
        self.is_contiguous = is_contiguous

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
        return cls(
            shape=shape,
            dtype=dtype,
            strides=strides,
            is_scalar=False,
            is_contiguous=False,
        )


class TestCase:
    """Test case with enhanced parsing support"""

    OUT_OF_PLACE = "out_of_place"
    IN_PLACE = "in_place"
    BOTH = "both"

    def __init__(self, operation_mode, inputs, output=None, **kwargs):
        if operation_mode not in [self.IN_PLACE, self.OUT_OF_PLACE, self.BOTH]:
            raise ValueError(f"Invalid operation_mode: {operation_mode}")

        if operation_mode == self.IN_PLACE and output is None:
            raise ValueError("IN_PLACE mode requires output specification")

        self.operation_mode = operation_mode
        self.inputs = []

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
                if hasattr(inp, "is_contiguous") and not inp.is_contiguous:
                    input_strs.append(f"strided_tensor{inp.shape}{dtype_str}")
                else:
                    input_strs.append(f"tensor{inp.shape}{dtype_str}")
            else:
                input_strs.append(str(inp))

        base_str = f"TestCase(mode={mode_str}, inputs=[{', '.join(input_strs)}]"
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
    """Enhanced base operator test with simplified test case parsing"""

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

    @abstractmethod
    def torch_operator(self, *inputs, out=None, **kwargs):
        """Unified PyTorch operator function"""
        pass

    @abstractmethod
    def infinicore_operator(self, *inputs, out=None, **kwargs):
        """Unified Infinicore operator function"""
        pass

    def create_strided_tensor(self, shape, strides, dtype, device_str):
        """Create a non-contiguous tensor with specific strides"""
        total_size = 1
        for i in range(len(shape)):
            total_size += (shape[i] - 1) * abs(strides[i])

        base_tensor = torch.rand(total_size, dtype=dtype, device=device_str)
        strided_tensor = torch.as_strided(base_tensor, shape, strides)
        return strided_tensor

    def prepare_inputs(self, test_case, device_str, dtype_config):
        """Prepare input data"""
        inputs = []

        for i, input_spec in enumerate(test_case.inputs):
            if isinstance(input_spec, TensorSpec):
                if input_spec.is_scalar:
                    inputs.append(input_spec.value)
                else:
                    shape = input_spec.shape

                    if input_spec.dtype is not None:
                        tensor_dtype = to_torch_dtype(input_spec.dtype)
                    elif (
                        isinstance(dtype_config, dict) and f"input_{i}" in dtype_config
                    ):
                        tensor_dtype = to_torch_dtype(dtype_config[f"input_{i}"])
                    elif isinstance(dtype_config, (list, tuple)) and i < len(
                        dtype_config
                    ):
                        tensor_dtype = to_torch_dtype(dtype_config[i])
                    else:
                        tensor_dtype = to_torch_dtype(dtype_config)

                    if input_spec.is_contiguous or input_spec.strides is None:
                        tensor = torch.rand(
                            shape, dtype=tensor_dtype, device=device_str
                        )
                    else:
                        tensor = self.create_strided_tensor(
                            shape, input_spec.strides, tensor_dtype, device_str
                        )

                    inputs.append(tensor)
            else:
                inputs.append(input_spec)

        return inputs, test_case.kwargs

    def get_output_dtype(self, test_case, dtype_config, torch_result=None):
        """Determine output dtype"""
        if test_case.output and test_case.output.dtype is not None:
            return to_torch_dtype(test_case.output.dtype)
        elif isinstance(dtype_config, dict) and "output" in dtype_config:
            return to_torch_dtype(dtype_config["output"])
        elif torch_result is not None:
            return torch_result.dtype
        else:
            if isinstance(dtype_config, (list, tuple)):
                return to_torch_dtype(dtype_config[0])
            else:
                return to_torch_dtype(dtype_config)

    def run_test(self, device, test_case, dtype_config, config):
        """Unified test execution flow"""
        device_str = torch_device_map[device]

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

        inputs, kwargs = self.prepare_inputs(test_case, device_str, dtype_config)

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

            comparison_dtype = to_infinicore_dtype(
                self.get_output_dtype(test_case, dtype_config, torch_result)
            )

            compare_fn = create_test_comparator(
                config, comparison_dtype, mode_name=f"{self.operator_name} {mode_name}"
            )
            is_valid = compare_fn(infini_result, torch_result)
            assert is_valid, f"{self.operator_name} {mode_name} test failed"

            if config.bench:
                profile_operation(
                    f"PyTorch {self.operator_name} {mode_name}",
                    torch_op,
                    device_str,
                    config.num_prerun,
                    config.num_iterations,
                )
                profile_operation(
                    f"Infinicore {self.operator_name} {mode_name}",
                    infini_op,
                    device_str,
                    config.num_prerun,
                    config.num_iterations,
                )

        else:
            if not test_case.output:
                raise ValueError("IN_PLACE test requires output specification")

            output_dtype = self.get_output_dtype(test_case, dtype_config)
            output_shape = test_case.output.shape

            if test_case.output.is_contiguous or test_case.output.strides is None:
                torch_output = torch.zeros(
                    output_shape, dtype=output_dtype, device=device_str
                )
            else:
                torch_output = self.create_strided_tensor(
                    output_shape, test_case.output.strides, output_dtype, device_str
                )
                torch_output.zero_()

            def torch_op_inplace():
                self.torch_operator(*inputs, out=torch_output, **kwargs)

            torch_op_inplace()

            torch_dummy = torch.zeros(
                output_shape, dtype=output_dtype, device=device_str
            )
            if test_case.output.is_contiguous or test_case.output.strides is None:
                infini_output = create_infinicore_tensor(torch_dummy, device_str)
            else:
                rearrange_tensor(torch_dummy, list(torch_output.stride()))
                infini_output = create_strided_infinicore_tensor(
                    torch_dummy, device_str
                )

            def infini_op_inplace():
                self.infinicore_operator(*infini_inputs, out=infini_output, **kwargs)

            infini_op_inplace()

            comparison_dtype = to_infinicore_dtype(
                self.get_output_dtype(test_case, dtype_config, torch_output)
            )
            compare_fn = create_test_comparator(
                config, comparison_dtype, mode_name=f"{self.operator_name} {mode_name}"
            )
            is_valid = compare_fn(infini_output, torch_output)
            assert is_valid, f"{self.operator_name} {mode_name} test failed"

            if config.bench:
                profile_operation(
                    f"PyTorch {self.operator_name} {mode_name}",
                    torch_op_inplace,
                    device_str,
                    config.num_prerun,
                    config.num_iterations,
                )
                profile_operation(
                    f"Infinicore {self.operator_name} {mode_name}",
                    infini_op_inplace,
                    device_str,
                    config.num_prerun,
                    config.num_iterations,
                )
