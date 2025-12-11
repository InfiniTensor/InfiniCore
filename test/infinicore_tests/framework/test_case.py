"""
Test case definitions and related functionality for the InfiniCore testing framework.
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from .tensor import TensorSpec


@dataclass
class TestResult:
    """Test result data structure"""

    success: bool
    return_code: int  # 0: success, -1: failure, -2: skipped, -3: partial
    torch_host_time: float = 0.0
    torch_device_time: float = 0.0
    infini_host_time: float = 0.0
    infini_device_time: float = 0.0
    error_message: str = ""
    test_case: Any = None
    device: Any = None


class TestCase:
    """Test case with all configuration included"""

    def __init__(
        self,
        inputs,
        kwargs=None,
        output_spec=None,
        output_specs=None,
        comparison_target=None,
        description="",
        tolerance=None,
        output_count=1,
    ):
        """
        Initialize a test case with complete configuration

        Args:
            inputs: List of TensorSpec objects, scalars, or tuples containing multiple TensorSpecs
            kwargs: Additional keyword arguments for the operator
            output_spec: TensorSpec for output tensor (for single output operations)
            output_specs: List of TensorSpec for multiple output tensors
            comparison_target: Target for comparison ('out', index, or None for return value)
            description: Test case description
            tolerance: Tolerance settings for this test case {'atol': float, 'rtol': float}
            output_count: Number of outputs (default: 1)
        """
        self.inputs = []

        # Process inputs - support both single TensorSpecs and tuples of TensorSpecs
        for i, inp in enumerate(inputs):
            if isinstance(inp, (list, tuple)):
                # Handle tuple/list of multiple TensorSpecs (e.g., for torch.cat)
                processed_tuple = []
                for j, item in enumerate(inp):
                    if isinstance(item, (list, tuple)):
                        # Nested tuple - recursively process
                        nested_processed = []
                        for k, nested_item in enumerate(item):
                            if isinstance(nested_item, TensorSpec):
                                nested_item.fill_name(f"in_{i}_{j}_{k}")
                                nested_processed.append(nested_item)
                            else:
                                nested_processed.append(nested_item)
                        processed_tuple.append(tuple(nested_processed))
                    elif isinstance(item, TensorSpec):
                        item.fill_name(f"in_{i}_{j}")
                        processed_tuple.append(item)
                    else:
                        processed_tuple.append(item)
                self.inputs.append(tuple(processed_tuple))
            elif isinstance(inp, TensorSpec):
                inp.fill_name(f"in_{i}")
                self.inputs.append(inp)
            else:
                self.inputs.append(inp)

        self.kwargs = kwargs or {}
        self.output_spec = output_spec
        self.output_specs = output_specs
        self.comparison_target = comparison_target
        self.description = description
        self.tolerance = tolerance or {"atol": 1e-5, "rtol": 1e-3}
        self.output_count = output_count

        if self.output_count > 1 and self.output_specs is not None:
            for idx, spec in enumerate(self.output_specs):
                spec.fill_name(f"out_{idx}")

        # Validate output configuration
        if self.output_count == 1:
            if self.output_specs is not None:
                raise ValueError("output_specs cannot be used when output_count=1")
        else:
            if self.output_spec is not None:
                raise ValueError("output_spec cannot be used when output_count>1")
            if (
                self.output_specs is not None
                and len(self.output_specs) != self.output_count
            ):
                raise ValueError(
                    f"output_specs count ({len(self.output_specs)}) must match output_count ({self.output_count})"
                )

    def get_tensor_input_count(self):
        """Count the number of tensor inputs (excluding scalars)"""
        count = 0
        for inp in self.inputs:
            if isinstance(inp, TensorSpec) and not inp.is_scalar:
                count += 1
            elif isinstance(inp, (list, tuple)):
                # Count all TensorSpecs within the tuple
                for item in inp:
                    if isinstance(item, TensorSpec) and not item.is_scalar:
                        count += 1
        return count

    def __str__(self):
        input_strs = []
        for inp in self.inputs:
            if isinstance(inp, (list, tuple)):
                # Handle tuple inputs (e.g., for torch.cat)
                tuple_strs = []
                for item in inp:
                    if isinstance(item, (list, tuple)):
                        # Handle nested tuples
                        nested_strs = []
                        for nested_item in item:
                            nested_strs.append(str(nested_item))
                        tuple_strs.append(f"tuple({', '.join(nested_strs)})")
                    else:
                        tuple_strs.append(str(item))
                input_strs.append(f"tuple({'; '.join(tuple_strs)})")
            else:
                input_strs.append(str(inp))

        base_str = f"TestCase("
        if self.description:
            base_str += f"{self.description}"
        base_str += f" - inputs=[{'; '.join(input_strs)}]"

        if self.kwargs or self.output_spec or self.output_specs:
            kwargs_strs = []
            for key, value in self.kwargs.items():
                if key == "out" and isinstance(value, int):
                    kwargs_strs.append(f"{key}={self.inputs[value].name}")
                else:
                    kwargs_strs.append(f"{key}={value}")

            # Handle output specifications using TensorSpec's __str__
            if self.output_count == 1 and self.output_spec:
                kwargs_strs.append(f"out={self.output_spec}")
            elif self.output_count > 1 and self.output_specs:
                for i, spec in enumerate(self.output_specs):
                    kwargs_strs.append(f"out_{i}={spec}")

            base_str += f", kwargs={{{'; '.join(kwargs_strs)}}}"

        base_str += ")"
        return base_str
