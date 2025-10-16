"""
Templates for common operator patterns to minimize code duplication

Available configuration methods in BaseOperatorTest:

1. get_test_cases() -> List[TestCase]
   - Define input/output shapes, strides, and operation modes
   - Operation modes: TestCase.OUT_OF_PLACE, TestCase.IN_PLACE, TestCase.BOTH

2. get_tensor_dtypes() -> List[infinicore.dtype]
   - Define supported data types for single-dtype tests
   - Used when dtype_combinations is None

3. get_tolerance_map() -> Dict[infinicore.dtype, Dict[str, float]]
   - Set tolerance (atol, rtol) for each data type
   - Example: {infinicore.float16: {"atol": 1e-3, "rtol": 1e-2}}

4. get_dtype_combinations() -> Optional[List[Dict]]
   - Define mixed dtype configurations for multi-dtype tests
   - Return None for single-dtype tests
   - Available mixed dtype definition methods:

   METHOD 1: Explicit dictionary per combination (Recommended)
     Format: [{"input_0": dtype1, "input_1": dtype2, "output": dtype3}]
     Example:
       [{"input_0": infinicore.float16, "input_1": infinicore.float32, "output": infinicore.float16}]

   METHOD 2: Rule-based combination generation
     Generate all valid combinations with business logic constraints
     Example: Output bf16 requires input bf16

   METHOD 3: Multi-output support with complex structure
     Format: [{"inputs": [dtype1, dtype2], "outputs": [dtype3, dtype4], "params": {"scale": dtype5}}]
     Requires special handling in prepare_inputs()

   METHOD 4: Per-tensor specification in TestCase
     Individual tensors can specify dtype in TensorSpec
     Overrides dtype_combinations for specific tensors

   METHOD 5: Hybrid approach with fallback
     Combine explicit combinations with generated ones
     Support both simple and complex dtype requirements

5. torch_operator(*inputs, out=None, **kwargs) -> torch.Tensor
   - Implement PyTorch reference implementation

6. infinicore_operator(*inputs, out=None, **kwargs) -> infinicore.Tensor
   - Implement Infinicore operator implementation

Usage examples:
- Single dtype: Return dtype list from get_tensor_dtypes(), None from get_dtype_combinations()
- Mixed dtype: Return dtype combinations from get_dtype_combinations(), basic dtypes from get_tensor_dtypes()
"""

import torch
import infinicore
from .base import BaseOperatorTest


class BinaryOperatorTest(BaseOperatorTest):
    """Template for binary operators (matmul, add, mul, etc.)"""

    def __init__(self, operator_name, test_cases, tensor_dtypes, tolerance_map):
        self._operator_name = operator_name
        self._test_cases = test_cases
        self._tensor_dtypes = tensor_dtypes
        self._tolerance_map = tolerance_map
        super().__init__(operator_name)

    def get_test_cases(self):
        return self._test_cases

    def get_tensor_dtypes(self):
        return self._tensor_dtypes

    def get_tolerance_map(self):
        return self._tolerance_map

    def torch_operator(self, *inputs, **kwargs):
        """Generic torch operator dispatch"""
        # Support both functional and method calls
        if hasattr(torch, self._operator_name):
            op = getattr(torch, self._operator_name)
        else:
            # Fallback to common operator mappings
            op_mapping = {
                "matmul": torch.matmul,
                "add": torch.add,
                "mul": torch.mul,
                "sub": torch.sub,
                "div": torch.div,
            }
            op = op_mapping.get(self._operator_name)
            if op is None:
                raise NotImplementedError(
                    f"Torch operator {self._operator_name} not implemented"
                )

        return op(*inputs, **kwargs)

    def infinicore_operator(self, *inputs, **kwargs):
        """Generic infinicore operator dispatch"""
        op = getattr(infinicore, self._operator_name)
        return op(*inputs, **kwargs)


class UnaryOperatorTest(BinaryOperatorTest):
    """Template for unary operators (exp, log, sin, etc.)"""

    def torch_operator(self, *inputs, **kwargs):
        # For unary operators, we only use the first input
        if hasattr(torch, self._operator_name):
            op = getattr(torch, self._operator_name)
            return op(inputs[0], **kwargs)
        else:
            return super().torch_operator(*inputs, **kwargs)

    def infinicore_operator(self, *inputs, **kwargs):
        op = getattr(infinicore, self._operator_name)
        return op(inputs[0], **kwargs)
