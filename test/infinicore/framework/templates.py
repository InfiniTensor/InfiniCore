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

5. torch_operator(*inputs, out=None, **kwargs) -> torch.Tensor
   - Implement PyTorch reference implementation

6. infinicore_operator(*inputs, out=None, **kwargs) -> infinicore.Tensor
   - Implement Infinicore operator implementation

New Tensor Initialization Modes:
- TensorInitializer.RANDOM (default): Random values using torch.rand
- TensorInitializer.ZEROS: All zeros using torch.zeros
- TensorInitializer.ONES: All ones using torch.ones
- TensorInitializer.RANDINT: Random integers using torch.randint
- TensorInitializer.MANUAL: Use a pre-existing tensor with shape/strides validation
- TensorInitializer.BINARY: Use a pre-existing tensor with shape validation only

Usage examples in TestCase creation:
- Basic: TensorSpec.from_tensor(shape)
- With initialization: TensorSpec.from_tensor(shape, init_mode=TensorInitializer.ZEROS)
- Strided with custom init: TensorSpec.from_strided_tensor(shape, strides, init_mode=TensorInitializer.ONES)
"""

import torch
import infinicore
from .base import BaseOperatorTest
from .tensor import TensorSpec, TensorInitializer


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
