"""
Templates for common operator patterns to minimize code duplication
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
