import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.tensor import TensorInitializer

# ==============================================================================
# Operator-specific configuration
# ==============================================================================

# Test cases format: (input_shape, num_classes, has_weight, p, margin, reduction)
_TEST_CASES_DATA = [
    # Basic cases without weight - 2D inputs only
    ((10, 5), 5, False, 1, 1.0, "mean"),
    ((10, 5), 5, False, 1, 1.0, "sum"),
    ((10, 5), 5, False, 1, 1.0, "none"),
    ((8, 3), 3, False, 2, 1.0, "mean"),
    ((8, 3), 3, False, 2, 0.5, "sum"),
    # Cases with weight tensor
    ((10, 5), 5, True, 1, 1.0, "mean"),
    ((10, 5), 5, True, 1, 1.0, "sum"),
    ((8, 3), 3, True, 2, 1.0, "mean"),
    ((8, 3), 3, True, 2, 0.5, "sum"),
    # Edge cases - only 2D inputs
    ((1, 3), 3, False, 1, 1.0, "mean"),  # Single sample
    ((5, 1), 1, False, 1, 1.0, "mean"),  # Single class
    ((100, 10), 10, True, 1, 2.0, "mean"),  # Larger tensors
]

# Tolerance configuration
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

# Data types to test
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    """
    Parse test case data for multi_margin_loss operation.
    All tensors will be created on the same device.
    """
    test_cases = []

    for data in _TEST_CASES_DATA:
        input_shape = data[0]
        num_classes = data[1]
        has_weight = data[2]
        p_value = data[3]
        margin_value = data[4]
        reduction = data[5]

        # Generate test cases for all data types
        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})

            # Create input tensor spec
            input_spec = TensorSpec.from_tensor(input_shape, dtype=dtype)

            # FIX: Create target as a tensor, not a scalar
            # For 2D input (batch_size, num_classes), target should be (batch_size,) tensor
            target_shape = (input_shape[0],)
            target_spec = TensorSpec.from_tensor(
                target_shape,
                dtype=infinicore.int64,  # target must be int64 for classification
                init_mode=TensorInitializer.RANDINT,
                low=0,
                high=num_classes,  # class indices from 0 to num_classes-1
            )

            base_description = "MultiMarginLoss"

            # Build kwargs
            kwargs = {"p": p_value, "margin": margin_value, "reduction": reduction}

            # Add weight tensor if specified
            if has_weight:
                weight_spec = TensorSpec.from_tensor(
                    (num_classes,), dtype=dtype, init_mode=TensorInitializer.RANDOM
                )
                kwargs["weight"] = weight_spec

            test_cases.append(
                TestCase(
                    inputs=[input_spec, target_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=base_description,
                )
            )

    return test_cases


class MultiMarginLossOpTest(BaseOperatorTest):
    """MultiMarginLoss operator test with device handling"""

    def __init__(self):
        super().__init__("MultiMarginLoss")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        """PyTorch multi_margin_loss implementation with device handling"""
        return F.multi_margin_loss(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        """InfiniCore multi_margin_loss implementation"""
        return None


def main():
    """Main entry point"""
    runner = GenericTestRunner(MultiMarginLossOpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
