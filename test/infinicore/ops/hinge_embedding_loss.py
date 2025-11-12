import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (input_shape, target_shape, input_strides_or_None, target_strides_or_None, margin_or_None)
# infinicore.nn.functional.hinge_embedding_loss(input, target, margin=1.0, reduction='mean')

_TEST_CASES_DATA = [
    ((4,), (4,), None, None, None),
    ((8,), (8,), None, None, 0.5),
    ((1,), (1,), None, None, None),
    ((16,), (16,), (4,), None, 1.0),
    ((3,), (3,), None, None, 0.2),
    ((2,), (2,), None, None, None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for shape, tgt_shape, s1, s2, margin in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            inp = TensorSpec.from_tensor(shape, s1, dtype)
            tgt = TensorSpec.from_tensor(tgt_shape, s2, dtype)

            kwargs = {}
            if margin is not None:
                kwargs["margin"] = margin

            test_cases.append(
                TestCase(
                    inputs=[inp, tgt],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="hinge_embedding_loss - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """hinge_embedding_loss operator test with simplified implementation"""

    def __init__(self):
        super().__init__("hinge_embedding_loss")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.hinge_embedding_loss(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.hinge_embedding_loss(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
