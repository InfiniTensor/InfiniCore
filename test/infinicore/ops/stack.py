import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# ==============================================================================
# Operator-specific configuration for stack
# ==============================================================================

# Test cases format: (base_shape, num_tensors, dim)
#
#   base_shape  : shape of each input tensor
#   num_tensors : number of tensors to stack
#   dim         : dimension at which to insert the new axis

_TEST_CASES_DATA = [
    # ========== Basic cases ==========
    ((8,),              2,  0),
    ((8,),              4,  1),   # stack 1D tensors along a new last dim
    ((2, 3),            3,  0),
    ((2, 3),            3,  1),
    ((2, 3),            3,  2),
    ((4, 5, 6),         2, -1),
    ((4, 5, 6),         4,  0),
    ((3, 4, 5, 6),      2,  2),

    # ========== Large-scale performance test cases ==========
    ((1024,),           8,   0),
    ((2048,),           16,  0),
    ((256, 256),        4,   0),
    ((256, 256),        8,   1),
    ((64, 128, 128),    4,   0),
    ((64, 128, 128),    8,   1),
    ((32, 64, 64, 64),  4,   0),
    ((32, 64, 64, 64),  4,   2),
    ((16, 32, 64, 128), 8,   1),
    ((16, 32, 64, 128), 8,  -1),
    ((8, 16, 32, 64),   16,  0),
    ((8, 16, 32, 64),   16,  3),

    # ========== Edge cases ==========
    ((1,),          2,  0),    # single element
    ((0, 3),        3,  0),    # zero-length dimension
    ((2, 0, 4),     4,  1),    # zero in middle dimension
    ((1, 1, 1),     1,  0),    # single tensor
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    """
    Parse test case data and return list of TestCase objects for stack.
    """
    cases = []
    for base_shape, num_tensors, dim in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP[dtype]

            # Create multiple input specs with the same base shape and dtype
            input_specs = []
            for i in range(num_tensors):
                input_specs.append(
                    TensorSpec.from_tensor(
                        base_shape,
                        None,
                        dtype,
                        name=f"input_{i}",
                    )
                )

            kwargs = {"dim": dim}

            cases.append(
                TestCase(
                    inputs=input_specs,
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="Stack - OUT_OF_PLACE",
                )
            )

    return cases


class OpTest(BaseOperatorTest):
    """Stack operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Stack")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.stack(*args, **kwargs)

    # Uncomment if InfiniCore implementation is available
    # def infinicore_operator(self, *args, **kwargs):
    #     return infinicore.stack(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
