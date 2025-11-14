import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (shape, dims_tuple, input_strides_or_None)
# infinicore.flip(input, dims)

_TEST_CASES_DATA = [
    ((13, 4), (0,), None),
    ((8, 16), (1,), (128, 1)),
    ((2, 3, 4), (2,), None),
    ((4, 5, 6), (0, 2), None),
    ((16, 64), (0, 1), None),
    ((2, 2, 3, 4), (1, 3), None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for data in _TEST_CASES_DATA:
        shape, dims = data[0], data[1]
        in_strides = data[2] if len(data) > 2 else None

        supports_inplace = not is_broadcast(in_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-4})
            in_spec = TensorSpec.from_tensor(shape, in_strides, dtype)

            kwargs = {"dims": dims}
            test_cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"flip - OUT_OF_PLACE",
                )
            )

            # infinicore.flip has no explicit out or inplace flag; skip in-place/out variants.

    return test_cases


class OpTest(BaseOperatorTest):
    """Flip operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Flip")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        # dims = kwargs.pop("dims", None)
        # if dims is not None:
        #     return infinicore.flip(*args, dims)
        return torch.flip(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     dims = kwargs.pop("dims", None)
    #     if dims is not None:
    #         return infinicore.flip(*args, dims)
    #     return infinicore.flip(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
