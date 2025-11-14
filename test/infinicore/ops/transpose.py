import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (shape, dim0, dim1, input_strides_or_None)
# infinicore.transpose(input, dim0, dim1)

_TEST_CASES_DATA = [
    ((13, 4), 0, 1, None),
    ((8, 16), 0, 1, (128, 1)),
    ((2, 3, 4), 0, 2, None),
    ((4, 5, 6), 1, 2, None),
    ((16, 64), 0, 1, None),
    ((2, 2, 3, 4), 1, 3, None),
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
        shape, dim0, dim1 = data[0], data[1], data[2]
        in_strides = data[3] if len(data) > 3 else None

        supports_inplace = not is_broadcast(in_strides)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-4})
            in_spec = TensorSpec.from_tensor(shape, in_strides, dtype)
            out_shape = list(shape)
            out_shape[dim0], out_shape[dim1] = out_shape[dim1], out_shape[dim0]
            out_spec = TensorSpec.from_tensor(tuple(out_shape), None, dtype)

            # Out-of-place
            kwargs = {"dim0": dim0, "dim1": dim1}
            test_cases.append(
                TestCase(
                    inputs=[in_spec],
                    kwargs={"dim0": dim0, "dim1": dim1},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"transpose - OUT_OF_PLACE",
                )
            )

            # In-place via out param not supported; skip explicit out tests.

    return test_cases


class OpTest(BaseOperatorTest):
    """Transpose operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Transpose")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        # dim0 = kwargs.pop("dim0", None)
        # dim1 = kwargs.pop("dim1", None)
        # if dim0 is not None and dim1 is not None:
        #     return infinicore.transpose(*args, dim0, dim1)
        return torch.transpose(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     dim0 = kwargs.pop("dim0", None)
    #     dim1 = kwargs.pop("dim1", None)
    #     if dim0 is not None and dim1 is not None:
    #         return infinicore.transpose(*args, dim0, dim1)
    #     return infinicore.transpose(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
