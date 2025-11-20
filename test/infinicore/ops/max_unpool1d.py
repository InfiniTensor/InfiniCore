import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.tensor import TensorInitializer

# Test cases format:
# (input_shape, kernel_size, stride_or_None, padding_or_None,
#  use_output_size, output_length_or_None)
#
#   input_shape      : (N, C, L_out)
#   kernel_size      : int
#   stride_or_None   : int or None (None -> default stride == kernel_size)
#   padding_or_None  : int or None (None -> default padding == 0)
#   use_output_size  : bool
#   output_length    : L_in when use_output_size is True, otherwise None
#
# torch.nn.functional.max_unpool1d(
#     input, indices, kernel_size, stride=None, padding=0, output_size=None
# )

_TEST_CASES_DATA = [
    # ========== Basic cases ==========
    # small sizes and various stride/padding combinations
    ((1, 1, 4),   2, 2,    0,    False, None),
    ((2, 3, 8),   2, 2,    0,    False, None),
    ((2, 3, 8),   3, 2,    1,    False, None),
    ((1, 4, 6),   2, None, 0,    False, None),    # default stride
    ((4, 8, 16),  2, 2,    0,    True,  32),      # L_in = (16-1)*2 - 0 + 2 = 32
    ((4, 8, 16),  3, 2,    1,    True,  31),      # L_in = (16-1)*2 - 2 + 3 = 31
    ((2, 1, 10),  3, 1,    1,    True,  10),      # L_in = (10-1)*1 - 2 + 3 = 10
    ((2, 1, 5),   2, None, 1,    True,  8),       # L_in = (5-1)*2 - 2 + 2 = 8

    # ========== Large-scale performance test cases ==========
    # medium to large sizes for performance and stability
    ((8,  64, 128),   2, 2,    0, False, None),
    ((8,  64, 256),   3, 2,    1, False, None),
    ((4,  128, 512),  2, 2,    0, False, None),
    ((4,  128, 512),  3, 2,    1, False, None),
    ((16, 32, 256),   4, 4,    0, False, None),
    ((16, 32, 256),   3, 1,    1, False, None),
    ((32, 16, 1024),  2, 2,    0, True,  2048),   # L_in = (1024-1)*2 + 2 = 2048
    ((32, 16, 512),   3, 2,    1, True,  1023),   # L_in = (512-1)*2 - 2 + 3 = 1023
    ((2,  256, 2048), 2, 2,    0, True,  4096),   # L_in = (2048-1)*2 + 2 = 4096
    ((2,  256, 2048), 3, 2,    1, True,  4095),   # L_in = (2048-1)*2 - 2 + 3 = 4095
    ((1,  64, 16384), 2, 2,    0, False, None),
    ((1,  64, 8192),  3, 2,    1, False, None),

    # ========== Edge cases ==========
    # extreme small / boundary sizes
    ((1, 1, 1),   1, 1,    0,   False, None),
    ((1, 4, 2),   2, 2,    0,   True,  4),        # L_in = (2-1)*2 + 2 = 4
    ((2, 1, 64),  3, 2,    1,   False, None),
    ((1, 1, 3),   2, 2,    1,   True,  4),        # L_in = (3-1)*2 - 2 + 2 = 4
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]
_INDEX_DTYPE = infinicore.int64


def parse_test_cases():
    """
    Parse test case data and return list of TestCase objects.

    max_unpool1d takes two inputs:
      - pooled input tensor
      - indices tensor (int64), values in [0, kernel_size)
    """
    test_cases = []

    for (input_shape, kernel_size, stride, padding,
         use_output_size, output_length) in _TEST_CASES_DATA:

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})

            # data tensor
            input_spec = TensorSpec.from_tensor(input_shape, None, dtype)

            # indices tensor: same shape, int64, random integers in [0, kernel_size)
            indices_spec = TensorSpec.from_tensor(
                input_shape,
                None,
                _INDEX_DTYPE,
                init_mode=TensorInitializer.RANDINT,
                low=0,
                high=int(kernel_size),
            )

            kwargs = {"kernel_size": kernel_size}
            if stride is not None:
                kwargs["stride"] = stride
            if padding is not None:
                kwargs["padding"] = padding
            if use_output_size and output_length is not None:
                n, c, _ = input_shape
                kwargs["output_size"] = (n, c, output_length)

            test_cases.append(
                TestCase(
                    inputs=[input_spec, indices_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="MaxUnpool1d - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """MaxUnpool1d operator test with simplified implementation"""

    def __init__(self):
        super().__init__("MaxUnpool1d")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.max_unpool1d(*args, **kwargs)

    # Uncomment if InfiniCore implementation is available
    # def infinicore_operator(self, *args, **kwargs):
    #     return infinicore.nn.functional.max_unpool1d(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
