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
#  use_output_size, output_hw_or_None)
#
#   input_shape      : (N, C, H_out, W_out), pooled feature map
#   kernel_size      : int or (kh, kw)
#   stride_or_None   : int / (sh, sw) / None (None -> stride == kernel_size)
#   padding_or_None  : int / (ph, pw) / None (None -> padding == 0)
#   use_output_size  : bool, whether to pass output_size explicitly
#   output_hw_or_None: (H_in, W_in) if use_output_size is True
#
# torch.nn.functional.max_unpool2d(
#     input, indices, kernel_size, stride=None, padding=0, output_size=None
# )

_TEST_CASES_DATA = [
    # ========== Basic cases ==========
    # small sizes with different stride/padding and optional output_size
    ((1, 1, 16, 16), (2, 2), (2, 2), (0, 0), False, None),
    ((2, 3, 16, 16), (2, 2), None,   None,   False, None),      # default stride / padding
    ((2, 3, 8, 8),   (3, 3), (2, 2), (1, 1), False, None),
    ((1, 4, 7, 9),   (2, 2), (2, 2), (0, 0), False, None),
    ((4, 8, 14, 14), (3, 3), (2, 2), (1, 1), True,  (27, 27)),  # H,W: (14-1)*2 - 2 + 3 = 27
    ((4, 8, 14, 14), (2, 2), None,   (1, 1), True,  (26, 26)),  # H,W: (14-1)*2 - 2 + 2 = 26
    ((2, 16, 10, 12), (2, 2), (2, 2), (0, 0), True, (20, 24)),  # H: (10-1)*2+2, W: (12-1)*2+2
    ((2, 16, 10, 12), (3, 3), (2, 2), (1, 1), True, (19, 23)),  # H: (10-1)*2-2+3, W: (12-1)*2-2+3

    # ========== Large-scale performance test cases ==========
    # typical CNN activation map sizes and larger inputs
    ((32, 64, 56, 56),   (2, 2), (2, 2), (0, 0), False, None),
    ((32, 64, 56, 56),   (3, 3), (2, 2), (1, 1), False, None),
    ((64, 128, 28, 28),  (2, 2), (2, 2), (0, 0), False, None),
    ((64, 128, 28, 28),  (3, 3), (2, 2), (1, 1), False, None),
    ((128, 256, 14, 14), (2, 2), (2, 2), (0, 0), False, None),
    ((128, 256, 14, 14), (3, 3), (2, 2), (1, 1), False, None),
    ((256, 512, 7, 7),   (2, 2), (2, 2), (0, 0), False, None),
    ((256, 512, 7, 7),   (3, 3), (2, 2), (1, 1), False, None),

    # large inputs with explicit output_size
    ((16, 32, 64, 64), (2, 2), (2, 2), (0, 0), True,  (128, 128)),
    # H,W: (64-1)*2 - 0 + 2 = 128
    ((16, 32, 64, 64), (3, 3), (2, 2), (1, 1), True,  (127, 127)),
    # H,W: (64-1)*2 - 2 + 3 = 127
    ((8, 64, 32, 48),  (2, 2), (2, 2), (0, 0), True,  (64, 96)),
    # H: (32-1)*2+2=64, W: (48-1)*2+2=96
    ((8, 64, 32, 48),  (3, 3), (2, 2), (1, 1), True,  (63, 95)),
    # H: (32-1)*2-2+3=63, W: (48-1)*2-2+3=95

    # ========== Edge cases ==========
    ((1, 1, 1, 1), (1, 1), (1, 1), (0, 0), False, None),
    ((1, 4, 2, 2), (2, 2), (2, 2), (0, 0), True,  (4, 4)),   # H,W: (2-1)*2+2=4
    ((1, 2, 1, 8), (2, 2), (2, 2), (0, 1), False, None),
    ((1, 2, 3, 5), (2, 2), (2, 2), (1, 0), True,  (4, 10)),  # H: (3-1)*2-2+2, W: (5-1)*2+2
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-2, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]
_INDEX_DTYPE = infinicore.int64


def _kernel_elems_2d(kernel_size):
    """Return number of elements in a 2D kernel."""
    if isinstance(kernel_size, int):
        kh = kw = kernel_size
    else:
        kh, kw = kernel_size
    return int(kh * kw)


def parse_test_cases():
    """
    Parse test case data and return list of TestCase objects.

    max_unpool2d takes two inputs:
      - pooled input tensor
      - indices tensor (int64), values in [0, kh * kw)
    """
    test_cases = []

    for (input_shape, kernel_size, stride, padding,
         use_output_size, output_hw) in _TEST_CASES_DATA:

        indices_high = _kernel_elems_2d(kernel_size)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-2, "rtol": 1e-2})

            input_spec = TensorSpec.from_tensor(input_shape, None, dtype)
            indices_spec = TensorSpec.from_tensor(
                input_shape,
                None,
                _INDEX_DTYPE,
                init_mode=TensorInitializer.RANDINT,
                low=0,
                high=indices_high,
            )

            kwargs = {"kernel_size": kernel_size}
            if stride is not None:
                kwargs["stride"] = stride
            if padding is not None:
                kwargs["padding"] = padding
            if use_output_size and output_hw is not None:
                n, c, _, _ = input_shape
                h_in, w_in = output_hw
                kwargs["output_size"] = (n, c, h_in, w_in)

            test_cases.append(
                TestCase(
                    inputs=[input_spec, indices_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="MaxUnpool2d - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """MaxUnpool2d operator test with simplified implementation"""

    def __init__(self):
        super().__init__("MaxUnpool2d")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.max_unpool2d(*args, **kwargs)

    # Uncomment if InfiniCore implementation is available
    # def infinicore_operator(self, *args, **kwargs):
    #     return infinicore.nn.functional.max_unpool2d(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
