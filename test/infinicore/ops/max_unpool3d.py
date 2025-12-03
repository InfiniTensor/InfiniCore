import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework import BaseOperatorTest, TensorSpec, TestCase, GenericTestRunner
from framework.tensor import TensorInitializer

# ==============================================================================
# Operator-specific configuration for max_unpool3d
# ==============================================================================

# Test cases format:
# (input_shape, kernel_size, stride_or_None, padding_or_None,
#  use_output_size, output_dhw_or_None)
#
#   input_shape      : (N, C, D_out, H_out, W_out), pooled feature map
#   kernel_size      : int or (kd, kh, kw)
#   stride_or_None   : int / (sd, sh, sw) / None (None -> stride == kernel_size)
#   padding_or_None  : int / (pd, ph, pw) / None (None -> padding == 0)
#   use_output_size  : bool, whether to pass output_size explicitly
#   output_dhw_or_None: (D_in, H_in, W_in) if use_output_size is True
#
# torch.nn.functional.max_unpool3d(
#     input, indices, kernel_size, stride=None, padding=0, output_size=None
# )

_TEST_CASES_DATA = [
    # ========== Basic cases ==========
    # small sizes with different stride/padding and optional output_size
    ((1, 1, 4, 4, 4), (2, 2, 2), (2, 2, 2), (0, 0, 0), False, None),
    ((2, 3, 2, 4, 4), (2, 2, 2), None, None, False, None),  # default stride / padding
    ((2, 3, 4, 4, 4), (3, 3, 3), (2, 2, 2), (1, 1, 1), False, None),
    ((2, 3, 4, 4, 4), (3, 3, 3), (2, 2, 2), (1, 1, 1), True, (7, 7, 7)),
    ((1, 4, 3, 5, 7), (2, 2, 2), (2, 2, 2), (0, 0, 0), False, None),
    ((1, 4, 3, 5, 7), (2, 2, 2), None, (1, 1, 1), True, (4, 8, 12)),
    # ========== Large-scale performance test cases ==========
    # larger volumes and batches for performance and stability
    ((4, 8, 8, 16, 16), (2, 2, 2), (2, 2, 2), (0, 0, 0), False, None),
    ((4, 8, 8, 16, 16), (3, 3, 3), (2, 2, 2), (1, 1, 1), False, None),
    ((8, 16, 4, 32, 32), (2, 2, 2), (2, 2, 2), (0, 0, 0), False, None),
    ((8, 16, 4, 32, 32), (2, 2, 2), (2, 2, 2), (0, 0, 0), True, (8, 64, 64)),
    ((8, 16, 4, 32, 32), (3, 3, 3), (2, 2, 2), (1, 1, 1), True, (7, 63, 63)),
    ((2, 32, 16, 16, 16), (2, 2, 2), (2, 2, 2), (0, 0, 0), False, None),
    ((2, 32, 16, 16, 16), (2, 2, 2), None, None, False, None),
    ((2, 32, 8, 32, 32), (2, 2, 2), (2, 2, 2), (0, 0, 0), False, None),
    ((2, 32, 8, 32, 32), (3, 3, 3), (2, 2, 2), (1, 1, 1), False, None),
    ((1, 64, 8, 64, 64), (2, 2, 2), (2, 2, 2), (0, 0, 0), False, None),
    ((1, 64, 8, 64, 64), (2, 2, 2), (2, 2, 2), (0, 0, 0), True, (16, 128, 128)),
    ((1, 64, 4, 64, 128), (2, 2, 2), (2, 2, 2), (0, 0, 0), False, None),
    # ========== Edge cases ==========
    # very small shapes and asymmetric sizes
    ((1, 1, 1, 1, 1), (1, 1, 1), (1, 1, 1), (0, 0, 0), False, None),
    ((1, 1, 1, 2, 2), (2, 2, 2), (2, 2, 2), (0, 0, 0), True, (2, 4, 4)),
    ((1, 1, 2, 2, 8), (2, 2, 2), (2, 2, 2), (0, 0, 0), False, None),
    ((1, 1, 2, 2, 8), (2, 2, 2), (2, 2, 2), (1, 1, 1), True, (2, 2, 14)),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-2, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]
_INDEX_DTYPE = infinicore.int64


def _kernel_elems_3d(kernel_size):
    """Return number of elements in a 3D kernel."""
    if isinstance(kernel_size, int):
        kd = kh = kw = kernel_size
    else:
        kd, kh, kw = kernel_size
    return int(kd * kh * kw)


def parse_test_cases():
    """
    Parse test case data and return list of TestCase objects.

    max_unpool3d takes two inputs:
      - pooled input tensor
      - indices tensor (int64), values in [0, kd * kh * kw)
    """
    test_cases = []

    for (
        input_shape,
        kernel_size,
        stride,
        padding,
        use_output_size,
        output_dhw,
    ) in _TEST_CASES_DATA:

        indices_high = _kernel_elems_3d(kernel_size)

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
            if use_output_size and output_dhw is not None:
                n, c, _, _, _ = input_shape
                d_in, h_in, w_in = output_dhw
                kwargs["output_size"] = (n, c, d_in, h_in, w_in)

            test_cases.append(
                TestCase(
                    inputs=[input_spec, indices_spec],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="MaxUnpool3d - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """MaxUnpool3d operator test with simplified implementation"""

    def __init__(self):
        super().__init__("MaxUnpool3d")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.max_unpool3d(*args, **kwargs)

    # Uncomment if InfiniCore implementation is available
    # def infinicore_operator(self, *args, **kwargs):
    #     return infinicore.nn.functional.max_unpool3d(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
