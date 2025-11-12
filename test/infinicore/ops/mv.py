import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (mat_shape, vec_shape, mat_strides_or_None, vec_strides_or_None)
# mv(mat, vec, out=None)

_TEST_CASES_DATA = [
    ((3, 4), (4,), None, None),
    ((8, 8), (8,), (512, 1), None),
    ((1, 5), (5,), None, None),
    ((6, 6), (6,), None, (0,)),
    ((12, 12), (12,), (144, 12), None),
    ((16, 8), (8,), None, None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for m_shape, v_shape, m_strides, v_strides in _TEST_CASES_DATA:
        out_supports_inplace = not is_broadcast(v_strides)
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})
            m = TensorSpec.from_tensor(m_shape, m_strides, dtype)
            v = TensorSpec.from_tensor(v_shape, v_strides, dtype)

            test_cases.append(
                TestCase(
                    inputs=[m, v],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="mv - OUT_OF_PLACE",
                )
            )

            if out_supports_inplace:
                out_spec = TensorSpec.from_tensor((m_shape[0],), None, dtype)
                test_cases.append(
                    TestCase(
                        inputs=[m, v],
                        kwargs=None,
                        output_spec=out_spec,
                        comparison_target="out",
                        tolerance=tol,
                        description="mv - INPLACE(out)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """mv operator test with simplified implementation"""

    def __init__(self):
        super().__init__("mv")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.mv(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.mv(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
