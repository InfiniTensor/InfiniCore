import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (start_shape, start_strides_or_None, end_shape_or_None, weight_scalar_or_None, weight_tensor_shape_or_None)
# infinicore.lerp(start, end, weight)

_TEST_CASES_DATA = [
    ((2, 3, 4), None, None, 0.5, None),
    ((1, 4, 8), (32, 8, 1), None, None, (1, 4, 8)),
    ((3, 2, 5, 7), None, None, 0.25, None),
    ((2, 1, 16), None, None, None, (2, 1, 16)),
    ((1, 8, 9, 11), (792, 99, 11, 1), None, 0.75, None),
    ((2, 6, 10), None, None, None, (2, 6, 10)),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
}
_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]


def parse_test_cases():
    cases = []
    for (
        start_shape,
        start_strides,
        end_shape,
        weight_scalar,
        weight_tensor_shape,
    ) in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP[dtype]
            start_spec = TensorSpec.from_tensor(start_shape, start_strides, dtype)
            end_spec = TensorSpec.from_tensor(
                start_shape if end_shape is None else end_shape, None, dtype
            )

            if weight_scalar is not None:
                weight = weight_scalar
                cases.append(
                    TestCase(
                        inputs=[start_spec, end_spec, weight],
                        kwargs={},
                        output_spec=None,
                        comparison_target=None,
                        tolerance=tol,
                        description="lerp_scalar_weight_out",
                    )
                )
                out_spec = TensorSpec.from_tensor(start_shape, None, dtype)
                cases.append(
                    TestCase(
                        inputs=[start_spec, end_spec, weight],
                        kwargs={},
                        output_spec=out_spec,
                        comparison_target="out",
                        tolerance=tol,
                        description="lerp_scalar_weight_explicit_out",
                    )
                )
                if not is_broadcast(start_spec.strides):
                    cases.append(
                        TestCase(
                            inputs=[start_spec, end_spec, weight],
                            kwargs={"out": 0},
                            output_spec=None,
                            comparison_target=0,
                            tolerance=tol,
                            description="lerp_scalar_inplace_start",
                        )
                    )
                if not is_broadcast(end_spec.strides):
                    cases.append(
                        TestCase(
                            inputs=[start_spec, end_spec, weight],
                            kwargs={"out": 1},
                            output_spec=None,
                            comparison_target=1,
                            tolerance=tol,
                            description="lerp_scalar_inplace_end",
                        )
                    )

            if weight_tensor_shape is not None:
                weight_spec = TensorSpec.from_tensor(weight_tensor_shape, None, dtype)
                cases.append(
                    TestCase(
                        inputs=[start_spec, end_spec, weight_spec],
                        kwargs={},
                        output_spec=None,
                        comparison_target=None,
                        tolerance=tol,
                        description="lerp_tensor_weight_out",
                    )
                )
                out_spec = TensorSpec.from_tensor(start_shape, None, dtype)
                cases.append(
                    TestCase(
                        inputs=[start_spec, end_spec, weight_spec],
                        kwargs={},
                        output_spec=out_spec,
                        comparison_target="out",
                        tolerance=tol,
                        description="lerp_tensor_weight_explicit_out",
                    )
                )
                if not is_broadcast(weight_spec.strides):
                    cases.append(
                        TestCase(
                            inputs=[start_spec, end_spec, weight_spec],
                            kwargs={"out": 2},
                            output_spec=None,
                            comparison_target=2,
                            tolerance=tol,
                            description="lerp_inplace_weight",
                        )
                    )

    return cases


class OpTest(BaseOperatorTest):
    """Lerp operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Lerp")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.lerp(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.lerp(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
