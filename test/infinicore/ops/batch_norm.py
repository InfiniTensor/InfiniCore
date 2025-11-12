import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (input_shape, input_strides_or_None, running_mean_present_bool, running_var_present_bool, weight_bias_present_bool, training_or_None, momentum_or_None, eps_or_None)
_TEST_CASES_DATA = [
    ((4, 3, 8, 8), None, True, True, True, False, None, None),
    ((2, 6, 4, 4), (384, 96, 1, 1), True, True, False, True, 0.2, 1e-5),
    ((1, 3, 16, 16), None, True, True, True, False, None, None),
    ((8, 5, 2, 2), None, True, True, True, False, 0.1, 1e-3),
    ((6, 4, 7, 7), None, False, False, True, True, None, 1e-4),
    ((3, 2, 9, 9), None, True, True, False, False, None, None),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-2, "rtol": 1e-1},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for (
        shape,
        strides,
        mean_p,
        var_p,
        wb_p,
        training,
        momentum,
        eps,
    ) in _TEST_CASES_DATA:
        C = shape[1]
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            inp = TensorSpec.from_tensor(shape, strides, dtype)

            running_mean = TensorSpec.from_tensor((C,), None, dtype) if mean_p else None
            running_var = TensorSpec.from_tensor((C,), None, dtype) if var_p else None
            inputs = [inp]
            kwargs = {}
            if running_mean is not None:
                inputs.append(running_mean)
            else:
                inputs.append(None)
            if running_var is not None:
                inputs.append(running_var)
            else:
                inputs.append(None)
            if wb_p:
                weight = TensorSpec.from_tensor((C,), None, dtype)
                bias = TensorSpec.from_tensor((C,), None, dtype)
                inputs.append(weight)
                inputs.append(bias)
            else:
                inputs.append(None)
                inputs.append(None)

            if training is not None:
                kwargs["training"] = training
            if momentum is not None:
                kwargs["momentum"] = momentum
            if eps is not None:
                kwargs["eps"] = eps

            test_cases.append(
                TestCase(
                    inputs=inputs,
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="batch_norm - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """batch_norm operator test with simplified implementation"""

    def __init__(self):
        super().__init__("batch_norm")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.batch_norm(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.batch_norm(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
