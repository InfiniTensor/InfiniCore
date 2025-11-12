import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.tensor import TensorInitializer

# Test cases format: (input_shape_logits_N_C, target_shape_N, input_strides_or_None, weight_present_bool, ignore_index_or_None)
# infinicore.nn.functional.cross_entropy(input, target, weight=None, ignore_index=-100, reduction='mean')

_TEST_CASES_DATA = [
    ((4, 5), (4,), None, False, None),
    ((8, 10), (8,), None, True, -1),
    ((1, 3), (1,), None, False, None),
    ((16, 100), (16,), (1600, 100), True, None),
    ((3, 7), (3,), None, False, None),
    ((2, 2), (2,), None, True, -100),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for (
        logits_shape,
        target_shape,
        logits_strides,
        weight_present,
        ignore_index,
    ) in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            logits = TensorSpec.from_tensor(logits_shape, logits_strides, dtype)
            target = TensorSpec.from_tensor(
                target_shape,
                None,
                infinicore.int64,
                init_mode=TensorInitializer.RANDINT,
                low=0,
                high=logits_shape[1],
            )

            inputs = [logits, target]
            kwargs = {}
            if weight_present:
                weight_spec = TensorSpec.from_tensor((logits_shape[1],), None, dtype)
                inputs.append(weight_spec)
            if ignore_index is not None:
                kwargs["ignore_index"] = ignore_index

            test_cases.append(
                TestCase(
                    inputs=inputs,
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="cross_entropy - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """cross_entropy operator test with simplified implementation"""

    def __init__(self):
        super().__init__("cross_entropy")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.nn.functional.cross_entropy(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.nn.functional.cross_entropy(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
