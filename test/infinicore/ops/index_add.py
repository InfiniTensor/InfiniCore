import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (target_shape, target_strides_or_None, dim, index_shape, src_shape)
_TEST_CASES_DATA = [
    ((5, 6), None, 1, (5, 2), (5, 2)),
    ((4, 4), (16, 1), 0, (2, 4), (2, 4)),
    ((3, 5), None, 1, (3, 3), (3, 3)),
    ((2, 6), None, 1, (2, 2), (2, 2)),
    ((6, 3), (18, 1), 0, (3, 3), (3, 3)),
    ((4, 7), None, 1, (4, 2), (4, 2)),
]

_TOLERANCE_MAP = {infinicore.float32: {"atol": 1e-5, "rtol": 1e-4}}
_TENSOR_DTYPES = [infinicore.float32]


def parse_test_cases():
    test_cases = []
    for target_shape, t_strides, dim, idx_shape, src_shape in _TEST_CASES_DATA:
        target_spec = TensorSpec.from_tensor(
            target_shape, t_strides, infinicore.float32
        )
        # index for index_add should be 1-D with length equal to source.size(dim)
        index_len = src_shape[dim]
        from framework.tensor import TensorInitializer

        index_spec = TensorSpec.from_tensor(
            (index_len,),
            None,
            infinicore.int64,
            init_mode=TensorInitializer.RANDINT,
            low=0,
            high=target_shape[dim],
        )
        src_spec = TensorSpec.from_tensor(src_shape, None, infinicore.float32)

        # out parameter can be used (explicit out)
        out_supports = not is_broadcast(t_strides)

        # Out-of-place (return value)
        # Use positional dim to match infinicore.index_add(input, dim, index, tensor)
        test_cases.append(
            TestCase(
                inputs=[target_spec, dim, index_spec, src_spec],
                kwargs={},
                output_spec=None,
                comparison_target=None,
                tolerance=_TOLERANCE_MAP[infinicore.float32],
                description=f"index_add - OUT_OF_PLACE",
            )
        )

        # In-place on target (out=target)
        if out_supports:
            test_cases.append(
                TestCase(
                    inputs=[target_spec, dim, index_spec, src_spec],
                    kwargs=None,
                    output_spec=target_spec,
                    comparison_target="out",
                    tolerance=_TOLERANCE_MAP[infinicore.float32],
                    description=f"index_add - INPLACE(out)",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """IndexAdd operator test with simplified implementation"""

    def __init__(self):
        super().__init__("IndexAdd")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.index_add(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.index_add(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
