import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (target_shape, target_strides_or_None, dim, index_shape, reduce)
_TEST_CASES_DATA = [
    ((5, 6), None, 1, (5, 2), "prod"),
    ((4, 4), (16, 1), 0, (2, 4), "amax"),
    ((3, 5), None, 1, (3, 3), "amin"),
    ((2, 6), None, 1, (2, 2), "prod"),
    ((2, 6), None, 1, (2, 2), "amin"),
    ((6, 3), (18, 1), 0, (3, 3), "mean"),
    ((4, 7), None, 1, (4, 2), "prod"),
]

_TOLERANCE_MAP = {infinicore.float32: {"atol": 1e-5, "rtol": 1e-4}}
_TENSOR_DTYPES = [infinicore.float32]


def parse_test_cases():
    test_cases = []
    for tgt_shape, tgt_strides, dim, idx_shape, reduce in _TEST_CASES_DATA:
        target_spec = TensorSpec.from_tensor(tgt_shape, tgt_strides, infinicore.float32)
        # idx_shape here represents the source shape for index_reduce; index itself must be 1-D
        src_shape = idx_shape
        # determine index length from source along the reduction dim
        index_len = src_shape[dim]
        from framework.tensor import TensorInitializer

        index_spec = TensorSpec.from_tensor(
            (index_len,),
            None,
            infinicore.int64,
            init_mode=TensorInitializer.RANDINT,
            low=0,
            high=tgt_shape[dim],
        )

        src_spec = TensorSpec.from_tensor(src_shape, None, infinicore.float32)

        out_supports = not is_broadcast(tgt_strides)

        kwargs = {"reduce": reduce}

        test_cases.append(
            TestCase(
                inputs=[target_spec, dim, index_spec, src_spec],
                kwargs=kwargs,
                output_spec=None,
                comparison_target=None,
                tolerance=_TOLERANCE_MAP[infinicore.float32],
                description=f"index_reduce - OUT_OF_PLACE",
            )
        )

        if out_supports:
            test_cases.append(
                TestCase(
                    inputs=[target_spec, dim, index_spec, src_spec],
                    kwargs=kwargs,
                    output_spec=target_spec,
                    comparison_target="out",
                    tolerance=_TOLERANCE_MAP[infinicore.float32],
                    description=f"index_reduce - INPLACE(out)",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """IndexReduce operator test with simplified implementation"""

    def __init__(self):
        super().__init__("IndexReduce")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch.index_reduce(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.index_reduce(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
