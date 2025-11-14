import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.tensor import TensorInitializer
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# Test cases format: (input_shape, input_strides_or_None, dim, index_shape, src_shape)
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
    for tgt_shape, tgt_strides, dim, idx_shape, src_shape in _TEST_CASES_DATA:
        tgt_spec = TensorSpec.from_tensor(tgt_shape, tgt_strides, infinicore.float32)
        # initialize index tensor within [0, size) for the target dim to avoid OOB
        effective_dim = dim if dim >= 0 else (dim + len(tgt_shape))
        max_index = tgt_shape[effective_dim]
        idx_spec = TensorSpec.from_tensor(
            idx_shape,
            None,
            infinicore.int64,
            init_mode=TensorInitializer.RANDINT,
            low=0,
            high=max_index,
        )
        src_spec = TensorSpec.from_tensor(src_shape, None, infinicore.float32)

        out_supports = not is_broadcast(tgt_strides)

        # Out-of-place
        test_cases.append(
            TestCase(
                inputs=[tgt_spec, idx_spec, src_spec],
                kwargs={"dim": dim},
                output_spec=None,
                comparison_target=None,
                tolerance=_TOLERANCE_MAP[infinicore.float32],
                description=f"scatter - OUT_OF_PLACE",
            )
        )

    return test_cases


class OpTest(BaseOperatorTest):
    """Scatter operator test with simplified implementation"""

    def __init__(self):
        super().__init__("Scatter")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        if len(args) < 3:
            raise TypeError("scatter requires input, index, src as positional args")
        inp, idx, src = args[0], args[1], args[2]

        dim = None
        if kwargs:
            dim = kwargs.get("dim", None)

        if dim is None:
            raise TypeError("scatter test did not provide 'dim' parameter")

        return torch.scatter(inp, dim, idx, src)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.scatter(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
