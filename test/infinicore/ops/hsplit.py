import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner

# Test cases format: (input_shape, input_strides_or_None, sections_or_None)
# infinicore.hsplit(input, sections)
# Note: PyTorch hsplit is a convenience wrapper around split/reshape. We include both int and list sections.

_TEST_CASES_DATA = [
    ((4, 8), None, 2),
    ((4, 9), None, [3, 6]),
    ((2, 6, 12), None, 3),
    ((1, 10), (10, 1), 5),
    ((8, 4), None, [1, 3]),
    ((6, 12), None, 4),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 5e-2},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    test_cases = []
    for shape, strides, sections in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})
            inp = TensorSpec.from_tensor(shape, strides, dtype)
            # infinicore.hsplit expects positional second arg (sections) rather than kw in this API;
            # put sections into inputs when present. Convert list -> tuple.
            if sections is not None:
                # wrap sections in a small wrapper so TestCase.__init__ does not
                # interpret the tuple as a Tensor shape
                class Sections:
                    def __init__(self, v):
                        self.v = v

                    def as_tuple(self):
                        return tuple(self.v) if isinstance(self.v, list) else self.v

                    def __repr__(self):
                        return f"sections({self.v})"

                sec = Sections(sections)
                test_inputs = [inp, sec]
            else:
                test_inputs = [inp]

            test_cases.append(
                TestCase(
                    inputs=test_inputs,
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="hsplit - OUT_OF_PLACE",
                )
            )

    return test_cases


class OpTest(BaseOperatorTest):
    """hsplit operator test with simplified implementation"""

    def __init__(self):
        super().__init__("hsplit")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        # unwrap Sections wrapper if present
        args = list(args)
        if len(args) >= 2 and hasattr(args[1], "as_tuple"):
            args[1] = args[1].as_tuple()
        return torch.hsplit(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore implementation (operator not yet available)."""
    #     return infinicore.hsplit(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
