import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework import BaseOperatorTest, TensorSpec, TestCase, GenericTestRunner

# Test cases format: (in_shape, proj_w_shape)
_TEST_CASES_DATA = [
    ((32, 4096), (4096, 4096)),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-4, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}
_TENSOR_DTYPES = [infinicore.float16, infinicore.float32, infinicore.bfloat16]


def parse_test_cases():
    cases = []
    for in_shape, proj_w_shape in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP[dtype]
            in_spec = TensorSpec.from_tensor(in_shape, dtype=dtype)
            proj_w_spec = TensorSpec.from_tensor(proj_w_shape, dtype=dtype)
            temp_spec = TensorSpec.from_tensor(in_shape, dtype=dtype)

            # Out-of-place
            cases.append(
                TestCase(
                    inputs=[in_spec, proj_w_spec, temp_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="Graph",
                )
            )

    return cases


class OpTest(BaseOperatorTest):
    """Test Operator Graph"""

    def __init__(self):
        super().__init__("Graph")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        a = args[0]
        b = args[1]

        return torch.matmul(a, b)

    def infinicore_operator(self, *args, **kwargs):
        """Record graph and run"""
        a = args[0]
        b = args[1]
        temp_a = args[2]

        infinicore.start_graph_recording()
        c = infinicore.matmul(temp_a, b)
        op_graph = infinicore.stop_graph_recording()

        temp_a.copy_(a)
        op_graph.run()

        return c


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
