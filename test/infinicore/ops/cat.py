import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework.base import BaseOperatorTest, TensorSpec, TestCase
from framework.runner import GenericTestRunner
from framework.utils import is_broadcast

# ==============================================================================
# Operator-specific configuration
# ==============================================================================

# Test cases format: (tensor_shapes, dim, input_strides_list, output_strides)
_TEST_CASES_DATA = [
    # Basic concatenation
    ([(2, 3), (2, 3)], 0, None, None),
    ([(2, 3), (2, 3)], 1, None, None),
    ([(1, 4), (3, 4)], 0, None, None),
    # Multiple tensors
    ([(1, 5), (2, 5), (3, 5)], 0, None, None),
    ([(3, 2), (3, 3), (3, 1)], 1, None, None),
    # 3D tensors
    ([(2, 3, 4), (2, 3, 4)], 0, None, None),
    ([(2, 3, 4), (2, 3, 4)], 1, None, None),
    ([(2, 3, 4), (2, 3, 4)], 2, None, None),
    # Strided tensors
    ([(3, 4), (3, 4)], 0, [(8, 1), (8, 1)], None),
    ([(2, 5), (2, 5)], 1, [(10, 1), (10, 1)], None),
    # Large tensors
    ([(16, 256), (16, 256)], 0, None, None),
    ([(8, 512), (8, 512)], 1, None, None),
]

# Tolerance configuration
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 0, "rtol": 1e-2},
    infinicore.float32: {"atol": 0, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 0, "rtol": 5e-2},
}

# Data types to test
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    """
    Parse cat test case data and return list of TestCase objects.
    """
    test_cases = []

    for data in _TEST_CASES_DATA:
        tensor_shapes = data[0]
        dim = data[1]
        input_strides_list = data[2] if len(data) > 2 else None
        output_strides = data[3] if len(data) > 3 else None

        # Calculate output shape
        output_shape = list(tensor_shapes[0])
        for shape in tensor_shapes[1:]:
            output_shape[dim] += shape[dim]

        # Check if output supports in-place
        output_supports_inplace = not is_broadcast(output_strides)

        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 0, "rtol": 1e-3})

            # Create input tensor specs as tuple
            input_specs = []
            for i, shape in enumerate(tensor_shapes):
                strides = (
                    input_strides_list[i]
                    if input_strides_list and i < len(input_strides_list)
                    else None
                )
                input_specs.append(TensorSpec.from_tensor(shape, strides, dtype))

            # Create output tensor spec
            output_spec = TensorSpec.from_tensor(output_shape, output_strides, dtype)

            # Out-of-place test case
            test_cases.append(
                TestCase(
                    inputs=[tuple(input_specs)],
                    kwargs={"dim": dim},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description="Cat - OUT_OF_PLACE",
                )
            )

            # In-place test case
            if output_supports_inplace:
                test_cases.append(
                    TestCase(
                        inputs=[tuple(input_specs)],
                        kwargs={"dim": dim},
                        output_spec=output_spec,
                        comparison_target="out",
                        tolerance=tolerance,
                        description="Cat - INPLACE(out)",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """Cat operator test implementation"""

    def __init__(self):
        super().__init__("Cat")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        """PyTorch cat implementation"""
        return torch.cat(*args, **kwargs)

    # def infinicore_operator(self, *args, **kwargs):
    #     """InfiniCore cat implementation"""
    #     return infinicore.cat(*args, **kwargs)


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
