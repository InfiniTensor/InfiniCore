import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework import (
    BaseOperatorTest,
    TensorSpec,
    TestCase,
    GenericTestRunner,
    is_broadcast,
)

# ==============================================================================
# Operator-specific configuration
# ==============================================================================

# Test cases format: (input_shape, input_strides, output_strides)
# SiLUAndMul: Input (..., 2*d) -> Output (..., d)
_TEST_CASES_DATA = [
    # Basic 2D: [2, 4] -> [2, 2]
    ((2, 4), None, None),
    # 2D Large: [1024, 1024] -> [1024, 512]
    ((1024, 1024), None, None),
    # 3D: [2, 4, 8] -> [2, 4, 4]
    ((2, 4, 8), None, None),
    # LLM typical hidden size (e.g., Llama SwiGLU: intermediate_size=11008)
    # [1, 11008*2] -> [1, 11008]
    ((1, 22016), None, None),
    # Strided tensors
    ((2, 4, 256), None, None),
]

# Tolerance configuration
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
    infinicore.bfloat16: {"atol": 5e-3, "rtol": 1e-2},
}

# Data types to test
_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def parse_test_cases():
    """
    Parse SiLUAndMul test case data.
    Input: [..., 2*d], Output: [..., d]
    """
    test_cases = []

    for data in _TEST_CASES_DATA:
        input_shape = data[0]
        input_strides = data[1] if len(data) > 1 else None
        output_strides = data[2] if len(data) > 2 else None

        # 推导输出形状：最后一维减半
        output_shape = list(input_shape)
        output_shape[-1] //= 2
        output_shape = tuple(output_shape)

        # SiLUAndMul 不支持原地 (In-place on input)，因为形状不匹配
        # 但支持指定输出 Tensor (out=output)
        output_supports_explicit_out = not is_broadcast(output_strides)

        for dtype in _TENSOR_DTYPES:
            tolerance = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-4})

            input_spec = TensorSpec.from_tensor(input_shape, input_strides, dtype)
            output_spec = TensorSpec.from_tensor(output_shape, output_strides, dtype)

            # Test Case 1: Out-of-place (return value)
            test_cases.append(
                TestCase(
                    inputs=[input_spec],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tolerance,
                    description=f"SiLUAndMul - OUT_OF_PLACE",
                )
            )

            # Test Case 2: Explicit output tensor (silu_and_mul(input, out=output))
            if output_supports_explicit_out:
                test_cases.append(
                    TestCase(
                        inputs=[input_spec],
                        kwargs=None,
                        output_spec=output_spec,
                        comparison_target="out",
                        tolerance=tolerance,
                        description=f"SiLUAndMul - OUT_PARAM",
                    )
                )

    return test_cases


class OpTest(BaseOperatorTest):
    """SiLUAndMul operator test (SwiGLU)"""

    def __init__(self):
        super().__init__("SiLUAndMul")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, input, out=None, **kwargs):
        """PyTorch SwiGLU reference: SiLU(x_gate) * x_up"""
        d = input.shape[-1] // 2
        # 将最后一维切分为两部分
        gate, up = torch.split(input, [d, d], dim=-1)
        result = torch.nn.functional.silu(gate) * up

        if out is not None:
            out.copy_(result)
            return out
        return result

    def infinicore_operator(self, input, out=None, **kwargs):
        """InfiniCore SiLUAndMul implementation"""
    
        import infinicore.nn.functional as F

        return F.silu_and_mul(input, out=out)



def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
