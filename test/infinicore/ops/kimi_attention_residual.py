import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase
from infinicore.nn.functional import kimi_attention_residual

import infinicore

_TEST_CASES = [
    (1, 2, 128),
    (4, 3, 512),
    (8, 7, 7168),
    (64, 4, 7168),
]

_DTYPES = [infinicore.float16, infinicore.bfloat16]


def build_test_cases():
    cases = []
    for num_tokens, num_sources, hidden_size in _TEST_CASES:
        for dtype in _DTYPES:
            values = TensorSpec.from_tensor(
                (num_tokens, num_sources, hidden_size), None, dtype
            )
            norm_weight = TensorSpec.from_tensor((hidden_size,), None, dtype)
            projection_weight = TensorSpec.from_tensor((1, hidden_size), None, dtype)
            cases.append(
                TestCase(
                    inputs=[values, norm_weight, projection_weight],
                    kwargs={"epsilon": 1e-6},
                    output_spec=None,
                    comparison_target=None,
                    tolerance={"atol": 3e-2, "rtol": 3e-2},
                    description=(
                        "KimiAttentionResidual - "
                        f"tokens={num_tokens}, sources={num_sources}, hidden={hidden_size}"
                    ),
                )
            )
    return cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("KimiAttentionResidual")

    def get_test_cases(self):
        return build_test_cases()

    def torch_operator(
        self, values, norm_weight, projection_weight, epsilon=1e-6, out=None
    ):
        values_float = values.float()
        inverse_rms = torch.rsqrt(
            values_float.square().mean(dim=-1, keepdim=True) + epsilon
        )
        combined_weight = norm_weight.float() * projection_weight.float().view(-1)
        logits = (values_float * inverse_rms * combined_weight).sum(dim=-1)
        probabilities = torch.softmax(logits, dim=1)
        result = (values_float * probabilities.unsqueeze(-1)).sum(dim=1).to(
            values.dtype
        )
        if out is not None:
            out.copy_(result)
            return out
        return result

    def infinicore_operator(
        self, values, norm_weight, projection_weight, epsilon=1e-6, out=None
    ):
        if out is not None:
            raise ValueError("KimiAttentionResidual does not support an out tensor")
        return kimi_attention_residual(
            values,
            norm_weight,
            projection_weight,
            epsilon,
        )


def main():
    GenericTestRunner(OpTest).run_and_exit()


if __name__ == "__main__":
    main()
