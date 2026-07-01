import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorInitializer,
    TensorSpec,
    TestCase,
)
from ops.mrope import make_positions, torch_mrope

import infinicore

_TEST_CASES_DATA = [
    # batch, seq_len, num_q_heads, num_kv_heads, head_dim, rotary_dim, mrope_section, mrope_interleaved, q_strides, k_strides
    (1, 5, 2, 1, 32, 32, (4, 6, 6), False, None, None),
    (2, 4, 3, 1, 32, 24, (2, 4, 6), False, None, None),
    (1, 6, 2, 2, 32, 24, (2, 3, 7), True, None, None),
    (1, 5, 2, 1, 32, 32, (4, 6, 6), False, (160, 1), (96, 1)),
]
_TENSOR_DTYPES = [infinicore.float16, infinicore.float32]
_POSITION_DTYPES = [infinicore.int32, infinicore.int64]
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-3, "rtol": 1e-3},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-5},
}


def create_sin_cos_table(max_position, rotary_dim, theta, device):
    pos = torch.arange(max_position, dtype=torch.float32, device=device)
    freqs = 1.0 / (
        theta
        ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float32, device=device)
            / rotary_dim
        )
    )
    angles = torch.outer(pos, freqs)
    return torch.cos(angles), torch.sin(angles)


def parse_test_cases():
    test_cases = []
    for (
        batch,
        seq_len,
        num_q_heads,
        num_kv_heads,
        head_dim,
        rotary_dim,
        mrope_section,
        mrope_interleaved,
        q_strides,
        k_strides,
    ) in _TEST_CASES_DATA:
        num_tokens = batch * seq_len
        positions = make_positions(num_tokens)
        for dtype in _TENSOR_DTYPES:
            for position_dtype in _POSITION_DTYPES:
                q_spec = TensorSpec.from_tensor(
                    (num_tokens, num_q_heads * head_dim), q_strides, dtype
                )
                k_spec = TensorSpec.from_tensor(
                    (num_tokens, num_kv_heads * head_dim), k_strides, dtype
                )
                pos_spec = TensorSpec.from_tensor(
                    positions.shape,
                    None,
                    position_dtype,
                    init_mode=TensorInitializer.MANUAL,
                    set_tensor=positions,
                )
                test_cases.append(
                    TestCase(
                        inputs=[q_spec, k_spec, pos_spec],
                        kwargs={
                            "max_position_embeddings": num_tokens * 2 + 4,
                            "rope_theta": 10000.0,
                            "head_dim": head_dim,
                            "rotary_dim": rotary_dim,
                            "mrope_section": mrope_section,
                            "mrope_interleaved": mrope_interleaved,
                        },
                        comparison_target=None,
                        tolerance=_TOLERANCE_MAP[dtype],
                        output_count=2,
                        description="nn.RoPE - MROPE_OUT_OF_PLACE",
                    )
                )
    return test_cases


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("nn.RoPE MRoPE")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(
        self,
        q,
        k,
        positions,
        max_position_embeddings,
        rope_theta,
        head_dim,
        rotary_dim,
        mrope_section,
        mrope_interleaved,
    ):
        cos, sin = create_sin_cos_table(
            max_position_embeddings, rotary_dim, rope_theta, q.device
        )
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
        return torch_mrope(
            q,
            k,
            cos,
            sin,
            positions,
            head_size=head_dim,
            rotary_dim=rotary_dim,
            section_t=mrope_section[0],
            section_h=mrope_section[1],
            section_w=mrope_section[2],
            interleaved=mrope_interleaved,
        )

    def infinicore_operator(
        self,
        q,
        k,
        positions,
        max_position_embeddings,
        rope_theta,
        head_dim,
        rotary_dim,
        mrope_section,
        mrope_interleaved,
    ):
        module = infinicore.nn.RoPE(
            max_position_embeddings,
            rope_theta,
            head_dim,
            device=q.device,
            dtype=q.dtype,
            rotary_dim=rotary_dim,
            mrope_section=mrope_section,
            mrope_interleaved=mrope_interleaved,
        )
        return module(q, k, positions)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
