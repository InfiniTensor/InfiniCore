import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import infinicore
from framework import BaseOperatorTest, GenericTestRunner, TensorInitializer, TensorSpec, TestCase


_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16]
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 5e-2, "rtol": 5e-2},
    infinicore.bfloat16: {"atol": 8e-2, "rtol": 8e-2},
}


def _make_aligned_metadata(topk_ids: torch.Tensor, num_experts: int, block_size: int):
    sorted_token_ids = []
    expert_ids = []
    flat = topk_ids.reshape(-1).tolist()
    for expert in range(num_experts):
        pairs = [idx for idx, value in enumerate(flat) if value == expert]
        if not pairs:
            continue
        padded = ((len(pairs) + block_size - 1) // block_size) * block_size
        for begin in range(0, padded, block_size):
            block = pairs[begin : begin + block_size]
            sorted_token_ids.extend(block + [-1] * (block_size - len(block)))
            expert_ids.append(expert)
    return (
        torch.tensor(sorted_token_ids, dtype=torch.int32),
        torch.tensor(expert_ids, dtype=torch.int32),
        torch.tensor([len(sorted_token_ids)], dtype=torch.int32),
    )


def _make_case(seed, num_tokens, hidden_size, intermediate_size, num_experts, topk, topk_ids, block_size):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    hidden_states = torch.randn(num_tokens, hidden_size, generator=generator, dtype=torch.float32) * 0.4
    w13 = torch.randn(num_experts, intermediate_size * 2, hidden_size, generator=generator, dtype=torch.float32) * 0.15
    w2 = torch.randn(num_experts, hidden_size, intermediate_size, generator=generator, dtype=torch.float32) * 0.15
    raw_weights = torch.rand(num_tokens, topk, generator=generator, dtype=torch.float32)
    topk_weights = raw_weights / raw_weights.sum(dim=-1, keepdim=True)
    sorted_token_ids, expert_ids, num_tokens_post_padded = _make_aligned_metadata(topk_ids, num_experts, block_size)
    return hidden_states, w13, w2, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded


def torch_moe_fused_dense(hidden_states, w13, w2, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded):
    del sorted_token_ids, expert_ids, num_tokens_post_padded
    hidden = hidden_states.float()
    w13f = w13.float()
    w2f = w2.float()
    weights = topk_weights.float()
    ids = topk_ids.to(torch.int64)
    out = torch.zeros_like(hidden)
    num_tokens, topk = ids.shape
    for token in range(num_tokens):
        for route in range(topk):
            expert = int(ids[token, route].item())
            gate_up = F.linear(hidden[token], w13f[expert])
            gate, up = gate_up.chunk(2, dim=-1)
            activated = F.silu(gate) * up
            expert_out = F.linear(activated, w2f[expert])
            out[token] += expert_out * weights[token, route]
    return out.to(hidden_states.dtype)


def parse_test_cases():
    raw_cases = [
        (
            "decode aligned sparse experts",
            _make_case(
                20260724,
                num_tokens=1,
                hidden_size=32,
                intermediate_size=64,
                num_experts=8,
                topk=3,
                topk_ids=torch.tensor([[6, 2, 4]], dtype=torch.int32),
                block_size=4,
            ),
        ),
        (
            "prefill aligned missing experts",
            _make_case(
                20260725,
                num_tokens=5,
                hidden_size=32,
                intermediate_size=64,
                num_experts=8,
                topk=2,
                topk_ids=torch.tensor([[0, 2], [2, 5], [5, 0], [2, 0], [5, 2]], dtype=torch.int32),
                block_size=4,
            ),
        ),
        (
            "prefill aligned uneven blocks",
            _make_case(
                20260726,
                num_tokens=7,
                hidden_size=32,
                intermediate_size=64,
                num_experts=6,
                topk=2,
                topk_ids=torch.tensor([[1, 3], [1, 4], [3, 1], [4, 3], [1, 5], [5, 3], [4, 1]], dtype=torch.int32),
                block_size=3,
            ),
        ),
    ]

    tests = []
    for description, tensors in raw_cases:
        hidden_states, w13, w2, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded = tensors
        for dtype in _TENSOR_DTYPES:
            tests.append(
                TestCase(
                    inputs=[
                        TensorSpec.from_tensor(tuple(hidden_states.shape), None, dtype, init_mode=TensorInitializer.MANUAL, set_tensor=hidden_states, name="hidden_states"),
                        TensorSpec.from_tensor(tuple(w13.shape), None, dtype, init_mode=TensorInitializer.MANUAL, set_tensor=w13, name="w13"),
                        TensorSpec.from_tensor(tuple(w2.shape), None, dtype, init_mode=TensorInitializer.MANUAL, set_tensor=w2, name="w2"),
                        TensorSpec.from_tensor(tuple(topk_weights.shape), None, infinicore.float32, init_mode=TensorInitializer.MANUAL, set_tensor=topk_weights, name="topk_weights"),
                        TensorSpec.from_tensor(tuple(topk_ids.shape), None, infinicore.int32, init_mode=TensorInitializer.MANUAL, set_tensor=topk_ids, name="topk_ids"),
                        TensorSpec.from_tensor(tuple(sorted_token_ids.shape), None, infinicore.int32, init_mode=TensorInitializer.MANUAL, set_tensor=sorted_token_ids, name="sorted_token_ids"),
                        TensorSpec.from_tensor(tuple(expert_ids.shape), None, infinicore.int32, init_mode=TensorInitializer.MANUAL, set_tensor=expert_ids, name="expert_ids"),
                        TensorSpec.from_tensor(tuple(num_tokens_post_padded.shape), None, infinicore.int32, init_mode=TensorInitializer.MANUAL, set_tensor=num_tokens_post_padded, name="num_tokens_post_padded"),
                    ],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=_TOLERANCE_MAP[dtype],
                    description=f"moe_fused_dense - {description} - dtype={dtype}",
                )
            )
    return tests


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("moe_fused_dense")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_moe_fused_dense(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.nn.functional.moe_fused_dense(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()