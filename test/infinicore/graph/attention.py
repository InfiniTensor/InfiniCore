import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase
from framework.tensor import TensorInitializer

import infinicore

# Test cases format: (nlayers, batch_size, hidden_size, nhead, nkvhead, dim, seqlen, past_seqlen, max_seqlen)
_TEST_CASES_DATA = [
    (28, 1, 3584, 28, 28, 128, 1, 256, 512),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-4, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-4, "rtol": 1e-3},
    infinicore.bfloat16: {"atol": 1e-4, "rtol": 5e-2},
}
_TENSOR_DTYPES = [infinicore.float16, infinicore.float32, infinicore.bfloat16]


def parse_test_cases():
    cases = []
    for (
        nlayers,
        batch_size,
        hidden_size,
        nhead,
        nkvhead,
        dim,
        seqlen,
        past_seqlen,
        max_seqlen,
    ) in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP[dtype]
            hidden_states = TensorSpec.from_tensor(
                (batch_size, seqlen, hidden_size), dtype=dtype, scale=1e-1, bias=-5e-2
            )
            pos_ids = TensorSpec.from_tensor(
                (batch_size, seqlen),
                dtype=infinicore.int64,
                init_mode=TensorInitializer.RANDINT,
                low=0,
                high=max_seqlen,
            )
            k_cache = TensorSpec.from_tensor(
                (nlayers, batch_size, nkvhead, max_seqlen, dim),
                dtype=dtype,
                scale=1e-1,
                bias=-5e-2,
            )
            v_cache = TensorSpec.from_tensor(
                (nlayers, batch_size, nkvhead, max_seqlen, dim),
                dtype=dtype,
                scale=1e-1,
                bias=-5e-2,
            )
            q_proj_w = TensorSpec.from_tensor(
                (nhead * dim, hidden_size), dtype=dtype, scale=1e-1, bias=-5e-2
            )
            k_proj_w = TensorSpec.from_tensor(
                (nkvhead * dim, hidden_size), dtype=dtype, scale=1e-1, bias=-5e-2
            )
            v_proj_w = TensorSpec.from_tensor(
                (nkvhead * dim, hidden_size), dtype=dtype, scale=1e-1, bias=-5e-2
            )
            o_proj_w = TensorSpec.from_tensor(
                (hidden_size, nhead * dim), dtype=dtype, scale=1e-1, bias=-5e-2
            )
            norm_w = TensorSpec.from_tensor(
                (hidden_size,), dtype=dtype, scale=1e-1, bias=-5e-2
            )
            sin_table = TensorSpec.from_tensor(
                (max_seqlen, dim // 2), dtype=dtype, scale=1e-1, bias=-5e-2
            )
            cos_table = TensorSpec.from_tensor(
                (max_seqlen, dim // 2), dtype=dtype, scale=1e-1, bias=-5e-2
            )

            # Out-of-place
            cases.append(
                TestCase(
                    inputs=[
                        hidden_states,
                        pos_ids,
                        nhead,
                        nkvhead,
                        dim,
                        past_seqlen,
                        nlayers,
                        k_cache,
                        v_cache,
                        q_proj_w,
                        k_proj_w,
                        v_proj_w,
                        o_proj_w,
                        norm_w,
                        sin_table,
                        cos_table,
                    ],
                    kwargs={},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="Graph",
                )
            )

    return cases


def torch_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    sin: torch.Tensor,
    cos: torch.Tensor,
    pos_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    q, k:      [B, H, S, D]
    sin, cos:  [max_S, D//2]
    pos_ids:   [B, S]
    """

    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        # x: [..., head_dim]
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    B, H, S, D = q.shape
    assert D % 2 == 0

    # Gather sin/cos by position
    # -> [B, S, D//2]
    sin = sin[pos_ids]
    cos = cos[pos_ids]

    # Expand to broadcast over heads
    # -> [B, 1, S, D//2]
    sin = sin.unsqueeze(1)
    cos = cos.unsqueeze(1)

    # Interleave to full dim
    sin = torch.repeat_interleave(sin, 2, dim=-1)
    cos = torch.repeat_interleave(cos, 2, dim=-1)

    # Apply RoPE
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)

    return q_rot, k_rot


class OpTest(BaseOperatorTest):
    """Test Operator Graph"""

    def __init__(self):
        super().__init__("Graph")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(
        self,
        hidden_states,
        pos_ids,
        nhead,
        nkvhead,
        dim,
        past_seqlen,
        nlayers,
        k_cache,
        v_cache,
        q_proj_w,
        k_proj_w,
        v_proj_w,
        o_proj_w,
        norm_w,
        sin_table,
        cos_table,
        **kwargs,
    ):
        B, S, D = hidden_states.shape

        for layer in range(nlayers):
            # ---- RMSNorm ----
            var = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(var + 1e-5) * norm_w

            # ---- QKV projection ----
            q = hidden_states @ q_proj_w.T
            k = hidden_states @ k_proj_w.T
            v = hidden_states @ v_proj_w.T

            q = q.view(B, S, nhead, dim).transpose(1, 2)  # [B,H,S,Dh]
            k = k.view(B, S, nkvhead, dim).transpose(1, 2)
            v = v.view(B, S, nkvhead, dim).transpose(1, 2)

            # ---- RoPE ----
            q, k = torch_rope(
                q,
                k,
                sin_table,
                cos_table,
                pos_ids,
            )

            # ---- KV cache update ----
            k_cache[layer, :, :, past_seqlen : past_seqlen + S, :] = k
            v_cache[layer, :, :, past_seqlen : past_seqlen + S, :] = v

            K = k_cache[layer, :, :, 0 : past_seqlen + S, :]
            V = v_cache[layer, :, :, 0 : past_seqlen + S, :]

            # ---- Scaled Dot Product Attention (fused) ----
            def scaled_dot_product_attention(
                query, key, value, is_causal=False, enable_gqa=False
            ) -> torch.Tensor:
                S, L = query.size(-2), key.size(-2)
                scale_factor = query.size(-1) ** -0.5
                attn_bias = torch.zeros(S, L, dtype=query.dtype, device=query.device)
                if is_causal:
                    mask = torch.tril(attn_bias + 1, diagonal=-1).flip(dims=[-2, -1])
                    attn_bias = torch.where(mask == 1, -torch.inf, 0.0)

                if enable_gqa:
                    key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
                    value = value.repeat_interleave(
                        query.size(-3) // value.size(-3), -3
                    )

                attn_weight = query @ key.transpose(-2, -1) * scale_factor
                attn_weight += attn_bias
                attn_weight = torch.softmax(attn_weight, dim=-1)
                return attn_weight @ value

            attn_out = scaled_dot_product_attention(
                q,
                K,
                V,
                is_causal=True,
                enable_gqa=True,
            )  # [B,H,S,Dh]

            # ---- Output projection ----
            attn_out = attn_out.transpose(1, 2).contiguous()
            attn_out = attn_out.view(B, S, nhead * dim)

            hidden_states = attn_out @ o_proj_w.T

        return hidden_states

    def infinicore_operator(
        self,
        hidden_states,
        pos_ids,
        nhead,
        nkvhead,
        dim,
        past_seqlen,
        nlayers,
        k_cache,
        v_cache,
        q_proj_w,
        k_proj_w,
        v_proj_w,
        o_proj_w,
        norm_w,
        sin_table,
        cos_table,
        **kwargs,
    ):
        """Record graph and run"""
        input_hidden_states = hidden_states
        B, S, D = input_hidden_states.shape

        infinicore.start_graph_recording()
        for layer in range(nlayers):
            hidden_states = infinicore.nn.functional.rms_norm(
                hidden_states, norm_w.shape, norm_w, 1e-5
            )
            q = infinicore.nn.functional.linear(hidden_states, q_proj_w)
            k = infinicore.nn.functional.linear(hidden_states, k_proj_w)
            v = infinicore.nn.functional.linear(hidden_states, v_proj_w)

            q = q.view((B, S, nhead, dim))
            k = k.view((B, S, nkvhead, dim))
            v = v.view((B, S, nkvhead, dim))
            q = infinicore.nn.functional.rope(
                q,
                pos_ids,
                sin_table,
                cos_table,
                infinicore.nn.functional.RopeAlgo.GPT_J,
            )
            k = infinicore.nn.functional.rope(
                k,
                pos_ids,
                sin_table,
                cos_table,
                infinicore.nn.functional.RopeAlgo.GPT_J,
            )

            # [B, KVH, total_len, D]
            full_k = (
                k_cache.narrow(0, layer, 1).squeeze(0).narrow(2, 0, past_seqlen + S)
            )
            full_v = (
                v_cache.narrow(0, layer, 1).squeeze(0).narrow(2, 0, past_seqlen + S)
            )
            full_k.narrow(2, past_seqlen, S).copy_(k.permute((0, 2, 1, 3)))
            full_v.narrow(2, past_seqlen, S).copy_(v.permute((0, 2, 1, 3)))

            G = nhead // nkvhead
            L = past_seqlen + S

            full_q = (
                q.permute((0, 2, 1, 3)).contiguous().view((B * nkvhead, G * S, dim))
            )
            full_k = full_k.view((B * nkvhead, L, dim))
            full_v = full_v.view((B * nkvhead, L, dim))

            attn_score = infinicore.matmul(
                full_q, full_k.permute((0, 2, 1)), alpha=dim**-0.5
            )
            # [B * H, S, total_len]
            attn_score = attn_score.view((B * nhead, S, L))
            infinicore.nn.functional.causal_softmax(attn_score, out=attn_score)
            attn_out = infinicore.matmul(attn_score, full_v)
            attn_out = (
                attn_out.view((B, nhead, S, dim))
                .permute((0, 2, 1, 3))
                .contiguous()
                .view((B, S, nhead * dim))
            )
            hidden_states = infinicore.nn.functional.linear(attn_out, o_proj_w)

        op_graph = infinicore.stop_graph_recording()

        op_graph.run()
        return hidden_states


def main():
    """Main entry point"""
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
