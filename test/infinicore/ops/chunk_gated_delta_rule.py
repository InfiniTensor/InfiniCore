import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from framework import (
    BaseOperatorTest,
    TensorSpec,
    TestCase,
    GenericTestRunner,
    TensorInitializer,
)

# Test cases:
# (n_khead, kdim, n_vhead, vdim, seqlens, init_state_indices, final_state_indices, state_pool_size)
_VARLEN_TEST_CASES_DATA = [
    (16, 128, 48, 128, (13,), (0,), (0,), 1),
    (16, 128, 48, 128, (13,), (1,), (0,), 2),
    (16, 128, 48, 128, (13, 20), (1, 1), (0, 0), 4),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-2, "rtol": 1e-2},
    infinicore.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-3, "rtol": 1e-3},
}

_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]


def torch_chunk_gated_delta_rule_ref(
    q,
    k,
    v,
    g,
    beta,
    initial_state,
    cu_seqlens=None,
    initial_state_indices=None,
    final_state_indices=None,
    use_qk_l2norm=False,
    chunk_size=64,
):

    def l2norm(x: torch.FloatTensor, dim: int = -1, eps: float = 1e-6):
        inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
        return x * inv_norm

    def _run_one(q, k, v, g, beta, init_state):
        # q/k:       [1, T, Hk, Dk]
        # v/out:     [1, T, Hv, Dv]
        # g/beta:    [1, T, Hv]
        # init_state: [B, Hv, Dv, Dk]
        # return state: [B, Hv, Dv, Dk]
        initial_dtype = q.dtype

        if use_qk_l2norm:
            q = l2norm(q, dim=-1, eps=1e-6)
            k = l2norm(k, dim=-1, eps=1e-6)

        B, T, Hk, Dk = q.shape
        _, _, Hv, Dv = v.shape
        assert B == 1
        assert Hv % Hk == 0
        assert init_state.shape == (B, Hv, Dv, Dk)

        group_size = Hv // Hk
        if group_size != 1:
            q = q.repeat_interleave(group_size, dim=2)
            k = k.repeat_interleave(group_size, dim=2)

        q = q.transpose(1, 2).contiguous().float()
        k = k.transpose(1, 2).contiguous().float()
        v = v.transpose(1, 2).contiguous().float()

        beta = beta.transpose(1, 2).contiguous().float()
        g = g.transpose(1, 2).contiguous().float()

        B, H, sequence_length, Dk = k.shape
        Dv = v.shape[-1]

        pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size

        q = torch.nn.functional.pad(q, (0, 0, 0, pad_size))
        k = torch.nn.functional.pad(k, (0, 0, 0, pad_size))
        v = torch.nn.functional.pad(v, (0, 0, 0, pad_size))
        beta = torch.nn.functional.pad(beta, (0, pad_size))
        g = torch.nn.functional.pad(g, (0, pad_size))

        total_sequence_length = sequence_length + pad_size
        scale = 1 / (q.shape[-1] ** 0.5)
        q = q * scale

        v_beta = v * beta.unsqueeze(-1)
        k_beta = k * beta.unsqueeze(-1)

        q, k, v, k_beta, v_beta = [
            x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
            for x in (q, k, v, k_beta, v_beta)
        ]
        g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)

        mask = torch.triu(
            torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device),
            diagonal=0,
        )

        g = g.cumsum(dim=-1)
        decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()

        attn = -((k_beta @ k.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)

        for i in range(1, chunk_size):
            row = attn[..., i, :i].clone()
            sub = attn[..., :i, :i].clone()
            attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)

        attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

        v = attn @ v_beta
        k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))

        # [B, Hv, Dv, Dk] -> [B, Hv, Dk, Dv]
        last_state = init_state.transpose(-1, -2).contiguous().float().clone()

        out = torch.zeros_like(v)

        for i in range(total_sequence_length // chunk_size):
            q_i = q[:, :, i]
            k_i = k[:, :, i]
            v_i = v[:, :, i]

            attn = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]

            v_prime = k_cumdecay[:, :, i] @ last_state
            v_new = v_i - v_prime

            attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_state
            out[:, :, i] = attn_inter + attn @ v_new

            last_state = (
                last_state * g[:, :, i, -1, None, None].exp()
                + (
                    k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]
                ).transpose(-1, -2)
                @ v_new
            )

        out = out.reshape(out.shape[0], out.shape[1], -1, out.shape[-1])
        out = out[:, :, :sequence_length]
        out = out.transpose(1, 2).contiguous().to(initial_dtype)

        # [B, Hv, Dk, Dv] -> [B, Hv, Dv, Dk]
        final_state = last_state.transpose(-1, -2).contiguous().to(init_state.dtype)

        return out, final_state

    if cu_seqlens is None:
        out, final_state = _run_one(q, k, v, g, beta, initial_state)

        if initial_state_indices is not None:
            for b, dst in enumerate(final_state_indices.cpu().tolist()):
                initial_state[dst].copy_(final_state[b].to(initial_state.dtype))

        return out

    cu = cu_seqlens.cpu().tolist()
    batch = len(cu) - 1
    total_tokens = cu[-1]

    out = torch.empty_like(v[:, :total_tokens])
    indexed_state_pool = initial_state_indices is not None

    for b in range(batch):
        start = cu[b]
        end = cu[b + 1]

        q_b = q[:, start:end]
        k_b = k[:, start:end]
        v_b = v[:, start:end]
        g_b = g[:, start:end]
        beta_b = beta[:, start:end]

        if indexed_state_pool:
            src = int(initial_state_indices[b].item())
            init_b = initial_state[src : src + 1]
        else:
            init_b = initial_state[b : b + 1]

        out_b, final_b = _run_one(q_b, k_b, v_b, g_b, beta_b, init_b)
        out[:, start:end].copy_(out_b)

        if indexed_state_pool:
            dst = int(final_state_indices[b].item())
            initial_state[dst].copy_(final_b[0].to(initial_state.dtype))

    return out


def parse_varlen_test_cases():
    tests = []

    for (
        n_khead,
        kdim,
        n_vhead,
        vdim,
        seqlens,
        init_state_indices,
        final_state_indices,
        state_pool_size,
    ) in _VARLEN_TEST_CASES_DATA:
        batch = len(seqlens)
        total_tokens = sum(seqlens)
        cu_seqlens = [0]
        for seqlen in seqlens:
            cu_seqlens.append(cu_seqlens[-1] + seqlen)

        q_shape = (1, total_tokens, n_khead, kdim)
        k_shape = (1, total_tokens, n_khead, kdim)
        v_shape = (1, total_tokens, n_vhead, vdim)
        g_shape = (1, total_tokens, n_vhead)
        beta_shape = (1, total_tokens, n_vhead)

        # Indexed state-pool mode in your wrapper doc:
        # initial_state: [pool_size, Hv, Dv, Dk]
        state_shape = (state_pool_size, n_vhead, vdim, kdim)

        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-5, "rtol": 1e-3})

            q_spec = TensorSpec.from_tensor(q_shape, None, dtype, scale=0.2, bias=-0.1)
            k_spec = TensorSpec.from_tensor(k_shape, None, dtype, scale=0.2, bias=-0.1)
            v_spec = TensorSpec.from_tensor(v_shape, None, dtype, scale=0.2, bias=-0.1)
            g_spec = TensorSpec.from_tensor(
                g_shape, None, infinicore.float32, scale=0.02, bias=-0.01
            )
            beta_spec = TensorSpec.from_tensor(
                beta_shape, None, infinicore.float32, scale=0.5, bias=0.0
            )
            state_spec = TensorSpec.from_tensor(
                state_shape, None, dtype, init_mode=TensorInitializer.ZEROS
            )

            cu_seqlens_spec = TensorSpec.from_tensor(
                (batch + 1,),
                None,
                infinicore.int32,
                init_mode=TensorInitializer.MANUAL,
                set_tensor=torch.tensor(cu_seqlens, dtype=torch.int32),
            )

            init_indices_spec = TensorSpec.from_tensor(
                (batch,),
                None,
                infinicore.int32,
                init_mode=TensorInitializer.MANUAL,
                set_tensor=torch.tensor(init_state_indices, dtype=torch.int32),
            )
            final_indices_spec = TensorSpec.from_tensor(
                (batch,),
                None,
                infinicore.int32,
                init_mode=TensorInitializer.MANUAL,
                set_tensor=torch.tensor(final_state_indices, dtype=torch.int32),
            )

            tests.append(
                TestCase(
                    inputs=[
                        q_spec,
                        k_spec,
                        v_spec,
                        g_spec,
                        beta_spec,
                        state_spec,
                        cu_seqlens_spec,
                        init_indices_spec,
                        final_indices_spec,
                    ],
                    kwargs={
                        "use_qk_l2norm": True,
                        "chunk_size": 64,
                    },
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description="ChunkGatedDeltaRule - VARLEN_INDEXED_STATE_POOL",
                )
            )

    return tests


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("ChunkGatedDeltaRule")

    def get_test_cases(self):
        return parse_varlen_test_cases()

    def torch_operator(self, *args, **kwargs):
        args = list(args)
        args[5] = args[5].clone()
        return torch_chunk_gated_delta_rule_ref(*args, **kwargs)

    def infinicore_operator(
        self,
        q,
        k,
        v,
        g,
        beta,
        states,
        cu_seqlens,
        initial_state_indices,
        final_state_indices,
        use_qk_l2norm,
        chunk_size=64,
    ):
        return infinicore.nn.functional.chunk_gated_delta_rule(
            q,
            k,
            v,
            g,
            beta,
            states,
            cu_seqlens=cu_seqlens,
            initial_state_indices=initial_state_indices,
            final_state_indices=final_state_indices,
            use_qk_l2norm=use_qk_l2norm,
            chunk_size=chunk_size,
        )


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
