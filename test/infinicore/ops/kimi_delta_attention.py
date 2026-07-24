import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import infinicore
from framework import BaseOperatorTest, GenericTestRunner, TensorSpec, TestCase, TensorInitializer


_TENSOR_DTYPES = [infinicore.float16, infinicore.bfloat16, infinicore.float32]
_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 3e-2, "rtol": 3e-2},
    infinicore.bfloat16: {"atol": 4e-2, "rtol": 4e-2},
    infinicore.float32: {"atol": 2e-4, "rtol": 2e-4},
}


def _l2norm(x):
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + 1e-6)


def torch_kimi_delta_attention_ref(
    q,
    k,
    v,
    g,
    beta,
    A_log,
    dt_bias,
    initial_state,
    cu_seqlens=None,
    initial_state_indices=None,
    final_state_indices=None,
    scale=1.0,
    lower_bound=-5.0,
    use_qk_l2norm=True,
):
    initial_dtype = q.dtype
    qf = q.float()
    kf = k.float()
    vf = v.float()
    gf = g.float()
    betaf = beta.float()
    state_pool = initial_state.float().clone()
    out = torch.empty_like(vf)

    if cu_seqlens is None:
        batch = q.shape[0]
        ranges = [(b, 0, q.shape[1], b) for b in range(batch)]
    else:
        cu = cu_seqlens.cpu().tolist()
        ranges = [(i, cu[i], cu[i + 1], 0) for i in range(len(cu) - 1)]

    for req_idx, begin, end, token_batch in ranges:
        read_slot = (
            int(initial_state_indices[req_idx].item())
            if initial_state_indices is not None
            else req_idx
        )
        write_slot = (
            int(final_state_indices[req_idx].item())
            if final_state_indices is not None
            else req_idx
        )
        if read_slot < 0 or write_slot < 0:
            out[token_batch, begin:end].zero_()
            continue

        for h in range(q.shape[2]):
            state = state_pool[read_slot, h].clone()
            a_log_exp = A_log[h].float().exp()
            for t in range(begin, end):
                q_t = qf[token_batch, t, h]
                k_t = kf[token_batch, t, h]
                if use_qk_l2norm:
                    q_t = _l2norm(q_t)
                    k_t = _l2norm(k_t)
                q_t = q_t * scale

                gate = lower_bound * torch.sigmoid(
                    a_log_exp * (gf[token_batch, t, h] + dt_bias[h].float())
                )
                decay = gate.exp()
                beta_t = betaf[token_batch, t, h].sigmoid()

                decayed_state = state * decay.view(1, -1)
                kv_mem = (decayed_state * k_t.view(1, -1)).sum(dim=-1)
                delta = (vf[token_batch, t, h] - kv_mem) * beta_t
                state = decayed_state + delta.view(-1, 1) * k_t.view(1, -1)
                out[token_batch, t, h] = (state * q_t.view(1, -1)).sum(dim=-1)
            state_pool[write_slot, h].copy_(state)

    return out.to(initial_dtype)


def parse_test_cases():
    tests = []
    for dtype in _TENSOR_DTYPES:
        tol = _TOLERANCE_MAP[dtype]
        for shape, cu, indexed in [
            ((2, 1, 2, 8), None, False),
            ((2, 1, 2, 8), None, True),
            ((2, 3, 2, 8), None, False),
            ((2, 3, 2, 8), None, True),
            ((1, 2, 2, 8), torch.tensor([0, 1, 2], dtype=torch.int64), True),
            ((1, 5, 2, 8), torch.tensor([0, 2, 5], dtype=torch.int64), False),
            ((1, 5, 2, 8), torch.tensor([0, 2, 5], dtype=torch.int64), True),
        ]:
            B_state = shape[0] if cu is None else cu.numel() - 1
            pool_size = 4 if indexed else B_state
            H = shape[2]
            D = shape[3]
            q = TensorSpec.from_tensor(shape, None, dtype)
            k = TensorSpec.from_tensor(shape, None, dtype)
            v = TensorSpec.from_tensor(shape, None, dtype)
            g = TensorSpec.from_tensor(shape, None, dtype)
            beta = TensorSpec.from_tensor(shape[:3], None, dtype)
            A_log = TensorSpec.from_tensor((H,), None, infinicore.float32)
            dt_bias = TensorSpec.from_tensor((H, D), None, infinicore.float32)
            initial_state = TensorSpec.from_tensor((pool_size, H, D, D), None, dtype)
            kwargs = {
                "scale": D**-0.5,
                "lower_bound": -5.0,
                "use_qk_l2norm": True,
            }
            if cu is not None:
                kwargs["cu_seqlens"] = TensorSpec.from_tensor(
                    tuple(cu.shape),
                    None,
                    infinicore.int64,
                    init_mode=TensorInitializer.MANUAL,
                    set_tensor=cu,
                )
            if indexed:
                initial_indices = torch.tensor([2, 0], dtype=torch.int64)
                final_indices = torch.tensor([1, 3], dtype=torch.int64)
                kwargs["initial_state_indices"] = TensorSpec.from_tensor(
                    tuple(initial_indices.shape),
                    None,
                    infinicore.int64,
                    init_mode=TensorInitializer.MANUAL,
                    set_tensor=initial_indices,
                )
                kwargs["final_state_indices"] = TensorSpec.from_tensor(
                    tuple(final_indices.shape),
                    None,
                    infinicore.int64,
                    init_mode=TensorInitializer.MANUAL,
                    set_tensor=final_indices,
                )
            tests.append(
                TestCase(
                    inputs=[q, k, v, g, beta, A_log, dt_bias, initial_state],
                    kwargs=kwargs,
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=(
                        "KimiDeltaAttention"
                        + (" indexed-pool" if indexed else "")
                        + (" varlen" if cu is not None else "")
                    ),
                )
            )
    return tests


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("KimiDeltaAttention")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        return torch_kimi_delta_attention_ref(*args, **kwargs)

    def infinicore_operator(self, *args, **kwargs):
        return infinicore.nn.functional.kimi_delta_attention(*args, **kwargs)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
