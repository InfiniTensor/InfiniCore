"""
Simple GLA (lightning) attention unit test matching HuggingFace MiniCPM-SALA LightningAttention.
Uses the same computation as HF: linear attention with per-head decay (g_gamma).
Shapes: q, k, v [B, T, H, D], g_gamma [H] (log-decay per head, from slopes * -1).

Baseline vs HF verification:
- HF: decay = _build_slope_tensor(num_attention_heads) * -1; attn_fn(q,k,v,decay=s,scale=1/sqrt(d)).
- fla naive_recurrent_simple_gla: gate = g[:,:,i].exp(); S = S*gate + k^T v; o_t = (q_t*scale)@S.
  With g_gamma (H,) broadcast to all t, gate is (H,) per step, matching our gate.view(1,-1,1,1).
- Our _torch_simple_gla_recurrent: same recurrence and scale; _build_slope_tensor matches HF exactly.

Why this test did not catch the Minicpm layer0 k/v repeat bug:
- Test cases use (B, T, H, D) with H=2 or 4 and the same H for q, k, v (no GQA).
- So num_attention_heads == num_key_value_heads; the repeat_kv path is never exercised.
- MiniCPM-SALA layer0 has n_h=32, n_kv=2; the bug was in the GQA repeat (as_strided) in InfiniLM.

Run (from InfiniCore): LD_LIBRARY_PATH=build/linux/x86_64/release:$LD_LIBRARY_PATH \\
  python test/infinicore/ops/gla_attention.py --cpu [--nvidia].
  If built with aten+nv-gpu, prepend torch lib dir to LD_LIBRARY_PATH so libtorch.so is found.
"""
import math
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
)

# (B, T, H, D) - small shapes to minimize GPU use
_TEST_CASES_DATA = [
    (1, 4, 2, 8),
    (1, 8, 2, 16),
    (2, 8, 4, 16),
]

_TOLERANCE_MAP = {
    infinicore.float16: {"atol": 1e-2, "rtol": 1e-2},
    infinicore.float32: {"atol": 1e-5, "rtol": 1e-4},
    infinicore.bfloat16: {"atol": 2e-2, "rtol": 2e-2},
}
_TENSOR_DTYPES = [infinicore.float16, infinicore.float32, infinicore.bfloat16]


def _build_slope_tensor(nheads: int, dtype=torch.float32):
    """Same as HF MiniCPM-SALA modeling_minicpm_sala._build_slope_tensor."""

    def get_slopes(n):
        def get_slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio**i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + get_slopes(
                2 * closest_power_of_2
            )[0::2][: n - closest_power_of_2]

    return torch.tensor(get_slopes(nheads), dtype=dtype)


def _torch_simple_gla_recurrent(q, k, v, g_gamma, scale):
    """Reference: Simple GLA recurrent (matches fla naive_recurrent_simple_gla with g_gamma (H,)).
    HF LightningAttention uses this with decay = _build_slope_tensor(H) * -1.
    Recurrence: S = S * exp(g_gamma) + k^T v; o_t = (q_t * scale) @ S."""
    dtype = q.dtype
    # q, k, v: (B, T, H, D) -> (B, H, T, D)
    q = q.transpose(1, 2).float()
    k = k.transpose(1, 2).float()
    v = v.transpose(1, 2).float()
    B, H, T, K = q.shape
    V = v.shape[-1]
    if scale is None:
        scale = K ** -0.5
    q = q * scale
    o = v.new_zeros(B, H, T, V)
    S = q.new_zeros(B, H, K, V)
    # g_gamma (H,) -> gate (H,) same for all t
    gate = g_gamma.exp()  # (H,)
    for i in range(T):
        key = k[:, :, i, :]   # (B, H, K)
        value = v[:, :, i, :]  # (B, H, V)
        kv = key.unsqueeze(-1) * value.unsqueeze(-2)  # (B, H, K, V)
        S = S * gate.view(1, -1, 1, 1) + kv
        q_i = q[:, :, i, :]  # (B, H, K)
        o_i = (q_i.unsqueeze(-1) * S).sum(-2)  # (B, H, V)
        o[:, :, i, :] = o_i
    return o.transpose(1, 2).to(dtype)  # (B, T, H, V)


def parse_test_cases():
    cases = []
    for (B, T, H, D) in _TEST_CASES_DATA:
        for dtype in _TENSOR_DTYPES:
            tol = _TOLERANCE_MAP.get(dtype, {"atol": 1e-4, "rtol": 1e-3})
            q_spec = TensorSpec.from_tensor((B, T, H, D), None, dtype)
            k_spec = TensorSpec.from_tensor((B, T, H, D), None, dtype)
            v_spec = TensorSpec.from_tensor((B, T, H, D), None, dtype)
            scale = 1.0 / (D ** 0.5)
            cases.append(
                TestCase(
                    inputs=[q_spec, k_spec, v_spec],
                    kwargs={"scale": scale},
                    output_spec=None,
                    comparison_target=None,
                    tolerance=tol,
                    description=f"SimpleGLA B={B} T={T} H={H} D={D}",
                )
            )
    return cases


def _to_torch_device(device):
    """Map InfiniCore or torch device to torch.device."""
    if isinstance(device, torch.device):
        return device
    if hasattr(device, "type"):
        t = getattr(device, "type", None)
        if t == "cuda" or (isinstance(t, str) and "cuda" in t.lower()):
            idx = getattr(device, "index", 0)
            return torch.device("cuda", idx if isinstance(idx, int) else 0)
        return torch.device("cpu")
    return torch.device("cpu")


def _g_gamma_for_heads(H, device):
    """Decay tensor matching HF: slopes * -1."""
    s = _build_slope_tensor(H)
    torch_dev = _to_torch_device(device)
    return (s * -1.0).to(torch_dev)


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("SimpleGLAAttention")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, q, k, v, *, scale):
        H = q.shape[2]
        g_gamma = _g_gamma_for_heads(H, q.device)
        return _torch_simple_gla_recurrent(q, k, v, g_gamma, scale)

    def infinicore_operator(self, q, k, v, *, scale):
        simple_gla = getattr(infinicore, "simple_gla_attention", None)
        if simple_gla is None:
            raise NotImplementedError(
                "simple_gla_attention not in InfiniCore; baseline matches HF Simple GLA."
            )
        H = q.shape[2]
        g_gamma_torch = _g_gamma_for_heads(H, q.device)
        g_gamma = infinicore.from_torch(g_gamma_torch)
        out = simple_gla(q, k, v, g_gamma, scale=scale)
        infinicore.sync_stream()
        return out


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
