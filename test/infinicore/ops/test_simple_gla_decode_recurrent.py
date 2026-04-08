"""
Operator unit test for InfiniCore simple_gla_decode_step.

Runs a multi-step decode loop with shared float32 state [B,H,D,D] and compares stacked
outputs to simple_gla_attention on [B,T,H,D] and a PyTorch recurrent reference.
Optional cross-check against FLA naive_recurrent_simple_gla when flash-linear-attention is installed.

Run (from InfiniCore dir):
  PYTHONPATH=<InfiniLM/python>:<InfiniCore/test/infinicore>:<InfiniCore/python> \\
  LD_LIBRARY_PATH=/root/.infini/lib:${LD_LIBRARY_PATH:-} \\
  python test/infinicore/ops/test_simple_gla_decode_recurrent.py --nvidia
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch

import infinicore

from framework import (
    BaseOperatorTest,
    TestCase,
    TensorSpec,
    GenericTestRunner,
    get_args,
)
from framework.devices import torch_device_map
from framework.results import CaseResult
from framework.utils.tensor_utils import convert_infinicore_to_torch, infinicore_tensor_from_torch


SIMPLE_GLA_DECODE_STEP_AVAILABLE = hasattr(infinicore, "simple_gla_decode_step")


def _torch_simple_gla_recurrent_ref(q, k, v, g_gamma, scale: float) -> torch.Tensor:
    """Reference recurrence (HF LightningAttention Simple GLA; matches InfiniLM/InfiniCore math)."""
    dtype = q.dtype
    qf = q.transpose(1, 2).float()
    kf = k.transpose(1, 2).float()
    vf = v.transpose(1, 2).float()
    B, H, T, K = qf.shape
    V = vf.shape[-1]
    qf = qf * scale
    o = vf.new_zeros(B, H, T, V)
    S = qf.new_zeros(B, H, K, V)
    gate = g_gamma.float().exp()
    for i in range(T):
        key = kf[:, :, i, :]
        value = vf[:, :, i, :]
        kv = key.unsqueeze(-1) * value.unsqueeze(-2)
        S = S * gate.view(1, -1, 1, 1) + kv
        q_i = qf[:, :, i, :]
        o[:, :, i, :] = (q_i.unsqueeze(-1) * S).sum(-2)
    return o.transpose(1, 2).to(dtype)


def _optional_fla_output(q, k, v, g_gamma, scale: float):
    """Best-effort FLA naive recurrent; returns None if package or API missing."""
    try:
        from fla.ops.simple_gla import naive_recurrent_simple_gla
    except Exception:
        try:
            from fla.ops.simple_gla.naive import naive_recurrent_simple_gla
        except Exception:
            return None
    try:
        return naive_recurrent_simple_gla(
            q.contiguous(),
            k.contiguous(),
            v.contiguous(),
            g_gamma.contiguous(),
            scale=scale,
        )
    except Exception:
        return None


def _make_case(dtype: str, *, B=2, T=16, H=8, D=64):
    dt = infinicore.bfloat16 if dtype == "bf16" else infinicore.float16
    scale = 1.0 / (D**0.5)

    q = TensorSpec.from_tensor((B, T, H, D), None, dt)
    k = TensorSpec.from_tensor((B, T, H, D), None, dt)
    v = TensorSpec.from_tensor((B, T, H, D), None, dt)
    g_gamma = TensorSpec.from_tensor((H,), None, infinicore.float32)

    return TestCase(
        inputs=[q, k, v, g_gamma],
        kwargs={"scale": scale, "_dtype": dtype, "_shape": (B, T, H, D)},
        output_spec=None,
        comparison_target=None,
        tolerance={"atol": 5e-2, "rtol": 5e-2} if dtype == "bf16" else {"atol": 2e-2, "rtol": 2e-2},
        description=f"simple_gla_decode_step loop vs ref ({dtype}) B={B} T={T} H={H} D={D}",
    )


class SimpleGLADecodeRecurrentTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("SimpleGLADecodeRecurrent")

    def get_test_cases(self):
        if not SIMPLE_GLA_DECODE_STEP_AVAILABLE:
            return []
        return [
            _make_case("fp16"),
            _make_case("bf16"),
            _make_case("fp16", D=128),
            _make_case("bf16", D=128),
        ]

    def torch_operator(self, *args, **kwargs):
        raise NotImplementedError("This test overrides run_test.")

    def infinicore_operator(self, *args, **kwargs):
        raise NotImplementedError("This test overrides run_test.")

    def run_test(self, device, test_case, config):
        tr = CaseResult(success=False, return_code=-1, test_case=test_case, device=device)

        if not SIMPLE_GLA_DECODE_STEP_AVAILABLE:
            tr.return_code = -2
            tr.error_message = "infinicore.simple_gla_decode_step not available (pybind not built?)"
            return tr

        torch.manual_seed(0)
        dev_str = torch_device_map[device]
        infinicore.set_device(infinicore.device(dev_str, 0))

        inputs, kwargs = self.prepare_pytorch_inputs_and_kwargs(test_case, device)
        scale = float(kwargs["scale"])
        assert len(inputs) == 4, "q, k, v, g_gamma"
        for i, t in enumerate(inputs):
            if not t.is_floating_point():
                continue
            with torch.no_grad():
                if t.dim() == 1:
                    t.uniform_(-1.2, -0.05)
                else:
                    t.uniform_(-0.1, 0.1)

        q, k, v, g_gamma = inputs
        B, T, H, D = q.shape

        ref_loop = _torch_simple_gla_recurrent_ref(q, k, v, g_gamma, scale).float()

        infini_inputs, _, _ = self.prepare_infinicore_inputs_and_kwargs(inputs, {"scale": scale}, None)
        q_ic, k_ic, v_ic, g_ic = infini_inputs

        full_out = infinicore.simple_gla_attention(q_ic, k_ic, v_ic, g_ic, scale=scale)
        infinicore.sync_stream()
        full_t = convert_infinicore_to_torch(full_out).float()

        S_buf = torch.zeros(B, H, D, D, device=q.device, dtype=torch.float32)
        S_ic = infinicore_tensor_from_torch(S_buf)
        outs = []
        for ti in range(T):
            qs = infinicore_tensor_from_torch(q[:, ti : ti + 1].contiguous())
            ks = infinicore_tensor_from_torch(k[:, ti : ti + 1].contiguous())
            vs = infinicore_tensor_from_torch(v[:, ti : ti + 1].contiguous())
            o_step = infinicore.simple_gla_decode_step(qs, ks, vs, S_ic, g_ic, scale=scale)
            infinicore.sync_stream()
            outs.append(convert_infinicore_to_torch(o_step))
        stacked = torch.cat(outs, dim=1).float()

        if not torch.isfinite(stacked).all():
            tr.error_message = "decode loop produced NaN/Inf"
            return tr
        if not torch.isfinite(ref_loop.float()).all():
            tr.error_message = "torch reference produced NaN/Inf"
            return tr

        tol = test_case.tolerance or {"atol": 2e-2, "rtol": 2e-2}

        def _diff_line(name, a, b):
            d = (a - b).abs()
            print(f"  {name}: max_abs={float(d.max()):.6f} mean_abs={float(d.mean()):.6f}")

        _diff_line("decode_steps vs torch_ref", stacked, ref_loop.float())
        if not torch.allclose(stacked, ref_loop.float(), atol=float(tol["atol"]), rtol=float(tol["rtol"])):
            d = (stacked - ref_loop.float()).abs()
            tr.error_message = f"vs torch ref: max_abs={float(d.max()):.6f}"
            return tr

        _diff_line("decode_steps vs simple_gla_attention", stacked, full_t)
        if not torch.allclose(stacked, full_t, atol=float(tol["atol"]), rtol=float(tol["rtol"])):
            d = (stacked - full_t).abs()
            tr.error_message = f"vs simple_gla_attention: max_abs={float(d.max()):.6f}"
            return tr

        fla_o = _optional_fla_output(q, k, v, g_gamma, scale)
        if fla_o is not None:
            fla_f = fla_o.float()
            _diff_line("decode_steps vs fla naive", stacked, fla_f)
            if not torch.allclose(stacked, fla_f, atol=float(tol["atol"]), rtol=float(tol["rtol"])):
                d = (stacked - fla_f).abs()
                tr.error_message = f"vs FLA: max_abs={float(d.max()):.6f}"
                return tr
            print("  FLA naive_recurrent_simple_gla cross-check: ok")
        else:
            print("  FLA cross-check skipped (not installed or API mismatch)")

        tr.success = True
        tr.return_code = 0
        return tr


def main():
    args = get_args()
    if not args.nvidia:
        print("simple_gla_decode_step recurrent test expects CUDA; run with --nvidia")
        sys.exit(0)
    if not SIMPLE_GLA_DECODE_STEP_AVAILABLE:
        print("simple_gla_decode_step not available in python bindings (rebuild InfiniCore)")
        sys.exit(1)

    runner = GenericTestRunner(SimpleGLADecodeRecurrentTest, args=args)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
