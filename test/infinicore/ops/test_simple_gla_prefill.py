"""
Operator unit test for InfiniCore simple_gla_prefill.

Validates that simple_gla_prefill(q,k,v,g_gamma,scale) matches the existing
simple_gla_attention reference. Covers head_dim=64 (naive fused path) and
head_dim=128 (chunked/tiled path for MiniCPM-SALA).

Run (from InfiniCore dir):
  PYTHONPATH=<InfiniLM/python>:<InfiniCore/python> LD_LIBRARY_PATH=/root/.infini/lib python test/infinicore/ops/test_simple_gla_prefill.py --nvidia
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
    TensorInitializer,
    GenericTestRunner,
    get_args,
)
from framework.results import CaseResult
from framework.utils.tensor_utils import convert_infinicore_to_torch


SIMPLE_GLA_PREFILL_AVAILABLE = hasattr(infinicore, "simple_gla_prefill")


def _make_case(dtype: str, *, B=1, T=32, H=8, D=64):
    dt = infinicore.bfloat16 if dtype == "bf16" else infinicore.float16
    scale = 1.0 / (D**0.5)

    # q/k/v: [B,T,H,D] ; g_gamma: [H] F32
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
        description=f"simple_gla_prefill vs simple_gla_attention ({dtype}) B={B} T={T} H={H} D={D}",
    )


class SimpleGLAPrefillTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("SimpleGLAPrefill")

    def get_test_cases(self):
        if not SIMPLE_GLA_PREFILL_AVAILABLE:
            return []
        return [
            _make_case("fp16"),
            _make_case("bf16"),
            _make_case("fp16", D=128),
            _make_case("bf16", D=128),
        ]

    def torch_operator(self, *args, **kwargs):
        raise NotImplementedError("This test compares two InfiniCore operators.")

    def infinicore_operator(self, *args, **kwargs):
        raise NotImplementedError("This test overrides run_test.")

    def run_test(self, device, test_case, config):
        tr = CaseResult(success=False, return_code=-1, test_case=test_case, device=device)

        if not SIMPLE_GLA_PREFILL_AVAILABLE:
            tr.return_code = -2
            tr.error_message = "infinicore.simple_gla_prefill not available (pybind not built?)"
            return tr

        torch.manual_seed(0)
        inputs, kwargs = self.prepare_pytorch_inputs_and_kwargs(test_case, device)
        scale = float(kwargs["scale"])
        # Use well-scaled inputs so the GLA recurrence stays finite (avoids FP16 overflow/NaN).
        # q,k,v: small values so k^T v and state S do not explode; g_gamma: negative decay.
        assert len(inputs) == 4, "q, k, v, g_gamma"
        for i, t in enumerate(inputs):
            if not t.is_floating_point():
                continue
            with torch.no_grad():
                if t.dim() == 1:
                    # g_gamma [H]: decay per head, negative so exp(g_gamma) in (0, 1)
                    t.uniform_(-1.2, -0.05)
                else:
                    # q, k, v [B,T,H,D]: keep recurrence bounded
                    t.uniform_(-0.1, 0.1)
        infini_inputs, infini_kwargs, _ = self.prepare_infinicore_inputs_and_kwargs(inputs, {"scale": scale}, None)

        q, k, v, g_gamma = infini_inputs

        out_ref = infinicore.simple_gla_attention(q, k, v, g_gamma, scale=scale)
        out_new = infinicore.simple_gla_prefill(q, k, v, g_gamma, scale=scale)
        infinicore.sync_stream()

        if out_ref is None or out_new is None:
            tr.error_message = "operator returned None"
            return tr

        t_ref = convert_infinicore_to_torch(out_ref).float()
        t_new = convert_infinicore_to_torch(out_new).float()

        if not torch.isfinite(t_ref).all():
            tr.error_message = "reference simple_gla_attention produced NaN/Inf"
            return tr
        if not torch.isfinite(t_new).all():
            tr.error_message = "simple_gla_prefill produced NaN/Inf"
            return tr

        diff = (t_ref - t_new).abs()
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
        print(f"  diff max_abs={max_abs:.6f} mean_abs={mean_abs:.6f}")

        tol = test_case.tolerance or {"atol": 2e-2, "rtol": 2e-2}
        if not torch.allclose(t_new, t_ref, atol=float(tol["atol"]), rtol=float(tol["rtol"])):
            tr.error_message = f"mismatch: max_abs={max_abs:.6f}, atol={tol['atol']} rtol={tol['rtol']}"
            return tr

        tr.success = True
        tr.return_code = 0
        return tr


def main():
    args = get_args()
    if not args.nvidia:
        print("simple_gla_prefill requires CUDA; run with --nvidia")
        sys.exit(0)
    if not SIMPLE_GLA_PREFILL_AVAILABLE:
        print("simple_gla_prefill not available in python bindings (rebuild InfiniCore)")
        sys.exit(1)

    runner = GenericTestRunner(SimpleGLAPrefillTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
