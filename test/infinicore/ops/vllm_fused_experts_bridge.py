"""
Operator-style test: InfiniCore ATen bridge + vLLM ``fused_experts`` vs naive PyTorch MoE.

Requires CUDA, InfiniCore ``--aten=y``, vLLM, and Triton (``HAS_TRITON``).
Otherwise both operators raise ``NotImplementedError`` (skipped in the framework).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import infinicore
import torch
from infinicore.lib import _infinicore

from framework import (
    BaseOperatorTest,
    GenericTestRunner,
    TensorSpec,
    TestCase,
)
from framework.tensor import TensorInitializer


def _bridge_deps_available() -> bool:
    if getattr(_infinicore, "_tensor_as_torch", None) is None:
        return False
    try:
        import torch as _t

        if not _t.cuda.is_available():
            return False
        from vllm.triton_utils import HAS_TRITON

        if not HAS_TRITON:
            return False
        import vllm.model_executor.layers.fused_moe  # noqa: F401
    except ImportError:
        return False
    return True


def _naive_fused_experts(
    hidden: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_w: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    """SwiGLU + SiLU reference matching vLLM unquantized MoE (``MoEActivation.SILU``)."""
    m, h = hidden.shape
    e, i2, hdim = w1.shape
    assert hdim == h
    inter = i2 // 2
    out = torch.empty(m, h, device=hidden.device, dtype=hidden.dtype)
    for ti in range(m):
        x = hidden[ti]
        acc = torch.zeros(h, device=hidden.device, dtype=torch.float32)
        for k in range(topk_ids.shape[1]):
            ex = int(topk_ids[ti, k].item())
            wt = float(topk_w[ti, k].item())
            w1e = w1[ex]
            lin = torch.nn.functional.linear(x, w1e)
            gate, up = lin.split(inter, dim=-1)
            mid = torch.nn.functional.silu(gate) * up
            y = torch.nn.functional.linear(mid, w2[ex])
            acc = acc + wt * y.float()
        out[ti] = acc.to(hidden.dtype)
    return out


def _mk_case(m, h, e, topk, inter, dt, tol, description):
    i2 = 2 * inter
    return TestCase(
        inputs=[
            TensorSpec.from_tensor((m, h), None, dt, init_mode=TensorInitializer.RANDOM),
            TensorSpec.from_tensor((e, i2, h), None, dt, init_mode=TensorInitializer.RANDOM),
            TensorSpec.from_tensor((e, h, inter), None, dt, init_mode=TensorInitializer.RANDOM),
            TensorSpec.from_tensor((m, topk), None, dt, init_mode=TensorInitializer.RANDOM),
            TensorSpec.from_tensor(
                (m, topk),
                None,
                infinicore.int32,
                init_mode=TensorInitializer.RANDINT,
                low=0,
                high=e,
            ),
        ],
        kwargs={},
        output_spec=None,
        comparison_target=None,
        tolerance=tol,
        description=description,
    )


def parse_test_cases():
    # bf16: typical vLLM shapes + a few edge sizes (naive ref is O(m * topk); keep m*topk modest).
    bf16 = infinicore.bfloat16
    fp16 = infinicore.float16
    tol_bf = {"atol": 8e-2, "rtol": 8e-2}
    tol_fp = {"atol": 2e-1, "rtol": 2e-1}
    return [
        _mk_case(8, 32, 4, 2, 16, bf16, tol_bf, "bf16 small"),
        _mk_case(24, 64, 6, 2, 32, bf16, tol_bf, "bf16 medium tokens"),
        _mk_case(2, 128, 8, 1, 48, bf16, tol_bf, "bf16 topk=1 wide hidden"),
        _mk_case(16, 48, 4, 3, 24, bf16, tol_bf, "bf16 topk=3"),
        _mk_case(32, 96, 8, 2, 64, bf16, tol_bf, "bf16 larger experts"),
        _mk_case(12, 56, 5, 2, 28, fp16, tol_fp, "fp16"),
    ]


class OpTest(BaseOperatorTest):
    def __init__(self):
        super().__init__("VllmFusedExpertsBridge")

    def get_test_cases(self):
        return parse_test_cases()

    def torch_operator(self, *args, **kwargs):
        if not _bridge_deps_available():
            raise NotImplementedError(
                "skip: need CUDA + InfiniCore ATen + vLLM + Triton for this test"
            )
        hidden, w1, w2, topk_w, topk_ids = args
        return _naive_fused_experts(hidden, w1, w2, topk_w, topk_ids)

    def infinicore_operator(self, *args, **kwargs):
        if not _bridge_deps_available():
            raise NotImplementedError(
                "skip: need CUDA + InfiniCore ATen + vLLM + Triton for this test"
            )
        from infinicore.vllm_fused_moe_bridge import fused_experts_ic

        hidden, w1, w2, topk_w, topk_ids = args
        return fused_experts_ic(hidden, w1, w2, topk_w, topk_ids)


def main():
    runner = GenericTestRunner(OpTest)
    runner.run_and_exit()


if __name__ == "__main__":
    main()
