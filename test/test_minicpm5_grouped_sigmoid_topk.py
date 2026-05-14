"""Operator tests for ``torch.ops.infinilm.minicpm5_grouped_sigmoid_topk`` (CUDA).

Lives under ``InfiniCore/test/`` (not ``test/infinicore/``) so ``import infinicore`` resolves to
``InfiniCore/python/infinicore`` when ``PYTHONPATH`` includes ``InfiniCore/python``.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Pytest prepends ``InfiniCore/test`` so ``import infinicore`` resolves to ``test/infinicore`` (no vendor).
_root = Path(__file__).resolve().parents[1]
_test_dir = str(_root / "test")
while _test_dir in sys.path:
    sys.path.remove(_test_dir)
_ic_py = str(_root / "python")
if _ic_py not in sys.path:
    sys.path.insert(0, _ic_py)

import pytest
import torch

from infinicore.vendor.vllm_fused_moe.minicpm5_grouped_sigmoid_topk import (
    grouped_sigmoid_topk_torch_reference,
)


@pytest.fixture(scope="module", autouse=True)
def _register_ops():
    import infinicore.vendor.vllm_fused_moe  # noqa: F401 — registers torch.ops.infinilm


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@requires_cuda
def test_minicpm5_grouped_sigmoid_topk_matches_reference_small():
    device = torch.device("cuda", 0)
    n_tokens, n_experts, n_group, topk_group, top_k = 4, 8, 2, 1, 2
    logits = torch.randn(n_tokens, n_experts, device=device, dtype=torch.float32)
    bias = torch.randn(n_experts, device=device, dtype=torch.float32)
    ref_w, ref_ids = grouped_sigmoid_topk_torch_reference(
        logits, top_k, True, n_group, topk_group, 1.0, bias
    )
    w, ids = torch.ops.infinilm.minicpm5_grouped_sigmoid_topk(
        logits, bias, top_k, True, n_group, topk_group, 1.0
    )
    assert w.shape == (n_tokens, top_k)
    assert ids.shape == (n_tokens, top_k)
    assert w.dtype == torch.float32
    assert ids.dtype == torch.int32
    torch.testing.assert_close(w, ref_w, rtol=1e-5, atol=1e-6)
    assert torch.equal(ids, ref_ids)


@requires_cuda
def test_minicpm5_grouped_sigmoid_topk_renorm_and_scale():
    device = torch.device("cuda", 0)
    n_tokens, n_experts, n_group, topk_group, top_k = 1, 16, 4, 2, 2
    logits = torch.randn(n_tokens, n_experts, device=device, dtype=torch.float32)
    bias = torch.zeros(n_experts, device=device, dtype=torch.float32)
    scale = 2.5
    w, _ids = torch.ops.infinilm.minicpm5_grouped_sigmoid_topk(
        logits, bias, top_k, True, n_group, topk_group, float(scale)
    )
    unnorm = w / scale
    row_sums = unnorm.sum(dim=-1)
    torch.testing.assert_close(row_sums, torch.ones_like(row_sums), rtol=1e-4, atol=1e-5)


@requires_cuda
def test_minicpm5_grouped_sigmoid_topk_single_token():
    device = torch.device("cuda", 0)
    n_tokens, n_experts, n_group, topk_group, top_k = 1, 8, 2, 1, 2
    logits = torch.randn(n_tokens, n_experts, device=device, dtype=torch.float32)
    bias = torch.randn(n_experts, device=device, dtype=torch.float32)
    w, ids = torch.ops.infinilm.minicpm5_grouped_sigmoid_topk(
        logits, bias, top_k, False, n_group, topk_group, 1.0
    )
    assert w.shape == (1, top_k)
    assert ids.shape == (1, top_k)
