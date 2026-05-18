"""Parity tests for vendored ``SWIGLUOAI`` / ``SWIGLUSTEP`` in ``apply_moe_activation`` (pure Torch).

Uses the same ``sys.path`` bootstrap as ``test_minicpm5_grouped_sigmoid_topk.py`` so
``import infinicore`` resolves to ``InfiniCore/python/infinicore``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
_test_dir = str(_root / "test")
while _test_dir in sys.path:
    sys.path.remove(_test_dir)
_ic_py = str(_root / "python")
if _ic_py not in sys.path:
    sys.path.insert(0, _ic_py)

import importlib.util

import torch
import torch.nn.functional as F
from torch.testing import assert_close

# Import vendored ``activation.py`` without ``import infinicore`` (avoids ``_infinicore`` / host libstdc++).
_act_path = _root / "python" / "infinicore" / "vendor" / "vllm_fused_moe" / "activation.py"
_spec = importlib.util.spec_from_file_location("_vendor_vllm_fused_moe_activation", _act_path)
assert _spec and _spec.loader
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MoEActivation = _mod.MoEActivation
apply_moe_activation = _mod.apply_moe_activation

_ALPHA = 1.702
_LIMIT = 7.0


def _reference_swigluoai(inp: torch.Tensor) -> torch.Tensor:
    gate, up = inp[..., ::2], inp[..., 1::2]
    gate = gate.clamp(min=None, max=_LIMIT)
    up = up.clamp(min=-_LIMIT, max=_LIMIT)
    glu = gate * torch.sigmoid(gate * _ALPHA)
    return (up + 1) * glu


def _reference_swiglustep(inp: torch.Tensor) -> torch.Tensor:
    gate, up = inp.chunk(2, dim=-1)
    gate = F.silu(gate)
    gate = gate.clamp(max=_LIMIT)
    up = up.clamp(min=-_LIMIT, max=_LIMIT)
    return gate * up


def test_swigluoai_matches_reference_bfloat16_cpu():
    torch.manual_seed(0)
    n, d = 16, 64
    inp = torch.randn(n, 2 * d, dtype=torch.bfloat16)
    ref = _reference_swigluoai(inp)
    out = torch.empty(n, d, dtype=torch.bfloat16)
    apply_moe_activation(MoEActivation.SWIGLUOAI, out, inp)
    assert_close(out, ref, rtol=1.6e-2, atol=1.6e-2)


def test_swiglustep_matches_reference_bfloat16_cpu():
    torch.manual_seed(1)
    n, d = 16, 64
    inp = torch.randn(n, 2 * d, dtype=torch.bfloat16)
    ref = _reference_swiglustep(inp)
    out = torch.empty(n, d, dtype=torch.bfloat16)
    apply_moe_activation(MoEActivation.SWIGLUSTEP, out, inp)
    assert_close(out, ref, rtol=1.6e-2, atol=1.6e-2)


def test_swigluoai_matches_reference_float32_cpu():
    torch.manual_seed(2)
    n, d = 8, 32
    inp = torch.randn(n, 2 * d, dtype=torch.float32)
    ref = _reference_swigluoai(inp)
    out = torch.empty(n, d, dtype=torch.float32)
    apply_moe_activation(MoEActivation.SWIGLUOAI, out, inp)
    assert_close(out, ref, rtol=1e-5, atol=1e-5)


def test_swiglustep_matches_reference_float32_cpu():
    torch.manual_seed(3)
    n, d = 8, 32
    inp = torch.randn(n, 2 * d, dtype=torch.float32)
    ref = _reference_swiglustep(inp)
    out = torch.empty(n, d, dtype=torch.float32)
    apply_moe_activation(MoEActivation.SWIGLUSTEP, out, inp)
    assert_close(out, ref, rtol=1e-5, atol=1e-5)
