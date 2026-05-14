"""
Primary knob for InfiniLM MiniCPM5 MoE fused path: vendor Triton copy vs upstream vLLM imports,
and optional CPU-router diagnostic on the vendor stack.

``INFINILM_MOE_FUSED_STACK`` — ``vendor`` (default when unset), ``vendor_router_cpu``, or ``upstream``.
"""

from __future__ import annotations

import os
import sys
from typing import Literal

_MoeFusedStack = Literal["vendor", "vendor_router_cpu", "upstream"]

_legacy_warned: bool = False
_obsolete_router_flags_warned: bool = False


def _warn_obsolete_moe_router_flags_once() -> None:
    """``INFINILM_MOE_ROUTER`` / ``INFINILM_MOE_ROUTER_ENGINE`` are no longer read."""
    global _obsolete_router_flags_warned
    if _obsolete_router_flags_warned:
        return
    if not os.environ.get("INFINILM_MOE_ROUTER") and not os.environ.get("INFINILM_MOE_ROUTER_ENGINE"):
        return
    _obsolete_router_flags_warned = True
    print(
        "[infinilm] MoE: INFINILM_MOE_ROUTER / INFINILM_MOE_ROUTER_ENGINE are ignored; "
        "use INFINILM_MOE_FUSED_STACK=vendor|vendor_router_cpu|upstream.",
        file=sys.stderr,
        flush=True,
    )


def _warn_legacy_moe_env_once() -> None:
    global _legacy_warned
    if _legacy_warned:
        return
    has_legacy = os.environ.get("INFINILM_USE_VLLM_GROUPED_TOPK_KERNEL", "") in (
        "1",
        "true",
        "TRUE",
        "yes",
        "YES",
    )
    if not has_legacy:
        return
    _legacy_warned = True
    print(
        "[infinilm] MoE: INFINILM_USE_VLLM_GROUPED_TOPK_KERNEL is legacy on the vendor stack; "
        "prefer INFINILM_MOE_FUSED_STACK=vendor|vendor_router_cpu|upstream (see InfiniLM/examples/minicpm5_moe_inference_profiling.md).",
        file=sys.stderr,
        flush=True,
    )


def resolve_moe_fused_stack() -> _MoeFusedStack:
    """
    Effective fused stack for router + fused experts.

    Explicit ``INFINILM_MOE_FUSED_STACK`` wins (case-insensitive):
    ``vendor``, ``vendor_router_cpu`` (same vendor fused experts; routing uses the CPU topk path in the fused block), or ``upstream``.
    When unset: ``vendor``.
    Invalid explicit values fall back to ``vendor``.
    """
    _warn_obsolete_moe_router_flags_once()
    raw = os.environ.get("INFINILM_MOE_FUSED_STACK")
    if raw is not None and raw.strip() != "":
        v = raw.strip().lower()
        if v == "upstream":
            _warn_legacy_moe_env_once()
            return "upstream"
        if v == "vendor_router_cpu":
            _warn_legacy_moe_env_once()
            return "vendor_router_cpu"
        if v == "vendor":
            _warn_legacy_moe_env_once()
            return "vendor"
        print(
            f"[infinilm] MoE: invalid INFINILM_MOE_FUSED_STACK={raw!r}; using vendor",
            file=sys.stderr,
            flush=True,
        )
        _warn_legacy_moe_env_once()
        return "vendor"

    _warn_legacy_moe_env_once()
    return "vendor"


def moe_fused_stack_upstream_env() -> bool:
    """C++/docs parity: true iff effective stack is upstream (same rules as resolve_moe_fused_stack)."""
    return resolve_moe_fused_stack() == "upstream"
