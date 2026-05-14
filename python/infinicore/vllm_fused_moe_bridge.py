"""
Bridge: run vendored vLLM-derived ``fused_experts`` on ATen views of InfiniCore tensors.

Requires InfiniCore built with ``--aten=y``, CUDA, `triton`, and the in-repo vendor module
``infinicore.vendor.vllm_fused_moe`` (registers ``torch.ops.infinilm.*``). No ``pip install vllm``.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from infinicore.lib import _infinicore
from infinicore.tensor import from_torch, to_torch

if TYPE_CHECKING:
    from infinicore.tensor import Tensor


def _require_aten_bridge() -> None:
    if getattr(_infinicore, "_tensor_as_torch", None) is None:
        raise RuntimeError(
            "vllm_fused_moe_bridge requires InfiniCore with ATen enabled "
            "(rebuild with --aten=y)."
        )


def fused_experts_ic(
    hidden_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    *,
    inplace: bool = False,
    apply_router_weight_on_input: bool = False,
):
    """
    Run fused MoE experts on InfiniCore tensors via ``to_torch`` / ``from_torch``.

    Weight layout matches vLLM ``FusedMoE`` / ``fused_experts``:
    ``w1`` shape ``[num_experts, 2 * intermediate_size, hidden_size]``,
    ``w2`` shape ``[num_experts, hidden_size, intermediate_size]``,
    last dimension contiguous (stride 1). ``hidden_states`` shape ``[num_tokens, hidden_size]``, contiguous.

    Returns a new InfiniCore tensor (aliases the output torch tensor via ``from_torch``).
    """
    _require_aten_bridge()

    try:
        import torch

        import infinicore.vendor.vllm_fused_moe as _vmoe  # noqa: F401 — registers torch.ops.infinilm
        from infinicore.vendor.vllm_fused_moe import MoEActivation, fused_experts
    except ImportError as e:
        raise RuntimeError(
            "fused_experts_ic requires InfiniLM vendored fused MoE + Triton."
        ) from e

    h = to_torch(hidden_states)
    t_w1 = to_torch(w1)
    t_w2 = to_torch(w2)
    t_tw = to_torch(topk_weights)
    t_ids = to_torch(topk_ids)

    if not h.is_contiguous():
        h = h.contiguous()
    if t_w1.stride(-1) != 1:
        t_w1 = t_w1.contiguous()
    if t_w2.stride(-1) != 1:
        t_w2 = t_w2.contiguous()

    if torch.cuda.is_available():
        torch.cuda.current_stream().synchronize()

    out_t = fused_experts(
        h,
        t_w1,
        t_w2,
        t_tw,
        t_ids,
        inplace=inplace,
        activation=MoEActivation.SILU,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    return from_torch(out_t)


def _load_hf_config_json(model_path: str) -> dict:
    path = os.path.join(os.path.expanduser(model_path), "config.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _torch_dtype_from_hf(cfg: dict):
    import torch

    td = cfg.get("torch_dtype")
    if td is None:
        return torch.bfloat16
    if isinstance(td, str):
        name = td.replace("torch.", "", 1)
        return getattr(torch, name, torch.bfloat16)
    return torch.bfloat16


def _moe_dims_from_config(cfg: dict) -> tuple[int, int, int, int] | None:
    """
    Return (num_experts, intermediate, hidden, top_k) if config looks like a MoE model, else None.
    """
    n_exp = cfg.get("n_routed_experts")
    if n_exp is None:
        n_exp = cfg.get("num_local_experts")
    if n_exp is None:
        return None

    inter = cfg.get("moe_intermediate_size")
    if inter is None:
        inter = cfg.get("intermediate_size")
    hidden = cfg.get("hidden_size")
    if inter is None or hidden is None:
        return None

    topk = cfg.get("num_experts_per_tok")
    if topk is None:
        topk = cfg.get("num_experts_per_token")
    if topk is None:
        topk = 1

    return (int(n_exp), int(inter), int(hidden), int(topk))


def _dtype_nbytes(torch_dtype) -> int:
    import torch

    return torch.tensor([], dtype=torch_dtype).element_size()


def _estimated_fused_moe_warmup_bytes(
    E: int, N: int, H: int, topk: int, num_tokens: int, torch_dtype
) -> int:
    """Rough peak device memory for dummy w1/w2 + activations (bf16/fp16 = 2 bytes)."""
    es = _dtype_nbytes(torch_dtype)
    w1 = E * (2 * N) * H * es
    w2 = E * H * N * es
    hs = num_tokens * H * es
    tw = num_tokens * topk * es
    tid = num_tokens * topk * 4
    return w1 + w2 + hs + tw + tid


def verify_vllm_fused_moe_config_for_checkpoint(model_path: str, device_index: int = 0) -> None:
    """
    Log whether a tuned fused_moe JSON exists for this model's (E, N) and GPU name.

    Search order mirrors upstream vLLM: ``INFINILM_TUNED_CONFIG_FOLDER`` / ``VLLM_TUNED_CONFIG_FOLDER``,
    then vendored ``configs/`` next to ``fused_moe.py``.
    """
    dims = _moe_dims_from_config(_load_hf_config_json(model_path))
    if dims is None:
        return

    import torch

    E, N, _, _ = dims
    if not torch.cuda.is_available():
        print(
            f"[vllm_fused_moe] preflight: MoE E={E} N={N} (CUDA unavailable; skip config check)",
            flush=True,
        )
        return

    torch.cuda.set_device(device_index)
    try:
        import infinicore.vendor.vllm_fused_moe.envs as moe_envs
        import infinicore.vendor.vllm_fused_moe.fused_moe as vmoe_fused

        get_config_file_name = vmoe_fused.get_config_file_name
    except ImportError:
        print(
            "[vllm_fused_moe] preflight: vendored fused_moe not importable; skip config check",
            flush=True,
        )
        return

    json_name = get_config_file_name(E, N, None, None)
    fused_moe_dir = os.path.dirname(vmoe_fused.__file__)
    default_path = os.path.join(fused_moe_dir, "configs", json_name)
    paths = []
    if moe_envs.VLLM_TUNED_CONFIG_FOLDER:
        paths.append(os.path.join(moe_envs.VLLM_TUNED_CONFIG_FOLDER, json_name))
    paths.append(default_path)

    found = next((p for p in paths if os.path.isfile(p)), None)
    if found:
        print(
            f"[vllm_fused_moe] preflight: using tuned config {found}",
            flush=True,
        )
    else:
        print(
            "[vllm_fused_moe] preflight: no tuned fused_moe JSON for this "
            f"(E={E}, N={N}); using defaults (see also {default_path})",
            flush=True,
        )


def warmup_vllm_fused_moe_from_checkpoint(model_path: str, device_index: int = 0) -> None:
    """
    Run one ``fused_experts`` call with shapes from ``config.json`` so Triton JIT and config
    resolution happen before TTFT timers (e.g. in ``jiuge.py``).

    Set ``INFINILM_SKIP_VLLM_FUSED_MOE_PREFLIGHT=1`` to disable.

    Optional ``INFINILM_VLLM_FUSED_WARMUP_MAX_BYTES`` (integer): skip allocating dummy expert
    weights when the estimate exceeds this budget (TTFT may then include one-time Triton JIT).
    """
    if os.environ.get("INFINILM_SKIP_VLLM_FUSED_MOE_PREFLIGHT") == "1":
        return

    cfg = _load_hf_config_json(model_path)
    dims = _moe_dims_from_config(cfg)
    if dims is None:
        return

    try:
        import torch

        import infinicore.vendor.vllm_fused_moe as _vmoe  # noqa: F401
        from infinicore.vendor.vllm_fused_moe import MoEActivation, fused_experts
    except ImportError:
        return

    E, N, H, topk = dims
    if not torch.cuda.is_available():
        return

    torch_dtype = _torch_dtype_from_hf(cfg)
    torch.cuda.set_device(device_index)
    device = torch.device("cuda", device_index)

    num_tokens = min(128, max(32, topk * 8))
    est = _estimated_fused_moe_warmup_bytes(E, N, H, topk, num_tokens, torch_dtype)
    max_b = os.environ.get("INFINILM_VLLM_FUSED_WARMUP_MAX_BYTES")
    if max_b is not None:
        try:
            if est > int(max_b):
                print(
                    f"[vllm_fused_moe] preflight: skip fused_experts warmup "
                    f"(estimated {est} bytes > INFINILM_VLLM_FUSED_WARMUP_MAX_BYTES={max_b})",
                    flush=True,
                )
                return
        except ValueError:
            pass

    hidden_states = torch.randn(num_tokens, H, device=device, dtype=torch_dtype)
    w1 = torch.randn(E, 2 * N, H, device=device, dtype=torch_dtype)
    w2 = torch.randn(E, H, N, device=device, dtype=torch_dtype)
    topk_weights = torch.randn(num_tokens, topk, device=device, dtype=torch_dtype)
    topk_ids = torch.randint(0, E, (num_tokens, topk), device=device, dtype=torch.int32)

    torch.cuda.current_stream().synchronize()

    try:
        _ = fused_experts(
            hidden_states,
            w1,
            w2,
            topk_weights,
            topk_ids,
            inplace=False,
            activation=MoEActivation.SILU,
            apply_router_weight_on_input=False,
        )
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as e:
        print(
            f"[vllm_fused_moe] preflight: fused_experts warmup OOM (skip): {e}",
            flush=True,
        )


def preflight_vllm_fused_moe_for_ttft(model_path: str, device_index: int = 0) -> None:
    """Verify fused_moe JSON presence and warm up kernels before timed generate."""
    verify_vllm_fused_moe_config_for_checkpoint(model_path, device_index=device_index)
    warmup_vllm_fused_moe_from_checkpoint(model_path, device_index=device_index)
