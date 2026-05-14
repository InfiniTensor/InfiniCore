"""
Bridge: run fused MoE experts on ATen views of InfiniCore tensors.

Default (**``INFINILM_MOE_FUSED_STACK=vendor``** or **``vendor_router_cpu``**): vendored vLLM-derived Triton in
``infinicore.vendor.vllm_fused_moe`` (registers ``torch.ops.infinilm.*``).

**``INFINILM_MOE_FUSED_STACK=upstream``** (``.venv-vllm``): ``fused_experts`` from
``vllm.model_executor.layers.fused_moe.fused_moe``; falls back to the vendor path if the import fails.

Requires InfiniCore built with ``--aten=y``, CUDA, and (for vendor) ``triton`` + the vendor module.
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

from infinicore.lib import _infinicore
from infinicore.moe_fused_stack import resolve_moe_fused_stack
from infinicore.tensor import from_torch, to_torch

if TYPE_CHECKING:
    from infinicore.tensor import Tensor


def _require_aten_bridge() -> None:
    if getattr(_infinicore, "_tensor_as_torch", None) is None:
        raise RuntimeError(
            "vllm_fused_moe_bridge requires InfiniCore with ATen enabled "
            "(rebuild with --aten=y)."
        )


_upstream_fused_import_warned: bool = False


def _get_fused_experts_and_activation():
    """
    Return ``(fused_experts, activation_enum_type)`` for the effective MoE stack.

    Upstream import failure falls back to vendored experts (warn once).
    """
    global _upstream_fused_import_warned
    stack = resolve_moe_fused_stack()
    if stack == "upstream":
        try:
            from vllm.model_executor.layers.fused_moe.activation import (  # type: ignore[import-not-found]
                MoEActivation as _UpstreamMoEActivation,
            )
            from vllm.model_executor.layers.fused_moe.fused_moe import (  # type: ignore[import-not-found]
                fused_experts as _upstream_fused_experts,
            )

            return _upstream_fused_experts, _UpstreamMoEActivation
        except ImportError:
            if not _upstream_fused_import_warned:
                _upstream_fused_import_warned = True
                print(
                    "[infinilm] MoE: upstream fused_experts (vLLM) not importable; "
                    "falling back to vendored Triton fused_experts",
                    flush=True,
                )
    try:
        import infinicore.vendor.vllm_fused_moe as _vmoe  # noqa: F401 — registers torch.ops.infinilm
        from infinicore.vendor.vllm_fused_moe import MoEActivation, fused_experts
    except ImportError as e:
        raise RuntimeError(
            "fused_experts_ic requires InfiniLM vendored fused MoE + Triton."
        ) from e
    return fused_experts, MoEActivation


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

    import torch

    fused_experts_fn, activation_enum = _get_fused_experts_and_activation()

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

    out_t = fused_experts_fn(
        h,
        t_w1,
        t_w2,
        t_tw,
        t_ids,
        inplace=inplace,
        activation=activation_enum.SILU,
        apply_router_weight_on_input=apply_router_weight_on_input,
    )
    return from_torch(out_t)


def grouped_sigmoid_topk_ic(
    router_logits: Tensor,
    e_score_correction_bias: Tensor,
    *,
    topk: int,
    renormalize: bool,
    num_expert_group: int,
    topk_group: int,
    routed_scaling_factor: float,
) -> tuple[Tensor, Tensor]:
    """
    Grouped sigmoid + e_score_correction_bias routing on InfiniCore tensors via ``torch.ops.infinilm``.

    Returns ``(topk_weights, topk_ids)`` as InfiniCore tensors (float32 weights, int32 ids), same contract
    as vLLM ``grouped_topk`` with ``scoring_func="sigmoid"``.
    """
    _require_aten_bridge()

    try:
        import torch

        import infinicore.vendor.vllm_fused_moe as _vmoe  # noqa: F401 — registers minicpm5_grouped_sigmoid_topk
    except ImportError as e:
        raise RuntimeError(
            "grouped_sigmoid_topk_ic requires InfiniLM vendored fused MoE (infinicore.vendor.vllm_fused_moe)."
        ) from e

    rl = to_torch(router_logits)
    bias = to_torch(e_score_correction_bias)
    if not rl.is_contiguous():
        rl = rl.contiguous()
    if not bias.is_contiguous():
        bias = bias.contiguous()
    if torch.cuda.is_available() and rl.is_cuda:
        torch.cuda.current_stream().synchronize()

    tw, tid = torch.ops.infinilm.minicpm5_grouped_sigmoid_topk(
        rl,
        bias,
        int(topk),
        bool(renormalize),
        int(num_expert_group),
        int(topk_group),
        float(routed_scaling_factor),
    )
    return from_torch(tw.contiguous()), from_torch(tid.contiguous())


def grouped_sigmoid_topk_ic_cpp(
    router_logits: Tensor,
    e_score_correction_bias: Tensor,
    top_k: int,
    norm_topk_prob: bool,
    routed_scaling_factor: float,
    n_group: int,
    topk_group: int,
) -> tuple[Tensor, Tensor]:
    """C++-friendly positional wrapper (no keyword-only args)."""
    return grouped_sigmoid_topk_ic(
        router_logits,
        e_score_correction_bias,
        topk=int(top_k),
        renormalize=bool(norm_topk_prob),
        num_expert_group=int(n_group),
        topk_group=int(topk_group),
        routed_scaling_factor=float(routed_scaling_factor),
    )


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
    vmoe_fused = None
    stack = resolve_moe_fused_stack()
    if stack == "upstream":
        try:
            import vllm.model_executor.layers.fused_moe.fused_moe as vmoe_fused  # type: ignore[import-not-found]
        except ImportError:
            stack = "vendor"
    if stack in ("vendor", "vendor_router_cpu"):
        try:
            import infinicore.vendor.vllm_fused_moe.envs as moe_envs
            import infinicore.vendor.vllm_fused_moe.fused_moe as vmoe_fused
        except ImportError:
            print(
                "[vllm_fused_moe] preflight: vendored fused_moe not importable; skip config check",
                flush=True,
            )
            return

    get_config_file_name = vmoe_fused.get_config_file_name

    json_name = get_config_file_name(E, N, None, None)
    fused_moe_dir = os.path.dirname(vmoe_fused.__file__)
    default_path = os.path.join(fused_moe_dir, "configs", json_name)
    paths = []
    tuned_root = os.environ.get("INFINILM_TUNED_CONFIG_FOLDER") or os.environ.get("VLLM_TUNED_CONFIG_FOLDER")
    if tuned_root:
        paths.append(os.path.join(tuned_root, json_name))
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

        fused_experts, MoEActivation = _get_fused_experts_and_activation()
    except ImportError:
        return
    except RuntimeError:
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
