# Copyright (c) 2025, InfiniCore
"""Bootstrap AOTInductor piecewise segment packages for native CG replay."""

from __future__ import annotations

import json
import logging
import os
from typing import Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

def piecewise_inductor_segment_enabled() -> bool:
    from infinilm.compile.env import piecewise_inductor_segment_enabled as _enabled

    return _enabled()


def any_compiled_subgraph_flag_enabled() -> bool:
    return piecewise_inductor_segment_enabled()


def _register_package(
    segment: str,
    layer_idx: int,
    bucket: int,
    package_path: str,
    tp_rank: int = 0,
    *,
    layer_agnostic: bool = False,
) -> None:
    from infinicore.lib import _infinicore as _ic

    reg_layer = -1 if layer_agnostic else int(layer_idx)
    _ic.register_piecewise_inductor_package(
        segment,
        reg_layer,
        int(bucket),
        os.path.abspath(package_path),
        int(tp_rank),
        bool(layer_agnostic),
    )


def register_piecewise_inductor_packages(
    *,
    model_path: str,
    segments: Sequence[str],
    layer_indices: Iterable[int],
    buckets: Iterable[int],
    cache_root: Optional[str] = None,
    tp_size: int = 1,
    tp_ranks: Optional[Sequence[int]] = None,
    layer_agnostic: Optional[bool] = None,
) -> int:
    """Register existing ``segment.pt2`` artifacts with the C++ runtime registry."""
    from infinilm.compile.piecewise_segments import (
        LAYER_AGNOSTIC_IDX,
        piecewise_inductor_package_path,
        piecewise_layer_agnostic_enabled,
    )

    if layer_agnostic is None:
        layer_agnostic = piecewise_layer_agnostic_enabled()

    tp_size = max(1, int(tp_size))
    ranks = list(tp_ranks) if tp_ranks is not None else list(range(tp_size))
    registered = 0

    if layer_agnostic:
        for segment in segments:
            for tp_rank in ranks:
                for bucket in buckets:
                    package_path = piecewise_inductor_package_path(
                        cache_root=cache_root,
                        model_path=model_path,
                        segment=segment,
                        layer_idx=LAYER_AGNOSTIC_IDX,
                        bucket=int(bucket),
                        tp_size=tp_size,
                        tp_rank=int(tp_rank),
                        layer_agnostic=True,
                    )
                    if not os.path.isfile(package_path):
                        logger.debug(
                            "piecewise inductor: skip missing layer-agnostic package "
                            "segment=%s tp%d/rank%d B%s path=%s",
                            segment,
                            tp_size,
                            tp_rank,
                            bucket,
                            package_path,
                        )
                        continue
                    _register_package(
                        segment,
                        LAYER_AGNOSTIC_IDX,
                        int(bucket),
                        package_path,
                        tp_rank=int(tp_rank),
                        layer_agnostic=True,
                    )
                    registered += 1
        return registered

    for segment in segments:
        for tp_rank in ranks:
            for layer_idx in layer_indices:
                for bucket in buckets:
                    package_path = piecewise_inductor_package_path(
                        cache_root=cache_root,
                        model_path=model_path,
                        segment=segment,
                        layer_idx=int(layer_idx),
                        bucket=int(bucket),
                        tp_size=tp_size,
                        tp_rank=int(tp_rank),
                        layer_agnostic=False,
                    )
                    if not os.path.isfile(package_path):
                        logger.debug(
                            "piecewise inductor: skip missing package segment=%s "
                            "tp%d/rank%d L%s B%s path=%s",
                            segment,
                            tp_size,
                            tp_rank,
                            layer_idx,
                            bucket,
                            package_path,
                        )
                        continue
                    _register_package(
                        segment,
                        int(layer_idx),
                        int(bucket),
                        package_path,
                        tp_rank=int(tp_rank),
                        layer_agnostic=False,
                    )
                    registered += 1
    return registered


def warmup_piecewise_inductor_packages(
    *,
    model_path: str,
    device_index: int,
    segments: Sequence[str],
    layer_indices: Iterable[int],
    buckets: Iterable[int],
    cache_root: Optional[str] = None,
    tp_size: int = 1,
    tp_device_ids: Optional[Sequence[int]] = None,
    layer_agnostic: Optional[bool] = None,
) -> int:
    """Optional Python-side warmup via ``aoti_load_package`` (validates loadability)."""
    from infinilm.compile.piecewise_segments import (
        LAYER_AGNOSTIC_IDX,
        piecewise_inductor_package_path,
        piecewise_layer_agnostic_enabled,
    )

    try:
        from torch._inductor import aoti_load_package
    except ImportError:
        logger.warning("torch._inductor.aoti_load_package unavailable; skip warmup")
        return 0

    if layer_agnostic is None:
        layer_agnostic = piecewise_layer_agnostic_enabled()

    tp_size = max(1, int(tp_size))
    if tp_device_ids is None:
        tp_device_ids = list(range(tp_size))

    warmed = 0
    for tp_rank in range(tp_size):
        dev_index = int(tp_device_ids[tp_rank]) if tp_rank < len(tp_device_ids) else device_index
        if layer_agnostic:
            for segment in segments:
                for bucket in buckets:
                    package_path = piecewise_inductor_package_path(
                        cache_root=cache_root,
                        model_path=model_path,
                        segment=segment,
                        layer_idx=LAYER_AGNOSTIC_IDX,
                        bucket=int(bucket),
                        tp_size=tp_size,
                        tp_rank=tp_rank,
                        layer_agnostic=True,
                    )
                    if not os.path.isfile(package_path):
                        continue
                    try:
                        aoti_load_package(
                            package_path,
                            run_single_threaded=True,
                            device_index=dev_index,
                        )
                        warmed += 1
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "piecewise inductor warmup failed segment=%s tp_rank=%s B%s: %s",
                            segment,
                            tp_rank,
                            bucket,
                            exc,
                        )
            continue

        for segment in segments:
            for layer_idx in layer_indices:
                for bucket in buckets:
                    package_path = piecewise_inductor_package_path(
                        cache_root=cache_root,
                        model_path=model_path,
                        segment=segment,
                        layer_idx=int(layer_idx),
                        bucket=int(bucket),
                        tp_size=tp_size,
                        tp_rank=tp_rank,
                        layer_agnostic=False,
                    )
                    if not os.path.isfile(package_path):
                        continue
                    try:
                        aoti_load_package(
                            package_path,
                            run_single_threaded=True,
                            device_index=dev_index,
                        )
                        warmed += 1
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "piecewise inductor warmup failed segment=%s tp_rank=%s L%s B%s: %s",
                            segment,
                            tp_rank,
                            layer_idx,
                            bucket,
                            exc,
                        )
    return warmed


def _assert_moe_compile_on_miss_disabled() -> None:
    from infinilm.tools.moe_serve_bootstrap import assert_moe_compile_on_miss_disabled

    assert_moe_compile_on_miss_disabled()


def _validate_minicpm5_moe_artifacts(*, model_path: str, cache_root: str) -> dict:
    from infinilm.tools.moe_serve_bootstrap import validate_minicpm5_moe_artifacts

    return validate_minicpm5_moe_artifacts(model_path=model_path, cache_root=cache_root)


def bootstrap_from_infinicore_device(
    *,
    infini_device,
    hidden_size: int,
    dtype,
    warmup: bool = True,
    model_path: Optional[str] = None,
    num_layers: Optional[int] = None,
    buckets: Optional[Sequence[int]] = None,
    cache_root: Optional[str] = None,
    tp_size: int = 1,
    tp_device_ids: Optional[Sequence[int]] = None,
    compile_on_miss: bool = False,
) -> dict:
    """Register (and optionally warm up) piecewise AOT packages for server init.

    Register-only: never compile-on-miss. Missing packages raise unless
    ``INFINI_AOT_CHECK_SKIP=1``. Offline compile via
    ``python -m infinilm.server.entry --phase compile|all``.
    """
    del hidden_size, dtype  # reserved for future post_attn segment example inputs
    del compile_on_miss  # deprecated; InferEngine never compiles
    if not piecewise_inductor_segment_enabled():
        return {"registered": 0, "warmed": 0}

    if model_path is None:
        raise ValueError("bootstrap_from_infinicore_device requires model_path")

    from infinilm.compile.env import (
        aot_check_skip,
        piecewise_inductor_compile_on_miss,
        piecewise_inductor_cache_root,
        prefill_native_cg_enabled,
    )
    from infinilm.compile.piecewise_bootstrap_plan import (
        build_piecewise_bootstrap_plan,
        missing_packages_error_message,
    )

    # Trigger one-shot DeprecationWarning if COM=1; always ignored.
    piecewise_inductor_compile_on_miss()

    if aot_check_skip():
        logger.warning(
            "INFINI_AOT_CHECK_SKIP=1: skipping piecewise AOT register/check"
        )
        return {"registered": 0, "warmed": 0, "skipped": True}

    tp_size = max(1, int(tp_size))
    root = cache_root or piecewise_inductor_cache_root()
    plan = build_piecewise_bootstrap_plan(
        model_path=model_path,
        tp_size=tp_size,
        buckets=buckets,
        cache_root=root,
        num_layers=num_layers,
    )
    model_type = plan.model_type
    num_layers = plan.num_layers
    buckets = plan.buckets
    segments = plan.segments
    layer_agnostic = plan.layer_agnostic
    moe_artifact_info: Optional[dict] = None

    if model_type == "minicpm5_moe":
        from infinilm.compile.piecewise_moe_segment import moe_routing_hparams
        from infinilm.torch_llama.moe_ops import (
            configure_moe_block_routing,
            register_fused_moe_routed_op,
        )

        register_fused_moe_routed_op()
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            hf_config = json.load(f)
        hp = moe_routing_hparams(hf_config)
        configure_moe_block_routing(
            top_k=hp["num_experts_per_tok"],
            n_group=hp["n_group"],
            topk_group=hp["topk_group"],
            norm_topk_prob=hp["norm_topk_prob"],
            routed_scaling_factor=hp["routed_scaling_factor"],
        )
        os.environ["INFINI_MOE_TOP_K"] = str(hp["num_experts_per_tok"])
        os.environ["INFINI_MOE_N_GROUP"] = str(hp["n_group"])
        os.environ["INFINI_MOE_TOPK_GROUP"] = str(hp["topk_group"])
        os.environ["INFINI_MOE_ROUTED_SCALING"] = str(hp["routed_scaling_factor"])
        os.environ["INFINI_MOE_NORM_TOPK"] = "1" if hp["norm_topk_prob"] else "0"
        _assert_moe_compile_on_miss_disabled()
        moe_artifact_info = _validate_minicpm5_moe_artifacts(
            model_path=model_path, cache_root=root
        )

    missing = (
        plan.missing_required()
        if model_type == "minicpm5_moe"
        else plan.missing_packages()
    )
    if missing:
        raise RuntimeError(missing_packages_error_message(plan, missing))

    # register_piecewise uses layer_indices only when not layer-agnostic.
    layer_indices = range(num_layers)

    if model_type == "minicpm5_moe" and plan.required_buckets:
        # Strict count for DEFAULT_BUCKETS × ranks (layer-agnostic: one pkg per bucket×rank).
        expected = len(plan.required_buckets) * (
            1 if layer_agnostic else num_layers
        ) * tp_size
    else:
        expected = plan.expected_count

    registered = register_piecewise_inductor_packages(
        model_path=model_path,
        segments=segments,
        layer_indices=layer_indices,
        buckets=buckets,
        cache_root=root,
        tp_size=tp_size,
        layer_agnostic=layer_agnostic,
    )
    warmed = 0
    skip_python_warmup = (
        piecewise_inductor_segment_enabled() and prefill_native_cg_enabled()
    )
    if skip_python_warmup and warmup and registered > 0:
        logger.info(
            "piecewise inductor bootstrap: skip Python aoti_load warmup "
            "(native CG compile warms per-rank AOT runners)"
        )
    if warmup and registered > 0 and not skip_python_warmup:
        device_index = getattr(infini_device, "index", 0)
        if device_index is None:
            device_index = 0
        warmed = warmup_piecewise_inductor_packages(
            model_path=model_path,
            device_index=int(device_index),
            segments=segments,
            layer_indices=layer_indices,
            buckets=buckets,
            cache_root=root,
            tp_size=tp_size,
            tp_device_ids=tp_device_ids,
            layer_agnostic=layer_agnostic,
        )

    if registered < expected:
        raise RuntimeError(
            f"piecewise inductor strict AOT: registered={registered} expected={expected} "
            f"cache={root} layer_agnostic={layer_agnostic} model_type={model_type}. "
            f"Run: python -m infinilm.server.entry --phase compile|all --model {model_path}"
        )

    logger.info(
        "piecewise inductor bootstrap: registered=%s warmed=%s expected=%s tp_size=%s "
        "layers=%s buckets=%s cache=%s layer_agnostic=%s",
        registered,
        warmed,
        expected,
        tp_size,
        num_layers,
        list(buckets),
        root,
        layer_agnostic,
    )
    out = {
        "registered": registered,
        "warmed": warmed,
        "expected": expected,
        "tp_size": tp_size,
        "num_layers": num_layers,
        "buckets": list(buckets),
        "layer_agnostic": layer_agnostic,
        "model_type": model_type,
    }
    if moe_artifact_info is not None:
        out["moe_artifacts"] = moe_artifact_info
    return out
