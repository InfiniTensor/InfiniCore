# Copyright (c) 2025, InfiniCore
"""Bootstrap AOTInductor piecewise segment packages for native CG replay."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Iterable, Optional, Sequence

logger = logging.getLogger(__name__)

_DEBUG_LOG = "/opt/offline/infinilm-metax-20260622/.cursor/debug-9ddc7d.log"


def _agent_log(location: str, message: str, hypothesis_id: str, data: dict) -> None:
    # #region agent log
    payload = {
        "sessionId": "9ddc7d",
        "location": location,
        "message": message,
        "hypothesisId": hypothesis_id,
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(_DEBUG_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except OSError:
        pass
    # #endregion


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
        _agent_log(
            "compiled_subgraphs.py:register_piecewise_inductor_packages",
            "registered_packages",
            "H3",
            {
                "registered": registered,
                "tp_size": tp_size,
                "ranks": ranks,
                "layer_agnostic": True,
            },
        )
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
    _agent_log(
        "compiled_subgraphs.py:register_piecewise_inductor_packages",
        "registered_packages",
        "H3",
        {"registered": registered, "tp_size": tp_size, "ranks": ranks},
    )
    return registered


def _compile_missing_packages(
    *,
    model_path: str,
    segments: Sequence[str],
    buckets: Sequence[int],
    cache_root: str,
    tp_size: int,
    tp_device_ids: Optional[Sequence[int]],
    layer_agnostic: bool,
) -> None:
    from infinilm.compile.piecewise_segments import (
        LAYER_AGNOSTIC_IDX,
        SEGMENT_PRE_ATTN,
        aot_compile_piecewise_segments_multi_bucket,
        piecewise_inductor_package_path,
    )
    from infinilm.compile.piecewise_moe_segment import (
        SEGMENT_MOE,
        aot_compile_minicpm5_moe_segment,
    )

    import torch

    for tp_rank in range(tp_size):
        dev_index = (
            int(tp_device_ids[tp_rank])
            if tp_device_ids is not None and tp_rank < len(tp_device_ids)
            else tp_rank
        )
        missing = []
        for segment in segments:
            for bucket in buckets:
                pkg = piecewise_inductor_package_path(
                    cache_root=cache_root,
                    model_path=model_path,
                    segment=segment,
                    layer_idx=LAYER_AGNOSTIC_IDX if layer_agnostic else 0,
                    bucket=int(bucket),
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    layer_agnostic=layer_agnostic,
                )
                if not os.path.isfile(pkg):
                    missing.append((segment, bucket))
        if not missing:
            continue
        logger.info(
            "piecewise inductor compile-on-miss: tp_rank=%s missing=%s",
            tp_rank,
            missing,
        )
        device = torch.device("cuda", dev_index)
        if SEGMENT_PRE_ATTN in segments and any(s == SEGMENT_PRE_ATTN for s, _ in missing):
            aot_compile_piecewise_segments_multi_bucket(
                model_path=model_path,
                segment=SEGMENT_PRE_ATTN,
                buckets=[int(b) for b in buckets],
                device=device,
                cache_root=cache_root,
                require_aot=True,
                tp_size=tp_size,
                tp_rank=tp_rank,
                tp_device_ids=tp_device_ids or list(range(tp_size)),
                layer_agnostic=layer_agnostic,
            )
        if SEGMENT_MOE in segments:
            moe_buckets = sorted({int(b) for s, b in missing if s == SEGMENT_MOE})
            for bucket in moe_buckets:
                aot_compile_minicpm5_moe_segment(
                    model_path=model_path,
                    bucket=bucket,
                    device=device,
                    cache_root=cache_root,
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    require_aot=True,
                )


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
    compile_on_miss: bool = True,
) -> dict:
    """Register (and optionally warm up) piecewise AOT packages for server init."""
    del hidden_size, dtype  # reserved for future post_attn segment example inputs
    if not piecewise_inductor_segment_enabled():
        return {"registered": 0, "warmed": 0}

    if model_path is None:
        raise ValueError("bootstrap_from_infinicore_device requires model_path")

    from infinilm.compile.env import (
        compile_max_seq_len,
        native_piecewise_capture_buckets,
        piecewise_inductor_cache_root,
        piecewise_inductor_compile_on_miss,
        piecewise_inductor_require_aot,
        prefill_native_cg_enabled,
    )
    from infinilm.compile.piecewise_segments import (
        SEGMENT_PRE_ATTN,
        expected_piecewise_package_count,
        piecewise_layer_agnostic_enabled,
    )
    from infinilm.compile.piecewise_moe_segment import SEGMENT_MOE

    tp_size = max(1, int(tp_size))
    layer_agnostic = piecewise_layer_agnostic_enabled()

    if num_layers is None:
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            hf_config = json.load(f)
        num_layers = int(hf_config.get("num_hidden_layers", 0))
    else:
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            hf_config = json.load(f)
    if num_layers <= 0:
        raise ValueError(f"invalid num_hidden_layers for model_path={model_path}")

    if buckets is None:
        buckets = native_piecewise_capture_buckets(compile_max_seq_len())

    root = cache_root or piecewise_inductor_cache_root()
    model_type = str(hf_config.get("model_type", ""))
    # Track B MiniCPM5: register MoE packages (pre_attn optional / separate).
    if model_type == "minicpm5_moe":
        segments = (SEGMENT_MOE,)
        # Union small MoE ladder with capture buckets so on-disk moe_B16/B64
        # are registered; C++ pick_moe_bucket walks this same ladder.
        moe_ladder = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
        buckets = tuple(sorted({int(b) for b in buckets} | set(moe_ladder)))
    else:
        segments = (SEGMENT_PRE_ATTN,)
    layer_indices = range(num_layers)
    # For MoE, "expected" is packages present on disk under the ladder (not
    # every ladder slot). Avoid false strict-AOT failure when only B16/B64/B512
    # have been compiled.
    if model_type == "minicpm5_moe":
        from infinilm.compile.piecewise_segments import (
            LAYER_AGNOSTIC_IDX,
            piecewise_inductor_package_path,
        )

        expected = 0
        for bucket in buckets:
            for tp_rank in range(tp_size):
                pkg = piecewise_inductor_package_path(
                    cache_root=root,
                    model_path=model_path,
                    segment=SEGMENT_MOE,
                    layer_idx=LAYER_AGNOSTIC_IDX if layer_agnostic else 0,
                    bucket=int(bucket),
                    tp_size=tp_size,
                    tp_rank=tp_rank,
                    layer_agnostic=layer_agnostic,
                )
                if os.path.isfile(pkg):
                    expected += 1 if layer_agnostic else num_layers
    else:
        expected = expected_piecewise_package_count(
            num_layers=num_layers,
            buckets=buckets,
            tp_size=tp_size,
            layer_agnostic=layer_agnostic,
        ) * len(segments)

    compile_on_miss = compile_on_miss and piecewise_inductor_compile_on_miss()
    if compile_on_miss:
        _compile_missing_packages(
            model_path=model_path,
            segments=segments,
            buckets=list(buckets),
            cache_root=root,
            tp_size=tp_size,
            tp_device_ids=tp_device_ids,
            layer_agnostic=layer_agnostic,
        )

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

    if piecewise_inductor_require_aot() and registered < expected:
        raise RuntimeError(
            f"piecewise inductor strict AOT: registered={registered} expected={expected} "
            f"cache={root} layer_agnostic={layer_agnostic}"
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
    _agent_log(
        "compiled_subgraphs.py:bootstrap_from_infinicore_device",
        "bootstrap_done",
        "H3",
        {
            "registered": registered,
            "warmed": warmed,
            "expected": expected,
            "tp_size": tp_size,
            "num_layers": num_layers,
            "buckets": list(buckets),
            "layer_agnostic": layer_agnostic,
        },
    )
    return {
        "registered": registered,
        "warmed": warmed,
        "expected": expected,
        "tp_size": tp_size,
        "num_layers": num_layers,
        "buckets": list(buckets),
        "layer_agnostic": layer_agnostic,
    }
