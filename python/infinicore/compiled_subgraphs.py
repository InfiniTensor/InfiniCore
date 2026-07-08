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
) -> None:
    from infinicore.lib import _infinicore as _ic

    _ic.register_piecewise_inductor_package(
        segment,
        int(layer_idx),
        int(bucket),
        os.path.abspath(package_path),
        int(tp_rank),
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
) -> int:
    """Register existing ``segment.pt2`` artifacts with the C++ runtime registry."""
    from infinilm.compile.piecewise_segments import piecewise_inductor_package_path

    tp_size = max(1, int(tp_size))
    ranks = list(tp_ranks) if tp_ranks is not None else list(range(tp_size))
    registered = 0
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
                    )
                    registered += 1
    _agent_log(
        "compiled_subgraphs.py:register_piecewise_inductor_packages",
        "registered_packages",
        "H3",
        {"registered": registered, "tp_size": tp_size, "ranks": ranks},
    )
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
) -> int:
    """Optional Python-side warmup via ``aoti_load_package`` (validates loadability)."""
    from infinilm.compile.piecewise_segments import piecewise_inductor_package_path

    try:
        from torch._inductor import aoti_load_package
    except ImportError:
        logger.warning("torch._inductor.aoti_load_package unavailable; skip warmup")
        return 0

    tp_size = max(1, int(tp_size))
    if tp_device_ids is None:
        tp_device_ids = list(range(tp_size))

    warmed = 0
    for tp_rank in range(tp_size):
        dev_index = int(tp_device_ids[tp_rank]) if tp_rank < len(tp_device_ids) else device_index
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
    )
    from infinilm.compile.piecewise_segments import SEGMENT_PRE_ATTN

    tp_size = max(1, int(tp_size))

    if num_layers is None:
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            hf_config = json.load(f)
        num_layers = int(hf_config.get("num_hidden_layers", 0))
    if num_layers <= 0:
        raise ValueError(f"invalid num_hidden_layers for model_path={model_path}")

    if buckets is None:
        buckets = native_piecewise_capture_buckets(compile_max_seq_len())

    root = cache_root or piecewise_inductor_cache_root()
    segments = (SEGMENT_PRE_ATTN,)
    layer_indices = range(num_layers)

    registered = register_piecewise_inductor_packages(
        model_path=model_path,
        segments=segments,
        layer_indices=layer_indices,
        buckets=buckets,
        cache_root=root,
        tp_size=tp_size,
    )
    warmed = 0
    if warmup and registered > 0:
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
        )

    logger.info(
        "piecewise inductor bootstrap: registered=%s warmed=%s tp_size=%s layers=%s buckets=%s cache=%s",
        registered,
        warmed,
        tp_size,
        num_layers,
        list(buckets),
        root,
    )
    _agent_log(
        "compiled_subgraphs.py:bootstrap_from_infinicore_device",
        "bootstrap_done",
        "H3",
        {
            "registered": registered,
            "warmed": warmed,
            "tp_size": tp_size,
            "num_layers": num_layers,
            "buckets": list(buckets),
        },
    )
    return {
        "registered": registered,
        "warmed": warmed,
        "tp_size": tp_size,
        "num_layers": num_layers,
        "buckets": list(buckets),
    }
