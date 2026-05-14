# SPDX-License-Identifier: Apache-2.0

"""vLLM-derived fused MoE (Triton) registered as ``torch.ops.infinilm.*``."""

from __future__ import annotations

# Import for side effects: registers torch.library fragments on ``infinilm``.
from . import fused_moe as _fused_moe  # noqa: F401
from . import minicpm5_grouped_sigmoid_topk as _minicpm5_grouped_sigmoid_topk  # noqa: F401
from .activation import MoEActivation
from .fused_moe import fused_experts, get_config_file_name

__all__ = ["MoEActivation", "fused_experts", "get_config_file_name"]
