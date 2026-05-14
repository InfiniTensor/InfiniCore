# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch


def moe_sum(inp: torch.Tensor, out: torch.Tensor) -> None:
    """Reduce over top-k expert outputs (dim=1). Matches vLLM `_moe_C.moe_sum` layout."""
    torch.sum(inp, dim=1, out=out)
