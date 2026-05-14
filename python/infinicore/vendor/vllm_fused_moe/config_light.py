# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import torch


def _get_config_dtype_str(
    dtype: torch.dtype,
    use_fp8_w8a8: bool = False,
    use_fp8_w8a16: bool = False,
    use_int8_w8a16: bool = False,
    use_int4_w4a16: bool = False,
    ocp_mx_scheme: str | None = None,
) -> str | None:
    if use_fp8_w8a8:
        return "fp8_w8a8"
    elif use_fp8_w8a16:
        return "fp8_w8a16"
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif use_int4_w4a16:
        return "int4_w4a16"
    elif ocp_mx_scheme is not None:
        return None
    elif dtype == torch.float:
        return "float32"
    return None


@dataclass
class FusedMoEQuantConfig:
    """Minimal quant bundle for fused_experts (unquantized MoE only in this vendor)."""

    use_fp8_w8a8: bool = False
    use_int8_w8a8: bool = False
    use_int8_w8a16: bool = False
    use_int4_w4a16: bool = False
    ocp_mx_scheme: str | None = None
    per_act_token_quant: bool = False
    block_shape: list[int] | None = None
    w1_scale: torch.Tensor | None = None
    w2_scale: torch.Tensor | None = None
    w1_zp: torch.Tensor | None = None
    w2_zp: torch.Tensor | None = None
    a1_scale: torch.Tensor | None = None
    a2_scale: torch.Tensor | None = None
    w1_bias: torch.Tensor | None = None
    w2_bias: torch.Tensor | None = None

    def config_name(self, dtype: torch.dtype) -> str | None:
        return _get_config_dtype_str(
            use_fp8_w8a8=self.use_fp8_w8a8,
            use_fp8_w8a16=False,
            use_int8_w8a16=self.use_int8_w8a16,
            use_int4_w4a16=self.use_int4_w4a16,
            ocp_mx_scheme=self.ocp_mx_scheme,
            dtype=dtype,
        )


FUSED_MOE_UNQUANTIZED_CONFIG = FusedMoEQuantConfig()
