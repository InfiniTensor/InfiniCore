# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import torch


class _Platform:
    @staticmethod
    def get_device_name() -> str:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name()
        return "CPU"

    @staticmethod
    def is_rocm() -> bool:
        return torch.version.hip is not None

    @staticmethod
    def is_cuda() -> bool:
        return torch.cuda.is_available()

    @staticmethod
    def is_cuda_alike() -> bool:
        return torch.cuda.is_available()

    @staticmethod
    def is_xpu() -> bool:
        return hasattr(torch, "xpu") and torch.xpu.is_available()

    @staticmethod
    def dispatch_key() -> str:
        return "CUDA" if torch.cuda.is_available() else "CPU"

    @staticmethod
    def fp8_dtype():
        return torch.float8_e4m3fn


current_platform = _Platform()
