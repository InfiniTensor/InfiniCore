# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import os

# Prefer InfiniLM env; keep VLLM_* as fallback for existing workflows.
INFINILM_TUNED_CONFIG_FOLDER = os.environ.get("INFINILM_TUNED_CONFIG_FOLDER")
VLLM_TUNED_CONFIG_FOLDER = INFINILM_TUNED_CONFIG_FOLDER or os.environ.get(
    "VLLM_TUNED_CONFIG_FOLDER"
)

VLLM_BATCH_INVARIANT = os.environ.get("VLLM_BATCH_INVARIANT", "0") == "1"
