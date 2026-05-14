# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch.library import Library, infer_schema

from .platform_ctx import current_platform

infinilm_fused_lib = Library("infinilm", "FRAGMENT")


def direct_register_custom_op(
    op_name: str,
    op_func: Callable[..., Any],
    mutates_args: list[str] | None = None,
    fake_impl: Callable[..., Any] | None = None,
    target_lib: Library | None = None,
    dispatch_key: str | None = None,
    tags: tuple[torch.Tag, ...] = (),
):
    if mutates_args is None:
        mutates_args = []

    if dispatch_key is None:
        dispatch_key = current_platform.dispatch_key()

    schema_str = infer_schema(op_func, mutates_args=mutates_args)
    my_lib = target_lib or infinilm_fused_lib
    my_lib.define(op_name + schema_str, tags=tags)
    my_lib.impl(op_name, op_func, dispatch_key=dispatch_key)
    if fake_impl is not None:
        my_lib._register_fake(op_name, fake_impl)
