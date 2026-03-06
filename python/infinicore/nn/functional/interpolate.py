from __future__ import annotations

from collections.abc import Iterable

from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _to_int64_list(value) -> list[int]:
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, Iterable):
        return [int(v) for v in value]
    raise TypeError(f"Expected int or iterable of ints, got {type(value).__name__}")


def _to_double_list(value) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, Iterable):
        return [float(v) for v in value]
    raise TypeError(
        f"Expected float or iterable of floats, got {type(value).__name__}"
    )


def interpolate(
    input: Tensor,
    size=None,
    scale_factor=None,
    mode: str = "nearest",
    align_corners=None,
) -> Tensor:
    size_list: list[int] = [] if size is None else _to_int64_list(size)
    scale_list: list[float] = [] if scale_factor is None else _to_double_list(scale_factor)

    if bool(size_list) == bool(scale_list):
        raise ValueError("Expected exactly one of size or scale_factor")

    align_i = 0 if align_corners is None else int(bool(align_corners))

    return Tensor(
        _infinicore.interpolate(input._underlying, str(mode), size_list, scale_list, align_i)
    )

