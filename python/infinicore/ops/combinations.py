import math

import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _num_combinations(n: int, r: int, with_replacement: bool) -> int:
    if r < 0:
        raise ValueError("r must be non-negative")

    if r == 0:
        return 1

    if n == 0:
        return 0

    if with_replacement:
        return math.comb(n + r - 1, r)

    if r > n:
        return 0

    return math.comb(n, r)


def combinations(
    input: Tensor,
    r: int = 2,
    with_replacement: bool = False,
    *,
    out=None,
) -> Tensor:
    r"""Compute combinations of length ``r`` of the given 1-D tensor."""

    assert input.ndim == 1, "combinations only supports 1-D input"

    r = int(r)
    with_replacement = bool(with_replacement)

    assert r >= 0, "r must be non-negative"
    if infinicore.use_ntops and input.device.type in ("cuda", "musa") and out is None:
        return infinicore.ntops.torch.combinations(
            input,
            r=r,
            with_replacement=with_replacement,
        )

    if out is None:
        return Tensor(
            _infinicore.combinations(
                input._underlying,
                r,
                with_replacement,
            )
        )
    _infinicore.combinations_(
        out._underlying,
        input._underlying,
        r,
        with_replacement,
    )

    return out