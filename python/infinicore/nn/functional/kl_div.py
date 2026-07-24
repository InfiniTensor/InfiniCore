import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


_REDUCTION_TO_INT = {
    "none": 0,
    "mean": 1,
    "sum": 2,
    "batchmean": 3,
}


def kl_div(
    input: Tensor,
    target: Tensor,
    reduction: str = "mean",
    log_target: bool = False,
    *,
    out=None,
) -> Tensor:
    r"""Compute the Kullback-Leibler divergence loss."""

    reduction_i = _REDUCTION_TO_INT.get(reduction)
    if reduction_i is None:
        raise ValueError(f"Unsupported reduction: {reduction!r}")

    if infinicore.use_ntops and input.device.type in ("cuda", "musa") and out is None:
        result = infinicore.ntops.torch.kl_div(
            input,
            target,
            reduction=reduction,
            log_target=log_target,
        )

        # ntops kernel 为了避免 0-dim output pointer 问题，返回 shape (1,)。
        # PyTorch kl_div(reduction="sum"/"batchmean"/"mean") 返回 scalar shape ()。
        if reduction != "none":
            return infinicore.squeeze(result, 0)

        return result

    if out is None:
        return Tensor(
            _infinicore.kl_div(
                input._underlying,
                target._underlying,
                int(reduction_i),
                bool(log_target),
            )
        )

    _infinicore.kl_div_(
        out._underlying,
        input._underlying,
        target._underlying,
        int(reduction_i),
        bool(log_target),
    )

    return out