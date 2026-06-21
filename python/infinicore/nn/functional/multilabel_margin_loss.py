import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _normalize_reduction(size_average=None, reduce=None, reduction="mean"):
    if size_average is not None or reduce is not None:
        if reduce is False:
            return "none"

        if size_average is False:
            return "sum"

        return "mean"

    return reduction


def multilabel_margin_loss(
    input: Tensor,
    target: Tensor,
    size_average=None,
    reduce=None,
    reduction: str = "mean",
    *,
    out=None,
) -> Tensor:
    r"""Compute multilabel margin loss.

    Args:
        input: Tensor with shape [C], [N, C], or higher dims flattened by ntops wrapper.
        target: LongTensor with same shape as input, padded by -1.
        reduction: "none", "mean", or "sum".
    """

    reduction = _normalize_reduction(
        size_average=size_average,
        reduce=reduce,
        reduction=reduction,
    )

    assert reduction in (
        "none",
        "mean",
        "sum",
    ), "`reduction` must be one of 'none', 'mean', or 'sum'."
    if (
        infinicore.use_ntops
        and input.device.type in ("cuda", "musa")
        and out is None
    ):
        return infinicore.ntops.torch.multilabel_margin_loss(
            input,
            target,
            size_average=size_average,
            reduce=reduce,
            reduction=reduction,
        )

    # C++ fallback
    if not hasattr(_infinicore, "multilabel_margin_loss"):
        raise NotImplementedError(
            "multilabel_margin_loss is not implemented in _infinicore, "
            "and ntops path is unavailable. Enable infinicore.use_ntops "
            "or add C++ backend implementation."
        )

    if out is None:
        return Tensor(
            _infinicore.multilabel_margin_loss(
                input._underlying,
                target._underlying,
                reduction,
            )
        )

    if not hasattr(_infinicore, "multilabel_margin_loss_"):
        raise NotImplementedError(
            "multilabel_margin_loss_ out variant is not implemented in _infinicore."
        )

    _infinicore.multilabel_margin_loss_(
        out._underlying,
        input._underlying,
        target._underlying,
        reduction,
    )

    return out