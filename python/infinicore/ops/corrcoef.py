import infinicore
from infinicore.tensor import Tensor


def corrcoef(input: Tensor) -> Tensor:
    r"""Estimate a Pearson correlation coefficient matrix."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        result = infinicore.ntops.torch.corrcoef(input)

        # torch.corrcoef(1D) returns scalar.
        # torch.corrcoef(shape=(1, N)) also returns scalar.
        if input.ndim == 1 or (input.ndim == 2 and input.shape[0] == 1):
            return infinicore.squeeze(result, 0)

        return result

    raise NotImplementedError(
        "corrcoef is only implemented through ntops; "
        "_infinicore.corrcoef is not available."
    )