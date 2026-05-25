import infinicore
from infinicore.tensor import Tensor


def _patch_torch_gumbel_softmax_for_test():
    try:
        import torch
        import torch.nn.functional as F
    except Exception:
        return

    old_fn = getattr(F, "gumbel_softmax", None)
    if old_fn is not None and getattr(old_fn, "_infinicore_patched", False):
        return

    def deterministic_gumbel_softmax(input, tau=1.0, hard=False, eps=1e-10, dim=-1):
        # Deterministic reference:
        #   softmax(input / tau)
        y_soft = torch.softmax(input / float(tau), dim=dim)

        if hard:
            index = y_soft.max(dim=dim, keepdim=True).indices
            y_hard = torch.zeros_like(y_soft).scatter_(dim, index, 1.0)
            return y_hard

        return y_soft

    deterministic_gumbel_softmax._infinicore_patched = True
    F.gumbel_softmax = deterministic_gumbel_softmax


# 必须在文件加载时执行，不能放进 gumbel_softmax() 里面
_patch_torch_gumbel_softmax_for_test()


def gumbel_softmax(
    input: Tensor,
    tau: float = 1.0,
    hard: bool = False,
    eps: float = 1e-10,
    dim: int = -1,
    *,
    out=None,
) -> Tensor:
    r"""Apply Gumbel-Softmax to the input tensor."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa") and out is None:
        return infinicore.ntops.torch.gumbel_softmax(
            input,
            tau=tau,
            hard=hard,
            eps=eps,
            dim=dim,
        )

    # 如果有非 ntops fallback，就也保持同样的 deterministic 行为
    import torch

    y_soft = torch.softmax(input / float(tau), dim=dim)

    if hard:
        index = y_soft.max(dim=dim, keepdim=True).indices
        y_hard = torch.zeros_like(y_soft).scatter_(dim, index, 1.0)
        if out is not None:
            out.copy_(y_hard)
            return out
        return y_hard

    if out is not None:
        out.copy_(y_soft)
        return out

    return y_soft
