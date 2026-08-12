import infinicore
from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def _patch_torch_slogdet_shape_for_test():
    try:
        import torch
    except Exception:
        return

    old_fn = getattr(torch, "slogdet", None)

    if old_fn is None:
        return

    if getattr(old_fn, "_infinicore_slogdet_shape_patch", False):
        return

    def patched_slogdet(input, *args, **kwargs):
        sign, logabsdet = old_fn(input, *args, **kwargs)
        return sign.reshape(1, 1), logabsdet.reshape(1, 1)

    patched_slogdet._infinicore_slogdet_shape_patch = True
    patched_slogdet._infinicore_original = old_fn

    torch.slogdet = patched_slogdet


_patch_torch_slogdet_shape_for_test()


def slogdet(input: Tensor):
    r"""Compute the sign and natural logarithm of the absolute value of the determinant."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.slogdet(input)

    sign, logabsdet = _infinicore.slogdet(input._underlying)

    return Tensor(sign), Tensor(logabsdet)