import infinicore
from infinicore.tensor import Tensor


def _hsplit_dim(input: Tensor) -> int:
    if input.ndim == 0:
        raise RuntimeError("hsplit expects a tensor with at least 1 dimension")

    return 0 if input.ndim == 1 else 1


def _normalize_index(index: int, size: int) -> int:
    if index < 0:
        index += size

    return max(0, min(index, size))


def _get_split_ranges(size: int, indices_or_sections):
    if isinstance(indices_or_sections, int):
        sections = indices_or_sections

        if sections <= 0:
            raise RuntimeError("number of sections must be larger than 0")

        if size % sections != 0:
            raise RuntimeError(
                "torch.hsplit with integer sections requires equal division"
            )

        step = size // sections
        return [(i * step, (i + 1) * step) for i in range(sections)]

    indices = [_normalize_index(int(index), size) for index in indices_or_sections]

    starts = [0] + indices
    ends = indices + [size]

    return list(zip(starts, ends))


def hsplit(
    input: Tensor,
    indices_or_sections,
) -> tuple[Tensor, ...]:
    r"""Split input into multiple tensors horizontally."""

    if infinicore.use_ntops and input.device.type in ("cuda", "musa"):
        return infinicore.ntops.torch.hsplit(input, indices_or_sections)

    dim = _hsplit_dim(input)
    ranges = _get_split_ranges(input.shape[dim], indices_or_sections)

    outputs = []

    for start, end in ranges:
        slices = [slice(None)] * input.ndim
        slices[dim] = slice(start, end)
        outputs.append(input[tuple(slices)])

    return tuple(outputs)