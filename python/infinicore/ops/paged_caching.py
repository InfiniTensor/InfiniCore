from infinicore.lib import _infinicore
from infinicore.tensor import Tensor


def paged_caching(
    k: Tensor,
    v: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
    slot_mapping: Tensor,
):
    Tensor(
        _infinicore.paged_caching_(
            k._underlying,
            v._underlying,
            k_cache._underlying,
            v_cache._underlying,
            slot_mapping._underlying,
        )
    )
    return (k_cache, v_cache)
