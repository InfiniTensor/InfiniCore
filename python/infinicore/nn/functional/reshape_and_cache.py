from infinicore.lib import _infinicore
from infinicore.tensor import Tensor, empty



def reshape_and_cache(
    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    slot_mapping: Tensor,
    kv_cache_dtype:str,
    k_scale: Tensor,
    v_scale: Tensor ,
):
    _infinicore.reshape_and_cache(
        key._underlying,
        value._underlying,
        key_cache._underlying,
        value_cache._underlying,
        slot_mapping._underlying,
        kv_cache_dtype,
        k_scale._underlying,
        v_scale._underlying,
    )
