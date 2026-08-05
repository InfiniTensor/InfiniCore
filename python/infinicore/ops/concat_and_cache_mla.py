from infinicore.lib import _infinicore


def concat_and_cache_mla(
    kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype="auto", scale=None
):
    """Write concat([kv_c, k_pe], -1) into paged MLA KV cache at slot_mapping."""
    if scale is None:
        raise ValueError(
            "concat_and_cache_mla requires a float32 scale tensor; pass a scalar tensor for kv_cache_dtype='auto'"
        )
    _infinicore.concat_and_cache_mla_(
        kv_c._underlying,
        k_pe._underlying,
        kv_cache._underlying,
        slot_mapping._underlying,
        kv_cache_dtype,
        scale._underlying,
    )
    return kv_cache
