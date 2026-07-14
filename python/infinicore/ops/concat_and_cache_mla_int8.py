from infinicore.lib import _infinicore


def concat_and_cache_mla_int8(
    kv_c_int8, kv_c_scale, k_pe_int8, k_pe_scale, kv_cache, kv_cache_scale, slot_mapping
):
    """Write pre-quantized MLA latent/rope KV and scales into int8 paged cache."""
    _infinicore.concat_and_cache_mla_int8_(
        kv_c_int8._underlying,
        kv_c_scale._underlying,
        k_pe_int8._underlying,
        k_pe_scale._underlying,
        kv_cache._underlying,
        kv_cache_scale._underlying,
        slot_mapping._underlying,
    )
    return kv_cache, kv_cache_scale
