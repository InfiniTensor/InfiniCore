from infinicore.lib import _infinicore


def fused_rotary_embedding_(query, key, positions, head_size, cos_sin_cache, is_neox):
    _infinicore.fused_rotary_embedding_(
        query._underlying,
        key._underlying,
        positions._underlying,
        head_size,
        cos_sin_cache._underlying,
        is_neox,
    )
    return query, key
