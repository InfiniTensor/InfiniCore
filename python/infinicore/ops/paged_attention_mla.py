from infinicore.lib import _infinicore


def paged_attention_mla_(
    output, query, kv_cache, scale, block_tables, context_lens, max_context_len
):
    _infinicore.paged_attention_mla_(
        output._underlying,
        query._underlying,
        kv_cache._underlying,
        scale,
        block_tables._underlying,
        context_lens._underlying,
        max_context_len,
    )
    return output
