from infinicore.lib import _infinicore


def _t(tensor):
    return tensor._underlying


def fused_deepseek_v2_indexer_postprocess_(
    q_out,
    k_out,
    weights_out,
    kv_cache,
    slot_mapping,
    q,
    kw,
    norm_weight,
    norm_bias,
    positions,
    cos_sin_cache,
    num_cache_tokens,
    is_neox,
    eps,
    weights_scale,
):
    _infinicore.fused_deepseek_v2_indexer_postprocess_(
        *map(
            _t,
            (
                q_out,
                k_out,
                weights_out,
                kv_cache,
                slot_mapping,
                q,
                kw,
                norm_weight,
                norm_bias,
                positions,
                cos_sin_cache,
            ),
        ),
        num_cache_tokens,
        is_neox,
        eps,
        weights_scale,
    )


def indexer_k_cache_(k, kv_cache, slot_mapping):
    _infinicore.indexer_k_cache_(_t(k), _t(kv_cache), _t(slot_mapping))


def compute_block_sparse_mqa_logits_(
    logits,
    q,
    kv_cache,
    cu_seqlens_q,
    cu_seqlens_kv,
    block_table,
    weights,
    max_q_len,
    max_kv_len,
    max_context_len,
):
    _infinicore.compute_block_sparse_mqa_logits_(
        *map(
            _t,
            (
                logits,
                q,
                kv_cache,
                cu_seqlens_q,
                cu_seqlens_kv,
                block_table,
                weights,
            ),
        ),
        max_q_len,
        max_kv_len,
        max_context_len,
    )


def select_prefill_topk_block_indices_(
    topk_indices, logits, cu_seqlen_ks, cu_seqlen_ke
):
    _infinicore.select_prefill_topk_block_indices_(
        _t(topk_indices), _t(logits), _t(cu_seqlen_ks), _t(cu_seqlen_ke)
    )


def select_decode_topk_block_indices_(topk_indices, logits, seq_lens):
    _infinicore.select_decode_topk_block_indices_(
        _t(topk_indices), _t(logits), _t(seq_lens)
    )


def map_prefill_request_block_indices_(
    output,
    req_id,
    block_table,
    token_indices,
    block_size,
    has_prefill_workspace=False,
    prefill_workspace_request_ids=None,
    prefill_workspace_starts=None,
):
    _infinicore.map_prefill_request_block_indices_(
        _t(output),
        _t(req_id),
        _t(block_table),
        _t(token_indices),
        block_size,
        has_prefill_workspace,
        None
        if prefill_workspace_request_ids is None
        else _t(prefill_workspace_request_ids),
        None if prefill_workspace_starts is None else _t(prefill_workspace_starts),
    )


def map_decode_request_block_indices_(
    output, req_id, block_table, token_indices, block_size
):
    _infinicore.map_decode_request_block_indices_(
        _t(output), _t(req_id), _t(block_table), _t(token_indices), block_size
    )


def topk_indices_context_lens_(topk_lens, indices):
    _infinicore.topk_indices_context_lens_(_t(topk_lens), _t(indices))


def sparse_flash_mla_(
    output,
    query,
    kv_cache,
    indices,
    topk_lens,
    scale,
    attn_sink=None,
):
    _infinicore.sparse_flash_mla_(
        _t(output),
        _t(query),
        _t(kv_cache),
        _t(indices),
        _t(topk_lens),
        scale,
        None if attn_sink is None else _t(attn_sink),
    )
