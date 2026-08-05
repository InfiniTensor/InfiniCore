#include "vendor_ops_dispatch.hpp"

namespace infinicore::op::vendor_ops {

#define INFINICORE_VENDOR_DISPATCHER(NAME, TYPE)      \
    common::OpDispatcher<TYPE> &NAME##_dispatcher() { \
        static common::OpDispatcher<TYPE> dispatcher; \
        return dispatcher;                            \
    }

INFINICORE_VENDOR_DISPATCHER(add_rms_norm_inplace, AddRmsNormInplaceFn)
INFINICORE_VENDOR_DISPATCHER(concat_and_cache_mla, ConcatAndCacheMlaFn)
INFINICORE_VENDOR_DISPATCHER(concat_and_cache_mla_int8, ConcatAndCacheMlaInt8Fn)
INFINICORE_VENDOR_DISPATCHER(concat_mla_q, ConcatMlaQFn)
INFINICORE_VENDOR_DISPATCHER(fused_indexer_postprocess, FusedIndexerPostprocessFn)
INFINICORE_VENDOR_DISPATCHER(indexer_k_cache, IndexerKCacheFn)
INFINICORE_VENDOR_DISPATCHER(indexer_k_quant_and_cache, IndexerKQuantAndCacheFn)
INFINICORE_VENDOR_DISPATCHER(block_sparse_mqa_logits, BlockSparseMqaLogitsFn)
INFINICORE_VENDOR_DISPATCHER(select_prefill_topk, SelectPrefillTopkFn)
INFINICORE_VENDOR_DISPATCHER(select_decode_topk, SelectDecodeTopkFn)
INFINICORE_VENDOR_DISPATCHER(map_prefill_indices, MapPrefillIndicesFn)
INFINICORE_VENDOR_DISPATCHER(map_decode_indices, MapDecodeIndicesFn)
INFINICORE_VENDOR_DISPATCHER(topk_context_lens, TopkContextLensFn)
INFINICORE_VENDOR_DISPATCHER(sparse_flash_mla, SparseFlashMlaFn)
INFINICORE_VENDOR_DISPATCHER(fused_rotary_embedding, FusedRotaryEmbeddingFn)
INFINICORE_VENDOR_DISPATCHER(grouped_topk, GroupedTopkFn)
INFINICORE_VENDOR_DISPATCHER(moe_argsort, MoeArgsortFn)
INFINICORE_VENDOR_DISPATCHER(moe_expand_input, MoeExpandInputFn)
INFINICORE_VENDOR_DISPATCHER(moe_silu_and_mul_quant, MoeSiluAndMulQuantFn)
INFINICORE_VENDOR_DISPATCHER(moe_sum, MoeSumFn)
INFINICORE_VENDOR_DISPATCHER(moe_topk_softmax, MoeTopkFn)
INFINICORE_VENDOR_DISPATCHER(moe_topk_sigmoid, MoeTopkFn)
INFINICORE_VENDOR_DISPATCHER(paged_attention_mla, PagedAttentionMlaFn)
INFINICORE_VENDOR_DISPATCHER(dynamic_scaled_int8_quant, DynamicScaledInt8QuantFn)
INFINICORE_VENDOR_DISPATCHER(scaled_mm_w4a8, ScaledMmFn)
INFINICORE_VENDOR_DISPATCHER(scaled_mm_w8a8, ScaledMmFn)
INFINICORE_VENDOR_DISPATCHER(w16a16_group_gemm, GroupGemmF16Fn)
INFINICORE_VENDOR_DISPATCHER(w4a8_group_gemm, GroupGemmFn)
INFINICORE_VENDOR_DISPATCHER(w8a8_group_gemm, GroupGemmFn)

#undef INFINICORE_VENDOR_DISPATCHER

} // namespace infinicore::op::vendor_ops
