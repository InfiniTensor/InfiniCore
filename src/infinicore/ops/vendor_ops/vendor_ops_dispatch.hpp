#pragma once

#include "infinicore/ops/common/dispatcher.hpp"
#include "infinicore/tensor.hpp"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

namespace infinicore::op::vendor_ops {

using AddRmsNormInplaceFn = void (*)(Tensor, Tensor, const Tensor &, float);
using ConcatAndCacheMlaFn = void (*)(const Tensor &, const Tensor &, Tensor, const Tensor &, const std::string &, Tensor);
using ConcatAndCacheMlaInt8Fn = void (*)(const Tensor &, const Tensor &, const Tensor &, const Tensor &, Tensor, Tensor, const Tensor &);
using ConcatMlaQFn = void (*)(const Tensor &, const Tensor &, Tensor);
using FusedIndexerPostprocessFn = void (*)(Tensor, Tensor, Tensor, Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, bool, double, double);
using IndexerKCacheFn = void (*)(const Tensor &, Tensor, const Tensor &);
using IndexerKQuantAndCacheFn = void (*)(const Tensor &, Tensor, const Tensor &, int64_t, const std::string &);
using BlockSparseMqaLogitsFn = void (*)(Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, int64_t, int64_t, int64_t);
using SelectPrefillTopkFn = void (*)(Tensor, const Tensor &, const Tensor &, const Tensor &);
using SelectDecodeTopkFn = void (*)(Tensor, const Tensor &, const Tensor &);
using MapPrefillIndicesFn = void (*)(Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t, bool, std::optional<Tensor>, std::optional<Tensor>);
using MapDecodeIndicesFn = void (*)(Tensor, const Tensor &, const Tensor &, const Tensor &, int64_t);
using TopkContextLensFn = void (*)(Tensor, const Tensor &);
using SparseFlashMlaFn = void (*)(Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, float, std::optional<Tensor>);
using FusedRotaryEmbeddingFn = void (*)(Tensor, Tensor, const Tensor &, int64_t, const Tensor &, bool);
using GroupedTopkFn = void (*)(Tensor, Tensor, const Tensor &, int64_t, int64_t, bool, const Tensor &, const std::string &);
using MoeArgsortFn = void (*)(Tensor, Tensor, Tensor, const Tensor &, int64_t);
using MoeExpandInputFn = void (*)(Tensor, std::optional<Tensor>, const Tensor &, const Tensor &, int64_t, int64_t, int64_t);
using MoeSiluAndMulQuantFn = void (*)(Tensor, std::optional<Tensor>, const Tensor &, int64_t);
using MoeSumFn = void (*)(Tensor, const Tensor &, std::optional<Tensor>, std::optional<Tensor>, double, double);
using MoeTopkFn = void (*)(Tensor, Tensor, Tensor, const Tensor &, bool, const Tensor &);
using PagedAttentionMlaFn = void (*)(Tensor, const Tensor &, const Tensor &, float, const Tensor &, const Tensor &, int64_t);
using DynamicScaledInt8QuantFn = void (*)(Tensor, const Tensor &, Tensor);
using ScaledMmFn = void (*)(Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, std::optional<Tensor>, bool);
using GroupGemmFn = void (*)(Tensor, const Tensor &, const Tensor &, const Tensor &, const Tensor &, const Tensor &, std::optional<Tensor>, std::optional<Tensor>, bool, bool);
using GroupGemmF16Fn = void (*)(Tensor, const Tensor &, const Tensor &, const Tensor &, std::optional<Tensor>, std::optional<Tensor>, bool, bool);

common::OpDispatcher<AddRmsNormInplaceFn> &add_rms_norm_inplace_dispatcher();
common::OpDispatcher<ConcatAndCacheMlaFn> &concat_and_cache_mla_dispatcher();
common::OpDispatcher<ConcatAndCacheMlaInt8Fn> &concat_and_cache_mla_int8_dispatcher();
common::OpDispatcher<ConcatMlaQFn> &concat_mla_q_dispatcher();
common::OpDispatcher<FusedIndexerPostprocessFn> &fused_indexer_postprocess_dispatcher();
common::OpDispatcher<IndexerKCacheFn> &indexer_k_cache_dispatcher();
common::OpDispatcher<IndexerKQuantAndCacheFn> &indexer_k_quant_and_cache_dispatcher();
common::OpDispatcher<BlockSparseMqaLogitsFn> &block_sparse_mqa_logits_dispatcher();
common::OpDispatcher<SelectPrefillTopkFn> &select_prefill_topk_dispatcher();
common::OpDispatcher<SelectDecodeTopkFn> &select_decode_topk_dispatcher();
common::OpDispatcher<MapPrefillIndicesFn> &map_prefill_indices_dispatcher();
common::OpDispatcher<MapDecodeIndicesFn> &map_decode_indices_dispatcher();
common::OpDispatcher<TopkContextLensFn> &topk_context_lens_dispatcher();
common::OpDispatcher<SparseFlashMlaFn> &sparse_flash_mla_dispatcher();
common::OpDispatcher<FusedRotaryEmbeddingFn> &fused_rotary_embedding_dispatcher();
common::OpDispatcher<GroupedTopkFn> &grouped_topk_dispatcher();
common::OpDispatcher<MoeArgsortFn> &moe_argsort_dispatcher();
common::OpDispatcher<MoeExpandInputFn> &moe_expand_input_dispatcher();
common::OpDispatcher<MoeSiluAndMulQuantFn> &moe_silu_and_mul_quant_dispatcher();
common::OpDispatcher<MoeSumFn> &moe_sum_dispatcher();
common::OpDispatcher<MoeTopkFn> &moe_topk_softmax_dispatcher();
common::OpDispatcher<MoeTopkFn> &moe_topk_sigmoid_dispatcher();
common::OpDispatcher<PagedAttentionMlaFn> &paged_attention_mla_dispatcher();
common::OpDispatcher<DynamicScaledInt8QuantFn> &dynamic_scaled_int8_quant_dispatcher();
common::OpDispatcher<ScaledMmFn> &scaled_mm_w4a8_dispatcher();
common::OpDispatcher<ScaledMmFn> &scaled_mm_w8a8_dispatcher();
common::OpDispatcher<GroupGemmF16Fn> &w16a16_group_gemm_dispatcher();
common::OpDispatcher<GroupGemmFn> &w4a8_group_gemm_dispatcher();
common::OpDispatcher<GroupGemmFn> &w8a8_group_gemm_dispatcher();

template <typename Fn>
Fn lookup(common::OpDispatcher<Fn> &dispatcher, Device::Type device_type, const char *op_name) {
    auto fn = dispatcher.lookup(device_type);
    if (fn == nullptr) {
        throw std::runtime_error(std::string(op_name) + " has no registered vendor implementation for this device");
    }
    return fn;
}

} // namespace infinicore::op::vendor_ops
