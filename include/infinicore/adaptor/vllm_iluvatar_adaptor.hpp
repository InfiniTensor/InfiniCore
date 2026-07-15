#pragma once

#ifdef ENABLE_ATEN
#include <ATen/ATen.h>
#include <optional>
#include <string>

namespace infinicore::adaptor::vllm_iluvatar {

bool available();
bool dynamic_scaled_int8_quant_available();
bool concat_mla_q_available();
bool concat_and_cache_mla_available();
bool concat_and_cache_mla_int8_available();
bool paged_attention_mla_available();
bool topk_softmax_available();
bool topk_sigmoid_available();
bool grouped_topk_available();
bool scaled_mm_w4a8_available();
bool scaled_mm_w8a8_available();
bool w4a8_group_gemm_available();
bool w8a8_group_gemm_available();
bool w16a16_group_gemm_available();
bool argsort_bincount_with_inv_pos_available();
bool expand_moe_input_with_inv_pos_available();
bool silu_and_mul_quant_available();
bool moe_sum_vllm_available();
void fused_add_rms_norm(at::Tensor &input, at::Tensor &residual, at::Tensor &weight, float epsilon);
void dynamic_scaled_int8_quant(at::Tensor &output, at::Tensor &input_scales, const at::Tensor &input);
void concat_mla_q(at::Tensor &ql_nope, at::Tensor &q_pe, at::Tensor &q_out);
void concat_and_cache_mla(at::Tensor &kv_c, at::Tensor &k_pe, at::Tensor &kv_cache, at::Tensor &slot_mapping, const std::string &kv_cache_dtype, at::Tensor &scale);
void concat_and_cache_mla_int8(at::Tensor &kv_c_int8, at::Tensor &kv_c_scale, at::Tensor &k_pe_int8, at::Tensor &k_pe_scale, at::Tensor &kv_cache, at::Tensor &kv_cache_scale, at::Tensor &slot_mapping);
void paged_attention_mla(at::Tensor &output,
                         at::Tensor &query,
                         at::Tensor &kv_cache,
                         double scale,
                         at::Tensor &block_tables,
                         at::Tensor &context_lens,
                         int64_t max_context_len,
                         bool use_cuda_graph,
                         at::Tensor &softmax_lse);
void topk_softmax(at::Tensor &topk_weights, at::Tensor &topk_ids, at::Tensor &token_expert_indices, const at::Tensor &gating_output, bool renormalize, std::optional<at::Tensor> correction_bias);
void topk_sigmoid(at::Tensor &topk_weights, at::Tensor &topk_ids, at::Tensor &token_expert_indices, const at::Tensor &gating_output, bool renormalize, std::optional<at::Tensor> correction_bias);
void grouped_topk(at::Tensor &topk_weights, at::Tensor &topk_ids, const at::Tensor &scores, std::optional<at::Tensor> bias, int64_t num_expert_group, int64_t topk_group, const std::string &scoring_func, bool renormalize);
void scaled_mm_w4a8(at::Tensor &out, const at::Tensor &a, const at::Tensor &b, const at::Tensor &a_scales, const at::Tensor &b_scales, std::optional<at::Tensor> bias, bool trans_weight);
void scaled_mm_w8a8(at::Tensor &out, const at::Tensor &a, const at::Tensor &b, const at::Tensor &a_scales, const at::Tensor &b_scales, std::optional<at::Tensor> bias, bool trans_weight);
void w4a8_group_gemm(at::Tensor &out, const at::Tensor &input, const at::Tensor &weight, const at::Tensor &input_scale, const at::Tensor &weight_scale, const at::Tensor &tokens_per_experts, std::optional<at::Tensor> sorted_token_ids, std::optional<at::Tensor> bias, bool trans_weight, bool is_decode);
void w8a8_group_gemm(at::Tensor &out, const at::Tensor &input, const at::Tensor &weight, const at::Tensor &input_scale, const at::Tensor &weight_scale, const at::Tensor &tokens_per_experts, std::optional<at::Tensor> sorted_token_ids, std::optional<at::Tensor> bias, bool trans_weight, bool is_decode);
void w16a16_group_gemm(at::Tensor &out, const at::Tensor &input, const at::Tensor &weight, const at::Tensor &tokens_per_experts, std::optional<at::Tensor> sorted_token_ids, std::optional<at::Tensor> bias, bool trans_weight, bool is_decode);
void argsort_bincount_with_inv_pos(const at::Tensor &topk_ids, at::Tensor &tokens_per_experts, at::Tensor &sorted_indices, at::Tensor &inv_pos, int64_t num_experts);
void expand_moe_input_with_inv_pos(at::Tensor &expand_states, std::optional<at::Tensor> expand_scales, const at::Tensor &hidden_states, const at::Tensor &inv_pos, int64_t top_k, int64_t group_size, int64_t format);
void silu_and_mul_quant(at::Tensor &output, std::optional<at::Tensor> output_scale, const at::Tensor &input, int64_t format);
void moe_sum_vllm(at::Tensor &output, const at::Tensor &input, std::optional<at::Tensor> topk_weights, std::optional<at::Tensor> extra_residual, double routed_scale, double residual_scale);

} // namespace infinicore::adaptor::vllm_iluvatar
#endif // ENABLE_ATEN
