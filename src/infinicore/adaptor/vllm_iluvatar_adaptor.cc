#ifdef ENABLE_ATEN
#include "infinicore/adaptor/vllm_iluvatar_adaptor.hpp"
#include "infinicore/adaptor/aten_adaptor.hpp"

#if defined(ENABLE_ILUVATAR_API)
#define INFINICORE_VLLM_ILUVATAR_STREAM_GUARD() \
    infinicore::adaptor::set_aten_stream_to_infinicore()
#else
#define INFINICORE_VLLM_ILUVATAR_STREAM_GUARD() ((void)0)
#endif

#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>

namespace infinicore::adaptor::vllm_iluvatar {
namespace {

using fused_add_rms_norm_fn = void (*)(at::Tensor &, at::Tensor &, at::Tensor &, float);
using dynamic_scaled_int8_quant_fn = void (*)(at::Tensor &, at::Tensor &, const at::Tensor &);
using concat_mla_q_fn = void (*)(at::Tensor &, at::Tensor &, at::Tensor &);
using concat_and_cache_mla_fn = void (*)(at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, const std::string &, at::Tensor &);
using concat_and_cache_mla_int8_fn = void (*)(at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &);
using topk_softmax_fn = void (*)(at::Tensor &, at::Tensor &, at::Tensor &, const at::Tensor &, bool, std::optional<at::Tensor>);
using topk_sigmoid_fn = void (*)(at::Tensor &, at::Tensor &, at::Tensor &, const at::Tensor &, bool, std::optional<at::Tensor>);
using grouped_topk_fn = void (*)(at::Tensor &, at::Tensor &, const at::Tensor &, const std::optional<at::Tensor> &, int64_t, int64_t, std::string, bool);
using scaled_mm_w4a8_fn = void (*)(at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const std::optional<at::Tensor> &, bool);
using scaled_mm_w8a8_fn = scaled_mm_w4a8_fn;
using w4a8_group_gemm_fn = void (*)(at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const std::optional<at::Tensor> &, const std::optional<at::Tensor> &, bool, bool);
using w8a8_group_gemm_fn = void (*)(at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const std::optional<at::Tensor> &, const std::optional<at::Tensor> &, bool, bool);
using argsort_bincount_with_inv_pos_fn = void (*)(const at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, int64_t);
using expand_moe_input_with_inv_pos_fn = void (*)(at::Tensor &, std::optional<at::Tensor>, const at::Tensor &, const at::Tensor &, int64_t, int64_t, int64_t);
using silu_and_mul_quant_fn = void (*)(at::Tensor &, std::optional<at::Tensor>, const at::Tensor &, int64_t);
using moe_sum_vllm_fn = void (*)(at::Tensor &, const at::Tensor &, std::optional<at::Tensor>, std::optional<at::Tensor>, double, double);

struct Symbols {
    void *handle = nullptr;
    fused_add_rms_norm_fn fused_add_rms_norm = nullptr;
    dynamic_scaled_int8_quant_fn dynamic_scaled_int8_quant = nullptr;
    concat_mla_q_fn concat_mla_q = nullptr;
    concat_and_cache_mla_fn concat_and_cache_mla = nullptr;
    concat_and_cache_mla_int8_fn concat_and_cache_mla_int8 = nullptr;
    topk_softmax_fn topk_softmax = nullptr;
    topk_sigmoid_fn topk_sigmoid = nullptr;
    grouped_topk_fn grouped_topk = nullptr;
    scaled_mm_w4a8_fn scaled_mm_w4a8 = nullptr;
    scaled_mm_w8a8_fn scaled_mm_w8a8 = nullptr;
    w4a8_group_gemm_fn w4a8_group_gemm = nullptr;
    w8a8_group_gemm_fn w8a8_group_gemm = nullptr;
    argsort_bincount_with_inv_pos_fn argsort_bincount_with_inv_pos = nullptr;
    expand_moe_input_with_inv_pos_fn expand_moe_input_with_inv_pos = nullptr;
    silu_and_mul_quant_fn silu_and_mul_quant = nullptr;
    moe_sum_vllm_fn moe_sum_vllm = nullptr;
    std::string error;
};

Symbols &symbols() {
    static Symbols syms;
    static std::once_flag once;
    std::call_once(once, []() {
#if defined(ENABLE_ILUVATAR_API)
        const char *env_path = std::getenv("INFINICORE_VLLM_ILUVATAR_SO");
        const char *paths[] = {
            env_path,
            "/usr/local/lib/python3.12/site-packages/vllm_iluvatar/_C.cpython-312-x86_64-linux-gnu.so",
            "/usr/local/lib/python3.10/site-packages/vllm_iluvatar/_C.cpython-310-x86_64-linux-gnu.so",
        };

        for (const char *path : paths) {
            if (path == nullptr || path[0] == '\0') {
                continue;
            }
            void *handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
            if (!handle) {
                syms.error = dlerror();
                continue;
            }
            auto fused_fn = reinterpret_cast<fused_add_rms_norm_fn>(
                dlsym(handle, "_ZN7pyinfer4perf18fused_add_rms_normERN2at6TensorES3_S3_f"));
            auto quant_fn = reinterpret_cast<dynamic_scaled_int8_quant_fn>(
                dlsym(handle, "_ZN7pyinfer4perf25dynamic_scaled_int8_quantERN2at6TensorES3_RKS2_"));
            auto concat_mla_q_fn_ptr = reinterpret_cast<concat_mla_q_fn>(
                dlsym(handle, "_ZN7pyinfer4perf12concat_mla_qERN2at6TensorES3_S3_"));
            auto concat_and_cache_mla_fn_ptr = reinterpret_cast<concat_and_cache_mla_fn>(
                dlsym(handle, "_ZN7pyinfer4perf20concat_and_cache_mlaERN2at6TensorES3_S3_S3_RKSsS3_"));
            auto concat_and_cache_mla_int8_fn_ptr = reinterpret_cast<concat_and_cache_mla_int8_fn>(
                dlsym(handle, "_ZN7pyinfer4perf25concat_and_cache_mla_int8ERN2at6TensorES3_S3_S3_S3_S3_S3_"));
            auto topk_softmax_fn_ptr = reinterpret_cast<topk_softmax_fn>(
                dlsym(handle, "_ZN7pyinfer4perf12topk_softmaxERN2at6TensorES3_S3_RKS2_bSt8optionalIS2_E"));
            auto topk_sigmoid_fn_ptr = reinterpret_cast<topk_sigmoid_fn>(
                dlsym(handle, "_ZN7pyinfer4perf12topk_sigmoidERN2at6TensorES3_S3_RKS2_bSt8optionalIS2_E"));
            auto grouped_topk_fn_ptr = reinterpret_cast<grouped_topk_fn>(
                dlsym(handle, "_ZN7pyinfer4perf16moe_grouped_topkERN2at6TensorES3_RKS2_RKSt8optionalIS2_EllSsb"));
            auto scaled_mm_w4a8_fn_ptr = reinterpret_cast<scaled_mm_w4a8_fn>(
                dlsym(handle, "_ZN7pyinfer7cuinfer14scaled_mm_w4a8ERN2at6TensorERKS2_S5_S5_S5_RKSt8optionalIS2_Eb"));
            auto scaled_mm_w8a8_fn_ptr = reinterpret_cast<scaled_mm_w8a8_fn>(
                dlsym(handle, "_ZN7pyinfer7cuinfer9scaled_mmERN2at6TensorERKS2_S5_S5_S5_RKSt8optionalIS2_Eb"));
            auto w4a8_group_gemm_fn_ptr = reinterpret_cast<w4a8_group_gemm_fn>(
                dlsym(handle, "_ZN7pyinfer7cuinfer15w4a8_group_gemmERN2at6TensorERKS2_S5_S5_S5_S5_RKSt8optionalIS2_ES9_bb"));
            auto w8a8_group_gemm_fn_ptr = reinterpret_cast<w8a8_group_gemm_fn>(
                dlsym(handle, "_ZN7pyinfer7cuinfer15w8a8_group_gemmERN2at6TensorERKS2_S5_S5_S5_S5_RKSt8optionalIS2_ES9_bb"));
            auto argsort_bincount_with_inv_pos_fn_ptr = reinterpret_cast<argsort_bincount_with_inv_pos_fn>(
                dlsym(handle, "_ZN7pyinfer4perf29argsort_bincount_with_inv_posERKN2at6TensorERS2_S5_S5_l"));
            auto expand_moe_input_with_inv_pos_fn_ptr = reinterpret_cast<expand_moe_input_with_inv_pos_fn>(
                dlsym(handle, "_ZN7pyinfer4perf29expand_moe_input_with_inv_posERN2at6TensorESt8optionalIS2_ERKS2_S7_lll"));
            auto silu_and_mul_quant_fn_ptr = reinterpret_cast<silu_and_mul_quant_fn>(
                dlsym(handle, "_ZN7pyinfer4perf18silu_and_mul_quantERN2at6TensorESt8optionalIS2_ERKS2_l"));
            auto moe_sum_vllm_fn_ptr = reinterpret_cast<moe_sum_vllm_fn>(
                dlsym(handle, "_ZN7pyinfer4perf7moe_sumERN2at6TensorERKS2_St8optionalIS2_ES7_dd"));
            if (!fused_fn && !quant_fn && !concat_mla_q_fn_ptr && !concat_and_cache_mla_fn_ptr && !concat_and_cache_mla_int8_fn_ptr && !topk_softmax_fn_ptr && !topk_sigmoid_fn_ptr && !grouped_topk_fn_ptr && !scaled_mm_w4a8_fn_ptr && !scaled_mm_w8a8_fn_ptr && !w4a8_group_gemm_fn_ptr && !w8a8_group_gemm_fn_ptr && !argsort_bincount_with_inv_pos_fn_ptr && !expand_moe_input_with_inv_pos_fn_ptr && !silu_and_mul_quant_fn_ptr && !moe_sum_vllm_fn_ptr) {
                syms.error = dlerror();
                dlclose(handle);
                continue;
            }
            syms.handle = handle;
            syms.fused_add_rms_norm = fused_fn;
            syms.dynamic_scaled_int8_quant = quant_fn;
            syms.concat_mla_q = concat_mla_q_fn_ptr;
            syms.concat_and_cache_mla = concat_and_cache_mla_fn_ptr;
            syms.concat_and_cache_mla_int8 = concat_and_cache_mla_int8_fn_ptr;
            syms.topk_softmax = topk_softmax_fn_ptr;
            syms.topk_sigmoid = topk_sigmoid_fn_ptr;
            syms.grouped_topk = grouped_topk_fn_ptr;
            syms.scaled_mm_w4a8 = scaled_mm_w4a8_fn_ptr;
            syms.scaled_mm_w8a8 = scaled_mm_w8a8_fn_ptr;
            syms.w4a8_group_gemm = w4a8_group_gemm_fn_ptr;
            syms.w8a8_group_gemm = w8a8_group_gemm_fn_ptr;
            syms.argsort_bincount_with_inv_pos = argsort_bincount_with_inv_pos_fn_ptr;
            syms.expand_moe_input_with_inv_pos = expand_moe_input_with_inv_pos_fn_ptr;
            syms.silu_and_mul_quant = silu_and_mul_quant_fn_ptr;
            syms.moe_sum_vllm = moe_sum_vllm_fn_ptr;
            syms.error.clear();
            return;
        }
        if (syms.error.empty()) {
            syms.error = "vllm_iluvatar extension not found";
        }
#else
            syms.error = "InfiniCore was not built with ENABLE_ILUVATAR_API";
#endif
    });
    return syms;
}

} // namespace

bool available() {
    return symbols().fused_add_rms_norm != nullptr;
}

bool dynamic_scaled_int8_quant_available() {
    return symbols().dynamic_scaled_int8_quant != nullptr;
}

bool concat_mla_q_available() {
    return symbols().concat_mla_q != nullptr;
}

bool concat_and_cache_mla_available() {
    return symbols().concat_and_cache_mla != nullptr;
}

bool concat_and_cache_mla_int8_available() {
    return symbols().concat_and_cache_mla_int8 != nullptr;
}

bool topk_softmax_available() {
    return symbols().topk_softmax != nullptr;
}

bool topk_sigmoid_available() {
    return symbols().topk_sigmoid != nullptr;
}

bool grouped_topk_available() {
    return symbols().grouped_topk != nullptr;
}

bool scaled_mm_w4a8_available() {
    return symbols().scaled_mm_w4a8 != nullptr;
}

bool scaled_mm_w8a8_available() {
    return symbols().scaled_mm_w8a8 != nullptr;
}

bool w4a8_group_gemm_available() {
    return symbols().w4a8_group_gemm != nullptr;
}

bool w8a8_group_gemm_available() {
    return symbols().w8a8_group_gemm != nullptr;
}

bool argsort_bincount_with_inv_pos_available() {
    return symbols().argsort_bincount_with_inv_pos != nullptr;
}

bool expand_moe_input_with_inv_pos_available() {
    return symbols().expand_moe_input_with_inv_pos != nullptr;
}

bool silu_and_mul_quant_available() {
    return symbols().silu_and_mul_quant != nullptr;
}

bool moe_sum_vllm_available() {
    return symbols().moe_sum_vllm != nullptr;
}

void fused_add_rms_norm(at::Tensor &input, at::Tensor &residual, at::Tensor &weight, float epsilon) {
    INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
    auto &syms = symbols();
    if (!syms.fused_add_rms_norm) {
        throw std::runtime_error("vllm_iluvatar fused_add_rms_norm unavailable: " + syms.error);
    }
    syms.fused_add_rms_norm(input, residual, weight, epsilon);
}

void dynamic_scaled_int8_quant(at::Tensor &output, at::Tensor &input_scales, const at::Tensor &input) {
    INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
    auto &syms = symbols();
    if (!syms.dynamic_scaled_int8_quant) {
        throw std::runtime_error("vllm_iluvatar dynamic_scaled_int8_quant unavailable: " + syms.error);
    }
    syms.dynamic_scaled_int8_quant(output, input_scales, input);
}

void concat_mla_q(at::Tensor &ql_nope, at::Tensor &q_pe, at::Tensor &q_out) {
    INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
    auto &syms = symbols();
    if (!syms.concat_mla_q) {
        throw std::runtime_error("vllm_iluvatar concat_mla_q unavailable: " + syms.error);
    }
    syms.concat_mla_q(ql_nope, q_pe, q_out);
}

void concat_and_cache_mla(at::Tensor &kv_c, at::Tensor &k_pe, at::Tensor &kv_cache, at::Tensor &slot_mapping, const std::string &kv_cache_dtype, at::Tensor &scale) {
    INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
    auto &syms = symbols();
    if (!syms.concat_and_cache_mla) {
        throw std::runtime_error("vllm_iluvatar concat_and_cache_mla unavailable: " + syms.error);
    }
    syms.concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale);
}

void concat_and_cache_mla_int8(at::Tensor &kv_c_int8, at::Tensor &kv_c_scale, at::Tensor &k_pe_int8, at::Tensor &k_pe_scale, at::Tensor &kv_cache, at::Tensor &kv_cache_scale, at::Tensor &slot_mapping) {
    INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
    auto &syms = symbols();
    if (!syms.concat_and_cache_mla_int8) {
        throw std::runtime_error("vllm_iluvatar concat_and_cache_mla_int8 unavailable: " + syms.error);
    }
    syms.concat_and_cache_mla_int8(kv_c_int8, kv_c_scale, k_pe_int8, k_pe_scale, kv_cache, kv_cache_scale, slot_mapping);
}

void topk_softmax(at::Tensor &topk_weights, at::Tensor &topk_ids, at::Tensor &token_expert_indices, const at::Tensor &gating_output, bool renormalize, std::optional<at::Tensor> correction_bias) {
    INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
    auto &syms = symbols();
    if (!syms.topk_softmax) {
        throw std::runtime_error("vllm_iluvatar topk_softmax unavailable: " + syms.error);
    }
    syms.topk_softmax(topk_weights, topk_ids, token_expert_indices, gating_output, renormalize, correction_bias);
}

void topk_sigmoid(at::Tensor &topk_weights, at::Tensor &topk_ids, at::Tensor &token_expert_indices, const at::Tensor &gating_output, bool renormalize, std::optional<at::Tensor> correction_bias) {
    INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
    auto &syms = symbols();
    if (!syms.topk_sigmoid) {
        throw std::runtime_error("vllm_iluvatar topk_sigmoid unavailable: " + syms.error);
    }
    syms.topk_sigmoid(topk_weights, topk_ids, token_expert_indices, gating_output, renormalize, correction_bias);
}

void grouped_topk(at::Tensor &topk_weights, at::Tensor &topk_ids, const at::Tensor &scores, std::optional<at::Tensor> bias, int64_t num_expert_group, int64_t topk_group, const std::string &scoring_func, bool renormalize) {
    INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
    auto &syms = symbols();
    if (!syms.grouped_topk) {
        throw std::runtime_error("vllm_iluvatar grouped_topk unavailable: " + syms.error);
    }
    syms.grouped_topk(topk_weights, topk_ids, scores, bias, num_expert_group, topk_group, scoring_func, renormalize);
}

void scaled_mm_w4a8(at::Tensor &out, const at::Tensor &a, const at::Tensor &b, const at::Tensor &a_scales, const at::Tensor &b_scales, std::optional<at::Tensor> bias, bool trans_weight) {
    INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
    auto &syms = symbols();
    if (!syms.scaled_mm_w4a8) {
        throw std::runtime_error("vllm_iluvatar scaled_mm_w4a8 unavailable: " + syms.error);
    }
    syms.scaled_mm_w4a8(out, a, b, a_scales, b_scales, bias, trans_weight);
}

void scaled_mm_w8a8(at::Tensor &out, const at::Tensor &a, const at::Tensor &b, const at::Tensor &a_scales, const at::Tensor &b_scales, std::optional<at::Tensor> bias, bool trans_weight) {
    INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
    auto &syms = symbols();
    if (!syms.scaled_mm_w8a8) {
        throw std::runtime_error("vllm_iluvatar scaled_mm W8A8 unavailable: " + syms.error);
    }
    syms.scaled_mm_w8a8(out, a, b, a_scales, b_scales, bias, trans_weight);
}

void w4a8_group_gemm(at::Tensor &out, const at::Tensor &input, const at::Tensor &weight, const at::Tensor &input_scale, const at::Tensor &weight_scale, const at::Tensor &tokens_per_experts, std::optional<at::Tensor> sorted_token_ids, std::optional<at::Tensor> bias, bool trans_weight, bool is_decode) {
    INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
    auto &syms = symbols();
    if (!syms.w4a8_group_gemm) {
        throw std::runtime_error("vllm_iluvatar w4a8_group_gemm unavailable: " + syms.error);
    }
    syms.w4a8_group_gemm(out, input, weight, input_scale, weight_scale, tokens_per_experts, sorted_token_ids, bias, trans_weight, is_decode);
}

void w8a8_group_gemm(at::Tensor &out, const at::Tensor &input, const at::Tensor &weight, const at::Tensor &input_scale, const at::Tensor &weight_scale, const at::Tensor &tokens_per_experts, std::optional<at::Tensor> sorted_token_ids, std::optional<at::Tensor> bias, bool trans_weight, bool is_decode) {
    INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
    auto &syms = symbols();
    if (!syms.w8a8_group_gemm) {
        throw std::runtime_error("vllm_iluvatar w8a8_group_gemm unavailable: " + syms.error);
    }
    syms.w8a8_group_gemm(out, input, weight, input_scale, weight_scale, tokens_per_experts, sorted_token_ids, bias, trans_weight, is_decode);
}

void argsort_bincount_with_inv_pos(const at::Tensor &topk_ids, at::Tensor &tokens_per_experts, at::Tensor &sorted_indices, at::Tensor &inv_pos, int64_t num_experts) {
    INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
    auto &syms = symbols();
    if (!syms.argsort_bincount_with_inv_pos) {
        throw std::runtime_error("vllm_iluvatar argsort_bincount_with_inv_pos unavailable: " + syms.error);
    }
    syms.argsort_bincount_with_inv_pos(topk_ids, tokens_per_experts, sorted_indices, inv_pos, num_experts);
}

void expand_moe_input_with_inv_pos(at::Tensor &expand_states, std::optional<at::Tensor> expand_scales, const at::Tensor &hidden_states, const at::Tensor &inv_pos, int64_t top_k, int64_t group_size, int64_t format) {
    INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
    auto &syms = symbols();
    if (!syms.expand_moe_input_with_inv_pos) {
        throw std::runtime_error("vllm_iluvatar expand_moe_input_with_inv_pos unavailable: " + syms.error);
    }
    syms.expand_moe_input_with_inv_pos(expand_states, expand_scales, hidden_states, inv_pos, top_k, group_size, format);
}

void silu_and_mul_quant(at::Tensor &output, std::optional<at::Tensor> output_scale, const at::Tensor &input, int64_t format) {
    INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
    auto &syms = symbols();
    if (!syms.silu_and_mul_quant) {
        throw std::runtime_error("vllm_iluvatar silu_and_mul_quant unavailable: " + syms.error);
    }
    syms.silu_and_mul_quant(output, output_scale, input, format);
}

void moe_sum_vllm(at::Tensor &output, const at::Tensor &input, std::optional<at::Tensor> topk_weights, std::optional<at::Tensor> extra_residual, double routed_scale, double residual_scale) {
    INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
    auto &syms = symbols();
    if (!syms.moe_sum_vllm) {
        throw std::runtime_error("vllm_iluvatar moe_sum unavailable: " + syms.error);
    }
    syms.moe_sum_vllm(output, input, topk_weights, extra_residual, routed_scale, residual_scale);
}

#undef INFINICORE_VLLM_ILUVATAR_STREAM_GUARD

} // namespace infinicore::adaptor::vllm_iluvatar
#endif // ENABLE_ATEN
