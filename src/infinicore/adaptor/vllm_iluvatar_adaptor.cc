#ifdef ENABLE_ATEN
#include "infinicore/adaptor/vllm_iluvatar_adaptor.hpp"
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/graph/graph.hpp"

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
#include <type_traits>
#include <utility>

namespace infinicore::adaptor::vllm_iluvatar {
namespace {

using fused_add_rms_norm_fn = void (*)(at::Tensor &, at::Tensor &, at::Tensor &, float);
using rotary_embedding_fn = void (*)(at::Tensor &, at::Tensor &, std::optional<at::Tensor>, int64_t, at::Tensor &, bool);
using dynamic_scaled_int8_quant_fn = void (*)(at::Tensor &, at::Tensor &, const at::Tensor &);
using concat_mla_q_fn = void (*)(at::Tensor &, at::Tensor &, at::Tensor &);
using concat_and_cache_mla_fn = void (*)(at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, const std::string &, at::Tensor &);
using concat_and_cache_mla_int8_fn = void (*)(at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &);
using paged_attention_mla_fn = at::Tensor (*)(at::Tensor &, at::Tensor &, at::Tensor &, double, at::Tensor &, at::Tensor &, int64_t, bool, at::Tensor &);
using topk_softmax_fn = void (*)(at::Tensor &, at::Tensor &, at::Tensor &, const at::Tensor &, bool, std::optional<at::Tensor>);
using topk_sigmoid_fn = void (*)(at::Tensor &, at::Tensor &, at::Tensor &, const at::Tensor &, bool, std::optional<at::Tensor>);
using grouped_topk_fn = void (*)(at::Tensor &, at::Tensor &, const at::Tensor &, const std::optional<at::Tensor> &, int64_t, int64_t, std::string, bool);
using scaled_mm_w4a8_fn = void (*)(at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const std::optional<at::Tensor> &, bool);
using scaled_mm_w8a8_fn = scaled_mm_w4a8_fn;
using w4a8_group_gemm_fn = void (*)(at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const std::optional<at::Tensor> &, const std::optional<at::Tensor> &, bool, bool);
using w8a8_group_gemm_fn = void (*)(at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const std::optional<at::Tensor> &, const std::optional<at::Tensor> &, bool, bool);
using w16a16_group_gemm_fn = void (*)(at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const std::optional<at::Tensor> &, const std::optional<at::Tensor> &, bool, bool);
using argsort_bincount_with_inv_pos_fn = void (*)(const at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, int64_t);
using expand_moe_input_with_inv_pos_fn = void (*)(at::Tensor &, std::optional<at::Tensor>, const at::Tensor &, const at::Tensor &, int64_t, int64_t, int64_t);
using silu_and_mul_quant_fn = void (*)(at::Tensor &, std::optional<at::Tensor>, const at::Tensor &, int64_t);

template <typename Fn>
class VllmIluvatarGraphOperator final : public graph::GraphOperator {
public:
    explicit VllmIluvatarGraphOperator(Fn fn) : fn_(std::move(fn)) {}

    void run() const override {
        fn_();
    }

private:
    mutable Fn fn_;
};

template <typename Fn>
void record_or_run(Fn &&fn) {
    using Op = VllmIluvatarGraphOperator<std::decay_t<Fn>>;
    auto op = std::make_shared<Op>(std::forward<Fn>(fn));
    if (context::isGraphRecording()) {
        context::addGraphOperator(op);
    } else {
        op->run();
    }
}
using moe_sum_vllm_fn = void (*)(at::Tensor &, const at::Tensor &, std::optional<at::Tensor>, std::optional<at::Tensor>, double, double);
using fused_indexer_postprocess_fn = void (*)(at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, bool, double, double);
using indexer_k_cache_fn = void (*)(const at::Tensor &, at::Tensor &, const at::Tensor &);
using indexer_k_quant_and_cache_fn = void (*)(at::Tensor &, at::Tensor &, at::Tensor &, int64_t, const std::string &);
using block_sparse_logits_fn = void (*)(const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, at::Tensor &, int64_t, int64_t, int64_t);
using select_prefill_topk_fn = void (*)(const at::Tensor &, const at::Tensor &, const at::Tensor &, at::Tensor &);
using select_decode_topk_fn = void (*)(const at::Tensor &, const at::Tensor &, at::Tensor &);
using map_prefill_result = std::tuple<at::Tensor, std::optional<at::Tensor>>;
using map_prefill_indices_fn = map_prefill_result (*)(const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, bool, std::optional<at::Tensor>, std::optional<at::Tensor>, bool, std::optional<at::Tensor>);
using map_decode_indices_fn = at::Tensor (*)(const at::Tensor &, const at::Tensor &, const at::Tensor &, int64_t, std::optional<at::Tensor>);
using sparse_flash_attn_fn = void (*)(at::Tensor &, const at::Tensor &, const at::Tensor &, const at::Tensor &, double);
using topk_indices_context_lens_fn = void (*)(at::Tensor &, const at::Tensor &);
using flash_mla_sparse_v2_fn = void (*)(at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, at::Tensor &, float, const std::optional<at::Tensor> &);

struct Symbols {
    void *handle = nullptr;
    fused_add_rms_norm_fn fused_add_rms_norm = nullptr;
    rotary_embedding_fn rotary_embedding = nullptr;
    dynamic_scaled_int8_quant_fn dynamic_scaled_int8_quant = nullptr;
    concat_mla_q_fn concat_mla_q = nullptr;
    concat_and_cache_mla_fn concat_and_cache_mla = nullptr;
    concat_and_cache_mla_int8_fn concat_and_cache_mla_int8 = nullptr;
    paged_attention_mla_fn paged_attention_mla = nullptr;
    topk_softmax_fn topk_softmax = nullptr;
    topk_sigmoid_fn topk_sigmoid = nullptr;
    grouped_topk_fn grouped_topk = nullptr;
    scaled_mm_w4a8_fn scaled_mm_w4a8 = nullptr;
    scaled_mm_w8a8_fn scaled_mm_w8a8 = nullptr;
    w4a8_group_gemm_fn w4a8_group_gemm = nullptr;
    w8a8_group_gemm_fn w8a8_group_gemm = nullptr;
    w16a16_group_gemm_fn w16a16_group_gemm = nullptr;
    argsort_bincount_with_inv_pos_fn argsort_bincount_with_inv_pos = nullptr;
    expand_moe_input_with_inv_pos_fn expand_moe_input_with_inv_pos = nullptr;
    silu_and_mul_quant_fn silu_and_mul_quant = nullptr;
    moe_sum_vllm_fn moe_sum_vllm = nullptr;
    fused_indexer_postprocess_fn fused_indexer_postprocess = nullptr;
    indexer_k_cache_fn indexer_k_cache = nullptr;
    indexer_k_quant_and_cache_fn indexer_k_quant_and_cache = nullptr;
    block_sparse_logits_fn block_sparse_logits = nullptr;
    select_prefill_topk_fn select_prefill_topk = nullptr;
    select_decode_topk_fn select_decode_topk = nullptr;
    map_prefill_indices_fn map_prefill_indices = nullptr;
    map_decode_indices_fn map_decode_indices = nullptr;
    sparse_flash_attn_fn sparse_flash_attn = nullptr;
    void *ix_handle = nullptr;
    topk_indices_context_lens_fn topk_indices_context_lens = nullptr;
    flash_mla_sparse_v2_fn flash_mla_sparse_v2 = nullptr;
    std::string error;
    std::string ix_error;
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
            auto rotary_embedding_fn_ptr = reinterpret_cast<rotary_embedding_fn>(
                dlsym(handle, "_ZN7pyinfer4perf16rotary_embeddingERN2at6TensorES3_St8optionalIS2_ElS3_b"));
            auto quant_fn = reinterpret_cast<dynamic_scaled_int8_quant_fn>(
                dlsym(handle, "_ZN7pyinfer4perf25dynamic_scaled_int8_quantERN2at6TensorES3_RKS2_"));
            auto concat_mla_q_fn_ptr = reinterpret_cast<concat_mla_q_fn>(
                dlsym(handle, "_ZN7pyinfer4perf12concat_mla_qERN2at6TensorES3_S3_"));
            auto concat_and_cache_mla_fn_ptr = reinterpret_cast<concat_and_cache_mla_fn>(
                dlsym(handle, "_ZN7pyinfer4perf20concat_and_cache_mlaERN2at6TensorES3_S3_S3_RKSsS3_"));
            auto concat_and_cache_mla_int8_fn_ptr = reinterpret_cast<concat_and_cache_mla_int8_fn>(
                dlsym(handle, "_ZN7pyinfer4perf25concat_and_cache_mla_int8ERN2at6TensorES3_S3_S3_S3_S3_S3_"));
            auto paged_attention_mla_fn_ptr = reinterpret_cast<paged_attention_mla_fn>(
                dlsym(handle, "_ZN7pyinfer7cuinfer24vllm_paged_attention_mlaERN2at6TensorES3_S3_dS3_S3_lbS3_"));
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
            auto w16a16_group_gemm_fn_ptr = reinterpret_cast<w16a16_group_gemm_fn>(
                dlsym(handle, "_ZN7pyinfer7cuinfer17w16a16_group_gemmERN2at6TensorERKS2_S5_S5_RKSt8optionalIS2_ES9_bb"));
            auto argsort_bincount_with_inv_pos_fn_ptr = reinterpret_cast<argsort_bincount_with_inv_pos_fn>(
                dlsym(handle, "_ZN7pyinfer4perf29argsort_bincount_with_inv_posERKN2at6TensorERS2_S5_S5_l"));
            auto expand_moe_input_with_inv_pos_fn_ptr = reinterpret_cast<expand_moe_input_with_inv_pos_fn>(
                dlsym(handle, "_ZN7pyinfer4perf29expand_moe_input_with_inv_posERN2at6TensorESt8optionalIS2_ERKS2_S7_lll"));
            auto silu_and_mul_quant_fn_ptr = reinterpret_cast<silu_and_mul_quant_fn>(
                dlsym(handle, "_ZN7pyinfer4perf18silu_and_mul_quantERN2at6TensorESt8optionalIS2_ERKS2_l"));
            auto moe_sum_vllm_fn_ptr = reinterpret_cast<moe_sum_vllm_fn>(
                dlsym(handle, "_ZN7pyinfer4perf7moe_sumERN2at6TensorERKS2_St8optionalIS2_ES7_dd"));
            auto fused_indexer_postprocess_fn_ptr = reinterpret_cast<fused_indexer_postprocess_fn>(
                dlsym(handle, "_ZN7pyinfer4perf37fused_deepseek_v2_indexer_postprocessERN2at6TensorES3_S3_S3_RKS2_S5_S5_S5_S5_S5_S5_lbdd"));
            auto indexer_k_cache_fn_ptr = reinterpret_cast<indexer_k_cache_fn>(
                dlsym(handle, "_ZN7pyinfer4perf15indexer_k_cacheERKN2at6TensorERS2_S4_"));
            auto indexer_k_quant_and_cache_fn_ptr = reinterpret_cast<indexer_k_quant_and_cache_fn>(
                dlsym(handle, "_ZN7pyinfer4perf25indexer_k_quant_and_cacheERN2at6TensorES3_S3_lRKSs"));
            auto block_sparse_logits_fn_ptr = reinterpret_cast<block_sparse_logits_fn>(
                dlsym(handle, "_ZN7pyinfer4perf31compute_block_sparse_mqa_logitsERKN2at6TensorES4_S4_S4_S4_S4_RS2_lll"));
            auto select_prefill_topk_fn_ptr = reinterpret_cast<select_prefill_topk_fn>(
                dlsym(handle, "_ZN7pyinfer4perf33select_prefill_topk_block_indicesERKN2at6TensorES4_S4_RS2_"));
            auto select_decode_topk_fn_ptr = reinterpret_cast<select_decode_topk_fn>(
                dlsym(handle, "_ZN7pyinfer4perf32select_decode_topk_block_indicesERKN2at6TensorES4_RS2_"));
            auto map_prefill_indices_fn_ptr = reinterpret_cast<map_prefill_indices_fn>(
                dlsym(handle, "_ZN7pyinfer4perf50map_prefill_request_block_indices_to_global_blocksERKN2at6TensorES4_S4_lbSt8optionalIS2_ES6_bS6_"));
            auto map_decode_indices_fn_ptr = reinterpret_cast<map_decode_indices_fn>(
                dlsym(handle, "_ZN7pyinfer4perf49map_decode_request_block_indices_to_global_blocksERKN2at6TensorES4_S4_lSt8optionalIS2_E"));
            auto sparse_flash_attn_fn_ptr = reinterpret_cast<sparse_flash_attn_fn>(
                dlsym(handle, "_ZN7pyinfer7cuinfer17sparse_flash_attnERN2at6TensorERKS2_S5_S5_d"));
            if (!fused_fn && !rotary_embedding_fn_ptr && !quant_fn && !concat_mla_q_fn_ptr && !concat_and_cache_mla_fn_ptr && !concat_and_cache_mla_int8_fn_ptr && !paged_attention_mla_fn_ptr && !topk_softmax_fn_ptr && !topk_sigmoid_fn_ptr && !grouped_topk_fn_ptr && !scaled_mm_w4a8_fn_ptr && !scaled_mm_w8a8_fn_ptr && !w4a8_group_gemm_fn_ptr && !w8a8_group_gemm_fn_ptr && !w16a16_group_gemm_fn_ptr && !argsort_bincount_with_inv_pos_fn_ptr && !expand_moe_input_with_inv_pos_fn_ptr && !silu_and_mul_quant_fn_ptr && !moe_sum_vllm_fn_ptr) {
                syms.error = dlerror();
                dlclose(handle);
                continue;
            }
            syms.handle = handle;
            syms.fused_add_rms_norm = fused_fn;
            syms.rotary_embedding = rotary_embedding_fn_ptr;
            syms.dynamic_scaled_int8_quant = quant_fn;
            syms.concat_mla_q = concat_mla_q_fn_ptr;
            syms.concat_and_cache_mla = concat_and_cache_mla_fn_ptr;
            syms.concat_and_cache_mla_int8 = concat_and_cache_mla_int8_fn_ptr;
            syms.paged_attention_mla = paged_attention_mla_fn_ptr;
            syms.topk_softmax = topk_softmax_fn_ptr;
            syms.topk_sigmoid = topk_sigmoid_fn_ptr;
            syms.grouped_topk = grouped_topk_fn_ptr;
            syms.scaled_mm_w4a8 = scaled_mm_w4a8_fn_ptr;
            syms.scaled_mm_w8a8 = scaled_mm_w8a8_fn_ptr;
            syms.w4a8_group_gemm = w4a8_group_gemm_fn_ptr;
            syms.w8a8_group_gemm = w8a8_group_gemm_fn_ptr;
            syms.w16a16_group_gemm = w16a16_group_gemm_fn_ptr;
            syms.argsort_bincount_with_inv_pos = argsort_bincount_with_inv_pos_fn_ptr;
            syms.expand_moe_input_with_inv_pos = expand_moe_input_with_inv_pos_fn_ptr;
            syms.silu_and_mul_quant = silu_and_mul_quant_fn_ptr;
            syms.moe_sum_vllm = moe_sum_vllm_fn_ptr;
            syms.fused_indexer_postprocess = fused_indexer_postprocess_fn_ptr;
            syms.indexer_k_cache = indexer_k_cache_fn_ptr;
            syms.indexer_k_quant_and_cache = indexer_k_quant_and_cache_fn_ptr;
            syms.block_sparse_logits = block_sparse_logits_fn_ptr;
            syms.select_prefill_topk = select_prefill_topk_fn_ptr;
            syms.select_decode_topk = select_decode_topk_fn_ptr;
            syms.map_prefill_indices = map_prefill_indices_fn_ptr;
            syms.map_decode_indices = map_decode_indices_fn_ptr;
            syms.sparse_flash_attn = sparse_flash_attn_fn_ptr;
            syms.error.clear();
            break;
        }
        if (!syms.handle && syms.error.empty()) {
            syms.error = "vllm_iluvatar extension not found";
        }

        const char *ix_env_path = std::getenv("INFINICORE_IXTRITURBO_OPS_SO");
        const char *ix_paths[] = {
            ix_env_path,
            "/usr/local/corex-4.5.0.20260619/lib64/python3/dist-packages/ixtriturbo/_C/ops.cpython-312-x86_64-linux-gnu.so",
        };
        for (const char *path : ix_paths) {
            if (path == nullptr || path[0] == '\0') {
                continue;
            }
            void *handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
            if (!handle) {
                syms.ix_error = dlerror();
                continue;
            }
            auto lens_fn = reinterpret_cast<topk_indices_context_lens_fn>(
                dlsym(handle, "_ZN7pyinfer4perf25topk_indices_context_lensERN2at6TensorERKS2_"));
            auto sparse_v2_fn = reinterpret_cast<flash_mla_sparse_v2_fn>(
                dlsym(handle, "_ZN7pyinfer4perf19flash_mla_sparse_v2ERN2at6TensorES3_S3_S3_S3_fRKSt8optionalIS2_E"));
            if (!lens_fn || !sparse_v2_fn) {
                const char *error = dlerror();
                syms.ix_error = error ? error : "ixtriturbo sparse MLA symbols not found";
                dlclose(handle);
                continue;
            }
            syms.ix_handle = handle;
            syms.topk_indices_context_lens = lens_fn;
            syms.flash_mla_sparse_v2 = sparse_v2_fn;
            syms.ix_error.clear();
            break;
        }
        if (!syms.ix_handle && syms.ix_error.empty()) {
            syms.ix_error = "ixtriturbo ops extension not found";
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

bool rotary_embedding_available() {
    return symbols().rotary_embedding != nullptr;
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

bool paged_attention_mla_available() {
    return symbols().paged_attention_mla != nullptr;
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

bool w16a16_group_gemm_available() {
    return symbols().w16a16_group_gemm != nullptr;
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

bool fused_deepseek_v2_indexer_postprocess_available() { return symbols().fused_indexer_postprocess != nullptr; }
bool indexer_k_cache_available() { return symbols().indexer_k_cache != nullptr; }
bool indexer_k_quant_and_cache_available() { return symbols().indexer_k_quant_and_cache != nullptr; }
bool compute_block_sparse_mqa_logits_available() { return symbols().block_sparse_logits != nullptr; }
bool select_prefill_topk_block_indices_available() { return symbols().select_prefill_topk != nullptr; }
bool select_decode_topk_block_indices_available() { return symbols().select_decode_topk != nullptr; }
bool map_prefill_request_block_indices_available() { return symbols().map_prefill_indices != nullptr; }
bool map_decode_request_block_indices_available() { return symbols().map_decode_indices != nullptr; }
bool sparse_flash_mla_available() { return symbols().flash_mla_sparse_v2 != nullptr || symbols().sparse_flash_attn != nullptr; }
bool topk_indices_context_lens_available() { return symbols().topk_indices_context_lens != nullptr; }

void fused_add_rms_norm(at::Tensor &input, at::Tensor &residual, at::Tensor &weight, float epsilon) {
    auto &syms = symbols();
    if (!syms.fused_add_rms_norm) {
        throw std::runtime_error("vllm_iluvatar fused_add_rms_norm unavailable: " + syms.error);
    }
    record_or_run([=]() mutable {
        INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
        auto &run_syms = symbols();
        run_syms.fused_add_rms_norm(input, residual, weight, epsilon);
    });
}

void rotary_embedding(at::Tensor &positions,
                      at::Tensor &query,
                      std::optional<at::Tensor> key,
                      int64_t head_size,
                      at::Tensor &cos_sin_cache,
                      bool is_neox) {
    auto &syms = symbols();
    if (!syms.rotary_embedding) {
        throw std::runtime_error("vllm_iluvatar rotary_embedding unavailable: " + syms.error);
    }
    record_or_run([=]() mutable {
        INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
        auto &run_syms = symbols();
        run_syms.rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox);
    });
}

void dynamic_scaled_int8_quant(at::Tensor &output, at::Tensor &input_scales, const at::Tensor &input) {
    auto &syms = symbols();
    if (!syms.dynamic_scaled_int8_quant) {
        throw std::runtime_error("vllm_iluvatar dynamic_scaled_int8_quant unavailable: " + syms.error);
    }
    record_or_run([=]() mutable {
        INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
        auto &run_syms = symbols();
        run_syms.dynamic_scaled_int8_quant(output, input_scales, input);
    });
}

void concat_mla_q(at::Tensor &ql_nope, at::Tensor &q_pe, at::Tensor &q_out) {
    auto &syms = symbols();
    if (!syms.concat_mla_q) {
        throw std::runtime_error("vllm_iluvatar concat_mla_q unavailable: " + syms.error);
    }
    record_or_run([=]() mutable {
        INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
        auto &run_syms = symbols();
        run_syms.concat_mla_q(ql_nope, q_pe, q_out);
    });
}

void concat_and_cache_mla(at::Tensor &kv_c, at::Tensor &k_pe, at::Tensor &kv_cache, at::Tensor &slot_mapping, const std::string &kv_cache_dtype, at::Tensor &scale) {
    auto &syms = symbols();
    if (!syms.concat_and_cache_mla) {
        throw std::runtime_error("vllm_iluvatar concat_and_cache_mla unavailable: " + syms.error);
    }
    record_or_run([=]() mutable {
        INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
        auto &run_syms = symbols();
        run_syms.concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping, kv_cache_dtype, scale);
    });
}

void concat_and_cache_mla_int8(at::Tensor &kv_c_int8, at::Tensor &kv_c_scale, at::Tensor &k_pe_int8, at::Tensor &k_pe_scale, at::Tensor &kv_cache, at::Tensor &kv_cache_scale, at::Tensor &slot_mapping) {
    auto &syms = symbols();
    if (!syms.concat_and_cache_mla_int8) {
        throw std::runtime_error("vllm_iluvatar concat_and_cache_mla_int8 unavailable: " + syms.error);
    }
    record_or_run([=]() mutable {
        INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
        auto &run_syms = symbols();
        run_syms.concat_and_cache_mla_int8(kv_c_int8, kv_c_scale, k_pe_int8, k_pe_scale, kv_cache, kv_cache_scale, slot_mapping);
    });
}

void paged_attention_mla(at::Tensor &output,
                         at::Tensor &query,
                         at::Tensor &kv_cache,
                         double scale,
                         at::Tensor &block_tables,
                         at::Tensor &context_lens,
                         int64_t max_context_len,
                         bool use_cuda_graph,
                         at::Tensor &softmax_lse) {
    auto &syms = symbols();
    if (!syms.paged_attention_mla) {
        throw std::runtime_error("vllm_iluvatar paged_attention_mla unavailable: " + syms.error);
    }
    const bool graph_recording = context::isGraphRecording();
    record_or_run([=]() mutable {
        INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
        auto &run_syms = symbols();
        run_syms.paged_attention_mla(
            output, query, kv_cache, scale, block_tables, context_lens, max_context_len, (use_cuda_graph || graph_recording), softmax_lse);
    });
}

void topk_softmax(at::Tensor &topk_weights, at::Tensor &topk_ids, at::Tensor &token_expert_indices, const at::Tensor &gating_output, bool renormalize, std::optional<at::Tensor> correction_bias) {
    auto &syms = symbols();
    if (!syms.topk_softmax) {
        throw std::runtime_error("vllm_iluvatar topk_softmax unavailable: " + syms.error);
    }
    record_or_run([=]() mutable {
        INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
        auto &run_syms = symbols();
        run_syms.topk_softmax(topk_weights, topk_ids, token_expert_indices, gating_output, renormalize, correction_bias);
    });
}

void topk_sigmoid(at::Tensor &topk_weights, at::Tensor &topk_ids, at::Tensor &token_expert_indices, const at::Tensor &gating_output, bool renormalize, std::optional<at::Tensor> correction_bias) {
    auto &syms = symbols();
    if (!syms.topk_sigmoid) {
        throw std::runtime_error("vllm_iluvatar topk_sigmoid unavailable: " + syms.error);
    }
    record_or_run([=]() mutable {
        INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
        auto &run_syms = symbols();
        run_syms.topk_sigmoid(topk_weights, topk_ids, token_expert_indices, gating_output, renormalize, correction_bias);
    });
}

void grouped_topk(at::Tensor &topk_weights, at::Tensor &topk_ids, const at::Tensor &scores, std::optional<at::Tensor> bias, int64_t num_expert_group, int64_t topk_group, const std::string &scoring_func, bool renormalize) {
    auto &syms = symbols();
    if (!syms.grouped_topk) {
        throw std::runtime_error("vllm_iluvatar grouped_topk unavailable: " + syms.error);
    }
    record_or_run([=]() mutable {
        INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
        auto &run_syms = symbols();
        run_syms.grouped_topk(topk_weights, topk_ids, scores, bias, num_expert_group, topk_group, scoring_func, renormalize);
    });
}

void scaled_mm_w4a8(at::Tensor &out, const at::Tensor &a, const at::Tensor &b, const at::Tensor &a_scales, const at::Tensor &b_scales, std::optional<at::Tensor> bias, bool trans_weight) {
    auto &syms = symbols();
    if (!syms.scaled_mm_w4a8) {
        throw std::runtime_error("vllm_iluvatar scaled_mm_w4a8 unavailable: " + syms.error);
    }
    record_or_run([=]() mutable {
        INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
        auto &run_syms = symbols();
        run_syms.scaled_mm_w4a8(out, a, b, a_scales, b_scales, bias, trans_weight);
    });
}

void scaled_mm_w8a8(at::Tensor &out, const at::Tensor &a, const at::Tensor &b, const at::Tensor &a_scales, const at::Tensor &b_scales, std::optional<at::Tensor> bias, bool trans_weight) {
    auto &syms = symbols();
    if (!syms.scaled_mm_w8a8) {
        throw std::runtime_error("vllm_iluvatar scaled_mm W8A8 unavailable: " + syms.error);
    }
    record_or_run([=]() mutable {
        INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
        auto &run_syms = symbols();
        run_syms.scaled_mm_w8a8(out, a, b, a_scales, b_scales, bias, trans_weight);
    });
}

void w4a8_group_gemm(at::Tensor &out, const at::Tensor &input, const at::Tensor &weight, const at::Tensor &input_scale, const at::Tensor &weight_scale, const at::Tensor &tokens_per_experts, std::optional<at::Tensor> sorted_token_ids, std::optional<at::Tensor> bias, bool trans_weight, bool is_decode) {
    auto &syms = symbols();
    if (!syms.w4a8_group_gemm) {
        throw std::runtime_error("vllm_iluvatar w4a8_group_gemm unavailable: " + syms.error);
    }
    record_or_run([=]() mutable {
        INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
        auto &run_syms = symbols();
        run_syms.w4a8_group_gemm(out, input, weight, input_scale, weight_scale, tokens_per_experts, sorted_token_ids, bias, trans_weight, is_decode);
    });
}

void w8a8_group_gemm(at::Tensor &out, const at::Tensor &input, const at::Tensor &weight, const at::Tensor &input_scale, const at::Tensor &weight_scale, const at::Tensor &tokens_per_experts, std::optional<at::Tensor> sorted_token_ids, std::optional<at::Tensor> bias, bool trans_weight, bool is_decode) {
    auto &syms = symbols();
    if (!syms.w8a8_group_gemm) {
        throw std::runtime_error("vllm_iluvatar w8a8_group_gemm unavailable: " + syms.error);
    }
    record_or_run([=]() mutable {
        INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
        auto &run_syms = symbols();
        run_syms.w8a8_group_gemm(out, input, weight, input_scale, weight_scale, tokens_per_experts, sorted_token_ids, bias, trans_weight, is_decode);
    });
}

void w16a16_group_gemm(at::Tensor &out, const at::Tensor &input, const at::Tensor &weight, const at::Tensor &tokens_per_experts, std::optional<at::Tensor> sorted_token_ids, std::optional<at::Tensor> bias, bool trans_weight, bool is_decode) {
    auto &syms = symbols();
    if (!syms.w16a16_group_gemm) {
        throw std::runtime_error("vllm_iluvatar w16a16_group_gemm unavailable: " + syms.error);
    }
    record_or_run([=]() mutable {
        INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
        auto &run_syms = symbols();
        run_syms.w16a16_group_gemm(out, input, weight, tokens_per_experts, sorted_token_ids, bias, trans_weight, is_decode);
    });
}

void argsort_bincount_with_inv_pos(const at::Tensor &topk_ids, at::Tensor &tokens_per_experts, at::Tensor &sorted_indices, at::Tensor &inv_pos, int64_t num_experts) {
    auto &syms = symbols();
    if (!syms.argsort_bincount_with_inv_pos) {
        throw std::runtime_error("vllm_iluvatar argsort_bincount_with_inv_pos unavailable: " + syms.error);
    }
    record_or_run([=]() mutable {
        INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
        auto &run_syms = symbols();
        run_syms.argsort_bincount_with_inv_pos(topk_ids, tokens_per_experts, sorted_indices, inv_pos, num_experts);
    });
}

void expand_moe_input_with_inv_pos(at::Tensor &expand_states, std::optional<at::Tensor> expand_scales, const at::Tensor &hidden_states, const at::Tensor &inv_pos, int64_t top_k, int64_t group_size, int64_t format) {
    auto &syms = symbols();
    if (!syms.expand_moe_input_with_inv_pos) {
        throw std::runtime_error("vllm_iluvatar expand_moe_input_with_inv_pos unavailable: " + syms.error);
    }
    record_or_run([=]() mutable {
        INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
        auto &run_syms = symbols();
        run_syms.expand_moe_input_with_inv_pos(expand_states, expand_scales, hidden_states, inv_pos, top_k, group_size, format);
    });
}

void silu_and_mul_quant(at::Tensor &output, std::optional<at::Tensor> output_scale, const at::Tensor &input, int64_t format) {
    auto &syms = symbols();
    if (!syms.silu_and_mul_quant) {
        throw std::runtime_error("vllm_iluvatar silu_and_mul_quant unavailable: " + syms.error);
    }
    record_or_run([=]() mutable {
        INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
        auto &run_syms = symbols();
        run_syms.silu_and_mul_quant(output, output_scale, input, format);
    });
}

void moe_sum_vllm(at::Tensor &output, const at::Tensor &input, std::optional<at::Tensor> topk_weights, std::optional<at::Tensor> extra_residual, double routed_scale, double residual_scale) {
    auto &syms = symbols();
    if (!syms.moe_sum_vllm) {
        throw std::runtime_error("vllm_iluvatar moe_sum unavailable: " + syms.error);
    }
    record_or_run([=]() mutable {
        INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
        auto &run_syms = symbols();
        run_syms.moe_sum_vllm(output, input, topk_weights, extra_residual, routed_scale, residual_scale);
    });
}

void fused_deepseek_v2_indexer_postprocess(at::Tensor &q_out, at::Tensor &k_out, at::Tensor &weights_out, at::Tensor &kv_cache, const at::Tensor &slot_mapping, const at::Tensor &q, const at::Tensor &kw, const at::Tensor &norm_weight, const at::Tensor &norm_bias, const at::Tensor &positions, const at::Tensor &cos_sin_cache, int64_t num_cache_tokens, bool is_neox, double eps, double weights_scale) {
    if (!symbols().fused_indexer_postprocess) {
        throw std::runtime_error("vllm_iluvatar fused indexer postprocess unavailable: " + symbols().error);
    }
    record_or_run([=]() mutable { INFINICORE_VLLM_ILUVATAR_STREAM_GUARD(); symbols().fused_indexer_postprocess(q_out, k_out, weights_out, kv_cache, slot_mapping, q, kw, norm_weight, norm_bias, positions, cos_sin_cache, num_cache_tokens, is_neox, eps, weights_scale); });
}

void indexer_k_cache(const at::Tensor &k, at::Tensor &kv_cache, const at::Tensor &slot_mapping) {
    if (!symbols().indexer_k_cache) {
        throw std::runtime_error("vllm_iluvatar indexer_k_cache unavailable: " + symbols().error);
    }
    record_or_run([=]() mutable { INFINICORE_VLLM_ILUVATAR_STREAM_GUARD(); symbols().indexer_k_cache(k, kv_cache, slot_mapping); });
}

void indexer_k_quant_and_cache(at::Tensor &k, at::Tensor &kv_cache, at::Tensor &slot_mapping, int64_t quant_block_size, const std::string &scale_fmt) {
    if (!symbols().indexer_k_quant_and_cache) {
        throw std::runtime_error("vllm_iluvatar indexer_k_quant_and_cache unavailable: " + symbols().error);
    }
    record_or_run([=]() mutable { INFINICORE_VLLM_ILUVATAR_STREAM_GUARD(); symbols().indexer_k_quant_and_cache(k, kv_cache, slot_mapping, quant_block_size, scale_fmt); });
}

void compute_block_sparse_mqa_logits(const at::Tensor &q, const at::Tensor &kv_cache, const at::Tensor &cu_seqlens_q, const at::Tensor &cu_seqlens_kv, const at::Tensor &block_table, const at::Tensor &weights, at::Tensor &logits, int64_t max_q_len, int64_t max_kv_len, int64_t max_context_len) {
    if (!symbols().block_sparse_logits) {
        throw std::runtime_error("vllm_iluvatar block sparse logits unavailable: " + symbols().error);
    }
    record_or_run([=]() mutable { INFINICORE_VLLM_ILUVATAR_STREAM_GUARD(); symbols().block_sparse_logits(q, kv_cache, cu_seqlens_q, cu_seqlens_kv, block_table, weights, logits, max_q_len, max_kv_len, max_context_len); });
}

void select_prefill_topk_block_indices(const at::Tensor &logits, const at::Tensor &cu_seqlen_ks, const at::Tensor &cu_seqlen_ke, at::Tensor &topk_indices) {
    if (!symbols().select_prefill_topk) {
        throw std::runtime_error("vllm_iluvatar prefill topk unavailable: " + symbols().error);
    }
    record_or_run([=]() mutable { INFINICORE_VLLM_ILUVATAR_STREAM_GUARD(); symbols().select_prefill_topk(logits, cu_seqlen_ks, cu_seqlen_ke, topk_indices); });
}

void select_decode_topk_block_indices(const at::Tensor &logits, const at::Tensor &seq_lens, at::Tensor &topk_indices) {
    if (!symbols().select_decode_topk) {
        throw std::runtime_error("vllm_iluvatar decode topk unavailable: " + symbols().error);
    }
    record_or_run([=]() mutable { INFINICORE_VLLM_ILUVATAR_STREAM_GUARD(); symbols().select_decode_topk(logits, seq_lens, topk_indices); });
}

void map_prefill_request_block_indices(at::Tensor &output, const at::Tensor &req_id, const at::Tensor &block_table, const at::Tensor &token_indices, int64_t block_size, bool has_prefill_workspace, std::optional<at::Tensor> prefill_workspace_request_ids, std::optional<at::Tensor> prefill_workspace_starts) {
    if (!symbols().map_prefill_indices) {
        throw std::runtime_error("vllm_iluvatar prefill index mapping unavailable: " + symbols().error);
    }
    record_or_run([=]() mutable { INFINICORE_VLLM_ILUVATAR_STREAM_GUARD(); (void)symbols().map_prefill_indices(req_id, block_table, token_indices, block_size, has_prefill_workspace, prefill_workspace_request_ids, prefill_workspace_starts, false, output); });
}

void map_decode_request_block_indices(at::Tensor &output, const at::Tensor &req_id, const at::Tensor &block_table, const at::Tensor &token_indices, int64_t block_size) {
    if (!symbols().map_decode_indices) {
        throw std::runtime_error("vllm_iluvatar decode index mapping unavailable: " + symbols().error);
    }
    record_or_run([=]() mutable { INFINICORE_VLLM_ILUVATAR_STREAM_GUARD(); (void)symbols().map_decode_indices(req_id, block_table, token_indices, block_size, output); });
}

void topk_indices_context_lens(at::Tensor &topk_lens, const at::Tensor &indices) {
    if (!symbols().topk_indices_context_lens) {
        throw std::runtime_error("ixtriturbo topk_indices_context_lens unavailable: " + symbols().ix_error);
    }
    record_or_run([=]() mutable { INFINICORE_VLLM_ILUVATAR_STREAM_GUARD(); symbols().topk_indices_context_lens(topk_lens, indices); });
}

void sparse_flash_mla(at::Tensor &output, at::Tensor &query, at::Tensor &kv_cache, at::Tensor &indices, at::Tensor &topk_lens, float scale, std::optional<at::Tensor> attn_sink) {
    auto &syms = symbols();
    if (!syms.flash_mla_sparse_v2 && !syms.sparse_flash_attn) {
        throw std::runtime_error("Iluvatar sparse FlashMLA unavailable: " + syms.error + "; " + syms.ix_error);
    }
    record_or_run([=]() mutable {
        INFINICORE_VLLM_ILUVATAR_STREAM_GUARD();
        auto &run_syms = symbols();
        if (run_syms.flash_mla_sparse_v2) {
            run_syms.flash_mla_sparse_v2(output, query, kv_cache, indices, topk_lens, scale, attn_sink);
        } else {
            run_syms.sparse_flash_attn(output, query, kv_cache, indices, scale);
        }
    });
}

#undef INFINICORE_VLLM_ILUVATAR_STREAM_GUARD

} // namespace infinicore::adaptor::vllm_iluvatar
#endif // ENABLE_ATEN
