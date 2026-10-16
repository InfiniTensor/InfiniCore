#if defined(ENABLE_ATEN) && defined(ENABLE_HYGON_API)

#include "infinicore/adaptor/hygon_lightop_adaptor.hpp"

#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace infinicore::adaptor::hygon_lightop {
namespace {

using fused_quantize_and_store_mla_kv_cache_fn = void (*)(
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    const std::string &,
    float);
using top_k_per_row_prefill_fn = void (*)(
    const at::Tensor &,
    const at::Tensor &,
    const at::Tensor &,
    at::Tensor &,
    int64_t,
    int64_t,
    int64_t,
    int64_t);
using top_k_per_row_decode_fn = void (*)(
    const at::Tensor &,
    int64_t,
    const at::Tensor &,
    at::Tensor &,
    int64_t,
    int64_t,
    int64_t,
    int64_t);
using convert_req_index_to_global_index_fn = at::Tensor (*)(
    at::Tensor,
    at::Tensor,
    at::Tensor,
    int,
    int,
    bool,
    std::optional<at::Tensor>,
    std::optional<at::Tensor>);

struct Symbols {
    void *handle = nullptr;
    fused_quantize_and_store_mla_kv_cache_fn
        fused_quantize_and_store_mla_kv_cache = nullptr;
    top_k_per_row_prefill_fn top_k_per_row_prefill = nullptr;
    top_k_per_row_decode_fn top_k_per_row_decode = nullptr;
    convert_req_index_to_global_index_fn convert_req_index_to_global_index =
        nullptr;
    std::string error;
};

Symbols &symbols() {
    static Symbols value;
    static std::once_flag once;
    std::call_once(once, []() {
        const char *env_path = std::getenv("INFINICORE_HYGON_LIGHTOP_SO");
        const char *paths[] = {
            env_path,
            "/usr/local/lib/python3.10/dist-packages/lightop/op.cpython-310-x86_64-linux-gnu.so",
        };
        for (const char *path : paths) {
            if (path == nullptr || path[0] == '\0') {
                continue;
            }
            void *handle = dlopen(path, RTLD_LAZY | RTLD_LOCAL);
            if (handle == nullptr) {
                value.error = dlerror();
                continue;
            }
            auto cache_fn = reinterpret_cast<fused_quantize_and_store_mla_kv_cache_fn>(
                dlsym(handle,
                      "_ZN2at6native37fused_quantize_and_store_mla_kv_cacheENS_6TensorES1_S1_S1_RKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEEf"));
            auto prefill_topk_fn = reinterpret_cast<top_k_per_row_prefill_fn>(
                dlsym(handle,
                      "_ZN2at6native21top_k_per_row_prefillERKNS_6TensorES3_S3_RS1_llll"));
            auto decode_topk_fn = reinterpret_cast<top_k_per_row_decode_fn>(
                dlsym(handle,
                      "_ZN2at6native20top_k_per_row_decodeERKNS_6TensorElS3_RS1_llll"));
            auto convert_indices_fn =
                reinterpret_cast<convert_req_index_to_global_index_fn>(
                    dlsym(handle,
                          "_ZN2at6native33convert_req_index_to_global_indexENS_6TensorES1_S1_iibSt8optionalIS1_ES3_"));
            if (cache_fn == nullptr || prefill_topk_fn == nullptr
                || decode_topk_fn == nullptr || convert_indices_fn == nullptr) {
                value.error = dlerror();
                dlclose(handle);
                continue;
            }
            value.handle = handle;
            value.fused_quantize_and_store_mla_kv_cache = cache_fn;
            value.top_k_per_row_prefill = prefill_topk_fn;
            value.top_k_per_row_decode = decode_topk_fn;
            value.convert_req_index_to_global_index = convert_indices_fn;
            value.error.clear();
            return;
        }
    });
    return value;
}

} // namespace

bool fused_quantize_and_store_mla_kv_cache_available() {
    return symbols().fused_quantize_and_store_mla_kv_cache != nullptr;
}

bool top_k_per_row_available() {
    const auto &value = symbols();
    return value.top_k_per_row_prefill != nullptr
        && value.top_k_per_row_decode != nullptr;
}

bool convert_req_index_to_global_index_available() {
    return symbols().convert_req_index_to_global_index != nullptr;
}

void fused_quantize_and_store_mla_kv_cache(
    at::Tensor kv_c,
    at::Tensor k_pe,
    at::Tensor kv_cache,
    at::Tensor slot_mapping,
    const std::string &kv_cache_dtype,
    float scale) {
    auto &value = symbols();
    if (value.fused_quantize_and_store_mla_kv_cache == nullptr) {
        throw std::runtime_error(
            "LightOp fused_quantize_and_store_mla_kv_cache is unavailable: "
            + value.error);
    }
    value.fused_quantize_and_store_mla_kv_cache(
        std::move(kv_c),
        std::move(k_pe),
        std::move(kv_cache),
        std::move(slot_mapping),
        kv_cache_dtype,
        scale);
}

void top_k_per_row_prefill(const at::Tensor &logits,
                           const at::Tensor &row_starts,
                           const at::Tensor &row_ends,
                           at::Tensor &topk_indices) {
    auto &value = symbols();
    if (value.top_k_per_row_prefill == nullptr) {
        throw std::runtime_error(
            "LightOp top_k_per_row_prefill is unavailable: " + value.error);
    }
    value.top_k_per_row_prefill(
        logits,
        row_starts,
        row_ends,
        topk_indices,
        logits.size(0),
        logits.stride(0),
        logits.stride(1),
        topk_indices.size(1));
}

void top_k_per_row_decode(const at::Tensor &logits,
                          const at::Tensor &seq_lens,
                          at::Tensor &topk_indices) {
    auto &value = symbols();
    if (value.top_k_per_row_decode == nullptr) {
        throw std::runtime_error(
            "LightOp top_k_per_row_decode is unavailable: " + value.error);
    }
    value.top_k_per_row_decode(
        logits,
        1,
        seq_lens,
        topk_indices,
        logits.size(0),
        logits.stride(0),
        logits.stride(1),
        topk_indices.size(1));
}

at::Tensor convert_req_index_to_global_index(
    at::Tensor req_id,
    at::Tensor block_table,
    at::Tensor token_indices,
    int block_size,
    bool has_prefill_workspace,
    std::optional<at::Tensor> prefill_workspace_request_ids,
    std::optional<at::Tensor> prefill_workspace_starts) {
    auto &value = symbols();
    if (value.convert_req_index_to_global_index == nullptr) {
        throw std::runtime_error(
            "LightOp convert_req_index_to_global_index is unavailable: "
            + value.error);
    }
    constexpr int block_n = 128;
    return value.convert_req_index_to_global_index(
        std::move(req_id),
        std::move(block_table),
        std::move(token_indices),
        block_size,
        block_n,
        has_prefill_workspace,
        std::move(prefill_workspace_request_ids),
        std::move(prefill_workspace_starts));
}

} // namespace infinicore::adaptor::hygon_lightop

#endif
