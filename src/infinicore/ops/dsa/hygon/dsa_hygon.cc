#if defined(ENABLE_HYGON_API)
#include "../../infiniop_impl.hpp"
#include "../../vendor_ops/vendor_ops_dispatch.hpp"

#include "infinicore/context/context.hpp"
#include "infinicore/ops/dsa.hpp"
#include "infiniop/ops/dsa_index_utils.h"
#include "infiniop/ops/select_decode_topk_block_indices.h"
#include "infiniop/ops/select_prefill_topk_block_indices.h"
#include "infiniop/ops/sparse_flash_mla.h"

#include <dlfcn.h>
#include <hip/hip_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>

namespace infinicore::op::dsa_impl::hygon {

#if defined(ENABLE_VENDOR_OPS) && defined(ENABLE_ATEN)
namespace {

struct alignas(8) SparseAttnFwdParams {
    int32_t s_q;
    int32_t s_kv;
    int32_t h_q;
    int32_t h_kv;
    int32_t d_qk;
    int32_t d_v;
    int32_t topk;
    float sm_scale;
    float sm_scale_div_log2;
    int32_t padding0;
    const void *q;
    const void *kv;
    const int32_t *indices;
    const float *attn_sink;
    const int32_t *topk_length;
    int32_t stride_q_s_q;
    int32_t stride_q_h_q;
    int32_t stride_kv_s_kv;
    int32_t stride_kv_h_kv;
    int32_t stride_indices_s_q;
    int32_t stride_indices_h_kv;
    void *out;
    float *max_logits;
    float *lse;
    int32_t num_sm;
    int32_t padding1;
    void *stream;
};

static_assert(sizeof(SparseAttnFwdParams) == 144);
static_assert(offsetof(SparseAttnFwdParams, q) == 40);
static_assert(offsetof(SparseAttnFwdParams, out) == 104);
static_assert(offsetof(SparseAttnFwdParams, stream) == 136);

using VendorSparseFlashMla = void (*)(const SparseAttnFwdParams &);

VendorSparseFlashMla vendor_sparse_flash_mla() {
    static const auto function = []() -> VendorSparseFlashMla {
        const char *configured = std::getenv("INFINICORE_HYGON_FLASH_MLA_SO");
        const char *path = configured != nullptr && configured[0] != '\0'
                             ? configured
                             : "/usr/local/lib/python3.10/dist-packages/"
                               "flash_mla/cuda.cpython-310-x86_64-linux-gnu.so";
        constexpr const char *symbol = "_ZN5gfx9314run_fwd_kernelERK19SparseAttnFwdParams";
        dlerror();
        if (void *address = dlsym(RTLD_DEFAULT, symbol)) {
            return reinterpret_cast<VendorSparseFlashMla>(address);
        }
        dlerror();
        void *handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            return nullptr;
        }
        dlerror();
        return reinterpret_cast<VendorSparseFlashMla>(
            dlsym(handle, symbol));
    }();
    return function;
}

int32_t current_device_num_sm() {
    static thread_local int cached_device = -1;
    static thread_local int32_t cached_num_sm = 0;
    int device = -1;
    if (hipGetDevice(&device) != hipSuccess) {
        return 0;
    }
    if (device != cached_device) {
        hipDeviceProp_t properties{};
        if (hipGetDeviceProperties(&properties, device) != hipSuccess) {
            return 0;
        }
        cached_device = device;
        cached_num_sm = properties.multiProcessorCount;
    }
    return cached_num_sm;
}

bool try_vendor_sparse_flash_mla(
    Tensor output,
    const Tensor &query,
    const Tensor &kv_cache,
    const Tensor &indices,
    const Tensor &topk_lens,
    float scale,
    std::optional<Tensor> attn_sink) {
    if (std::getenv("INFINICORE_HYGON_DISABLE_FLASH_MLA") != nullptr) {
        return false;
    }
    const auto function = vendor_sparse_flash_mla();
    const int32_t num_sm = current_device_num_sm();
    if (function == nullptr || num_sm <= 0
        || output->dtype() != DataType::BF16
        || query->dtype() != DataType::BF16
        || kv_cache->dtype() != DataType::BF16
        || indices->dtype() != DataType::I32
        || topk_lens->dtype() != DataType::I32
        || !output->is_contiguous()
        || !query->is_contiguous()
        || !kv_cache->is_contiguous()
        || !indices->is_contiguous()
        || !topk_lens->is_contiguous()
        || query->size(2) != 576
        || output->size(2) != 512
        || kv_cache->size(2) != 576
        || kv_cache->size(1) != 1
        || indices->size(1) != kv_cache->size(1)
        || topk_lens->size(0) != query->size(0)
        || (attn_sink
            && ((*attn_sink)->dtype() != DataType::F32
                || !(*attn_sink)->is_contiguous()
                || (*attn_sink)->numel() != query->size(1)))) {
        return false;
    }

    auto max_logits = Tensor::empty(
        {query->size(0), query->size(1)}, DataType::F32, query->device());
    auto lse = Tensor::empty(
        {query->size(0), query->size(1)}, DataType::F32, query->device());
    SparseAttnFwdParams params{
        static_cast<int32_t>(query->size(0)),
        static_cast<int32_t>(kv_cache->size(0)),
        static_cast<int32_t>(query->size(1)),
        static_cast<int32_t>(kv_cache->size(1)),
        static_cast<int32_t>(query->size(2)),
        static_cast<int32_t>(output->size(2)),
        static_cast<int32_t>(indices->size(2)),
        scale,
        scale * 1.4426950408889634f,
        0,
        query->data(),
        kv_cache->data(),
        reinterpret_cast<const int32_t *>(indices->data()),
        attn_sink ? reinterpret_cast<const float *>((*attn_sink)->data())
                  : nullptr,
        reinterpret_cast<const int32_t *>(topk_lens->data()),
        static_cast<int32_t>(query->stride(0)),
        static_cast<int32_t>(query->stride(1)),
        static_cast<int32_t>(kv_cache->stride(0)),
        static_cast<int32_t>(kv_cache->stride(1)),
        static_cast<int32_t>(indices->stride(0)),
        static_cast<int32_t>(indices->stride(1)),
        output->data(),
        reinterpret_cast<float *>(max_logits->data()),
        reinterpret_cast<float *>(lse->data()),
        num_sm,
        0,
        context::getStream(),
    };
    function(params);
    return true;
}

} // namespace
#endif

void select_prefill_topk(
    Tensor topk_indices,
    const Tensor &logits,
    const Tensor &cu_seqlen_ks,
    const Tensor &cu_seqlen_ke) {
    INFINICORE_CHECK_ERROR(infiniopSelectPrefillTopkBlockIndices(
        context::getInfiniopHandle(logits->device()),
        topk_indices->desc(),
        logits->desc(),
        cu_seqlen_ks->desc(),
        cu_seqlen_ke->desc(),
        topk_indices->data(),
        logits->data(),
        cu_seqlen_ks->data(),
        cu_seqlen_ke->data(),
        context::getStream()));
}

void select_decode_topk(
    Tensor topk_indices,
    const Tensor &logits,
    const Tensor &seq_lens) {
    INFINICORE_CHECK_ERROR(infiniopSelectDecodeTopkBlockIndices(
        context::getInfiniopHandle(logits->device()),
        topk_indices->desc(),
        logits->desc(),
        seq_lens->desc(),
        topk_indices->data(),
        logits->data(),
        seq_lens->data(),
        context::getStream()));
}

void map_decode_indices(
    Tensor output,
    const Tensor &request_ids,
    const Tensor &block_table,
    const Tensor &token_indices,
    int64_t block_size) {
    INFINICORE_CHECK_ERROR(infiniopMapDecodeRequestBlockIndices(
        context::getInfiniopHandle(output->device()),
        output->desc(),
        request_ids->desc(),
        block_table->desc(),
        token_indices->desc(),
        output->data(),
        request_ids->data(),
        block_table->data(),
        token_indices->data(),
        block_size,
        context::getStream()));
}

void topk_context_lens(Tensor topk_lens, const Tensor &indices) {
    INFINICORE_CHECK_ERROR(infiniopTopkIndicesContextLens(
        context::getInfiniopHandle(indices->device()),
        topk_lens->desc(),
        indices->desc(),
        topk_lens->data(),
        indices->data(),
        context::getStream()));
}

void sparse_flash_mla(
    Tensor output,
    const Tensor &query,
    const Tensor &kv_cache,
    const Tensor &indices,
    const Tensor &topk_lens,
    float scale,
    std::optional<Tensor> attn_sink) {
    if (attn_sink.has_value()) {
        throw std::runtime_error(
            "Hygon sparse_flash_mla fallback does not support attention sinks");
    }
#if defined(ENABLE_VENDOR_OPS) && defined(ENABLE_ATEN)
    if (try_vendor_sparse_flash_mla(
            output, query, kv_cache, indices, topk_lens, scale, attn_sink)) {
        return;
    }
#endif
    INFINICORE_CHECK_ERROR(infiniopSparseFlashMla(
        context::getInfiniopHandle(output->device()),
        output->desc(),
        query->desc(),
        kv_cache->desc(),
        indices->desc(),
        topk_lens->desc(),
        output->data(),
        query->data(),
        kv_cache->data(),
        indices->data(),
        topk_lens->data(),
        scale,
        context::getStream()));
}

static bool registered = []() {
    vendor_ops::select_prefill_topk_dispatcher().registerDevice(
        Device::Type::HYGON, &select_prefill_topk);
    vendor_ops::select_decode_topk_dispatcher().registerDevice(
        Device::Type::HYGON, &select_decode_topk);
    vendor_ops::map_decode_indices_dispatcher().registerDevice(
        Device::Type::HYGON, &map_decode_indices);
    vendor_ops::topk_context_lens_dispatcher().registerDevice(
        Device::Type::HYGON, &topk_context_lens);
    vendor_ops::sparse_flash_mla_dispatcher().registerDevice(
        Device::Type::HYGON, &sparse_flash_mla);
    return true;
}();

} // namespace infinicore::op::dsa_impl::hygon
#endif
