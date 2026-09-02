#include "kv_caching_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::kv_caching::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t k_cache,
    infiniopTensorDescriptor_t v_cache,
    infiniopTensorDescriptor_t k,
    infiniopTensorDescriptor_t v,
    infiniopTensorDescriptor_t past_kv_lengths) {

    auto info = KVCachingInfo::createKVCachingInfo(k_cache, v_cache, k, v, past_kv_lengths);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        nullptr,
        info.take(),
        0,
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata, typename Tidx>
static void calculateImpl(
    const KVCachingInfo &info,
    Tdata *k_cache,
    Tdata *v_cache,
    const Tdata *k,
    const Tdata *v,
    const Tidx *past_kv_lengths) {

    const auto batch_size = static_cast<ptrdiff_t>(info.batch_size);
    const auto num_kv_heads = static_cast<ptrdiff_t>(info.num_kv_heads);
    const auto seq_len = static_cast<ptrdiff_t>(info.seq_len);
    const auto hidden_dim = static_cast<ptrdiff_t>(info.hidden_dim);

    const ptrdiff_t total = batch_size * num_kv_heads * seq_len * hidden_dim;

#ifdef ENABLE_OMP
#pragma omp parallel for schedule(static)
#endif
    for (ptrdiff_t linear = 0; linear < total; ++linear) {
        ptrdiff_t idx = linear;
        const ptrdiff_t d = idx % hidden_dim;
        idx /= hidden_dim;
        const ptrdiff_t s = idx % seq_len;
        idx /= seq_len;
        const ptrdiff_t h = idx % num_kv_heads;
        const ptrdiff_t b = idx / num_kv_heads;

        const ptrdiff_t cache_s = static_cast<ptrdiff_t>(past_kv_lengths[b]) + s;

        const ptrdiff_t k_cache_offset = b * info.k_cache_strides_0
                                       + h * info.k_cache_strides_1
                                       + cache_s * info.k_cache_strides_2
                                       + d * info.k_cache_strides_3;
        const ptrdiff_t v_cache_offset = b * info.v_cache_strides_0
                                       + h * info.v_cache_strides_1
                                       + cache_s * info.v_cache_strides_2
                                       + d * info.v_cache_strides_3;
        const ptrdiff_t k_offset = b * info.k_strides_0
                                 + h * info.k_strides_1
                                 + s * info.k_strides_2
                                 + d * info.k_strides_3;
        const ptrdiff_t v_offset = b * info.v_strides_0
                                 + h * info.v_strides_1
                                 + s * info.v_strides_2
                                 + d * info.v_strides_3;

        k_cache[k_cache_offset] = k[k_offset];
        v_cache[v_cache_offset] = v[v_offset];
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *k_cache,
    void *v_cache,
    const void *k,
    const void *v,
    const void *past_kv_lengths,
    void *stream) const {

#define DISPATCH_PAST_LEN(TDATA)                                                           \
    if (_info.past_len_dtype == INFINI_DTYPE_I32) {                                        \
        calculateImpl<TDATA, int32_t>(_info,                                               \
                                      reinterpret_cast<TDATA *>(k_cache),                  \
                                      reinterpret_cast<TDATA *>(v_cache),                  \
                                      reinterpret_cast<const TDATA *>(k),                  \
                                      reinterpret_cast<const TDATA *>(v),                  \
                                      reinterpret_cast<const int32_t *>(past_kv_lengths)); \
        return INFINI_STATUS_SUCCESS;                                                      \
    }                                                                                      \
    if (_info.past_len_dtype == INFINI_DTYPE_I64) {                                        \
        calculateImpl<TDATA, int64_t>(_info,                                               \
                                      reinterpret_cast<TDATA *>(k_cache),                  \
                                      reinterpret_cast<TDATA *>(v_cache),                  \
                                      reinterpret_cast<const TDATA *>(k),                  \
                                      reinterpret_cast<const TDATA *>(v),                  \
                                      reinterpret_cast<const int64_t *>(past_kv_lengths)); \
        return INFINI_STATUS_SUCCESS;                                                      \
    }                                                                                      \
    return INFINI_STATUS_BAD_TENSOR_DTYPE

    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        DISPATCH_PAST_LEN(fp16_t);
    case INFINI_DTYPE_BF16:
        DISPATCH_PAST_LEN(bf16_t);
    case INFINI_DTYPE_F32:
        DISPATCH_PAST_LEN(float);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#undef DISPATCH_PAST_LEN
}

} // namespace op::kv_caching::cpu
