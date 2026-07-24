
#include "../../../devices/moore/moore_common.h"
#include "kv_caching_moore.h"

namespace {

template <typename Tdata, typename Tidx>
__global__ void kvCachingMooreKernel(
    Tdata *k_cache,
    Tdata *v_cache,
    const Tdata *k,
    const Tdata *v,
    const Tidx *past_kv_lengths,
    size_t batch_size,
    size_t num_kv_heads,
    size_t seq_len,
    size_t hidden_dim,
    ptrdiff_t k_cache_strides_0,
    ptrdiff_t k_cache_strides_1,
    ptrdiff_t k_cache_strides_2,
    ptrdiff_t k_cache_strides_3,
    ptrdiff_t v_cache_strides_0,
    ptrdiff_t v_cache_strides_1,
    ptrdiff_t v_cache_strides_2,
    ptrdiff_t v_cache_strides_3,
    ptrdiff_t k_strides_0,
    ptrdiff_t k_strides_1,
    ptrdiff_t k_strides_2,
    ptrdiff_t k_strides_3,
    ptrdiff_t v_strides_0,
    ptrdiff_t v_strides_1,
    ptrdiff_t v_strides_2,
    ptrdiff_t v_strides_3) {

    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = batch_size * num_kv_heads * seq_len * hidden_dim;
    size_t grid_size = blockDim.x * gridDim.x;

    for (size_t linear = tid; linear < total; linear += grid_size) {
        size_t idx = linear;
        size_t d = idx % hidden_dim;
        idx /= hidden_dim;
        size_t s = idx % seq_len;
        idx /= seq_len;
        size_t h = idx % num_kv_heads;
        size_t b = idx / num_kv_heads;

        ptrdiff_t cache_s = static_cast<ptrdiff_t>(past_kv_lengths[b]) + static_cast<ptrdiff_t>(s);
        ptrdiff_t k_cache_offset = static_cast<ptrdiff_t>(b) * k_cache_strides_0
                                 + static_cast<ptrdiff_t>(h) * k_cache_strides_1
                                 + cache_s * k_cache_strides_2
                                 + static_cast<ptrdiff_t>(d) * k_cache_strides_3;
        ptrdiff_t v_cache_offset = static_cast<ptrdiff_t>(b) * v_cache_strides_0
                                 + static_cast<ptrdiff_t>(h) * v_cache_strides_1
                                 + cache_s * v_cache_strides_2
                                 + static_cast<ptrdiff_t>(d) * v_cache_strides_3;
        ptrdiff_t k_src_offset = static_cast<ptrdiff_t>(b) * k_strides_0
                               + static_cast<ptrdiff_t>(h) * k_strides_1
                               + static_cast<ptrdiff_t>(s) * k_strides_2
                               + static_cast<ptrdiff_t>(d) * k_strides_3;
        ptrdiff_t v_src_offset = static_cast<ptrdiff_t>(b) * v_strides_0
                               + static_cast<ptrdiff_t>(h) * v_strides_1
                               + static_cast<ptrdiff_t>(s) * v_strides_2
                               + static_cast<ptrdiff_t>(d) * v_strides_3;
        k_cache[k_cache_offset] = k[k_src_offset];
        v_cache[v_cache_offset] = v[v_src_offset];
    }
}

template <typename Tdata, typename Tidx>
infiniStatus_t launchMooreKernel(
    musaStream_t stream,
    void *k_cache,
    void *v_cache,
    const void *k,
    const void *v,
    const void *past_kv_lengths,
    const op::kv_caching::KVCachingInfo &info) {
    constexpr unsigned int block_size = 256;
    size_t total = info.batch_size * info.num_kv_heads * info.seq_len * info.hidden_dim;
    if (total == 0) {
        return INFINI_STATUS_SUCCESS;
    }
    size_t num_blocks = (total + block_size - 1) / block_size;
    kvCachingMooreKernel<Tdata, Tidx><<<num_blocks, block_size, 0, stream>>>(
        static_cast<Tdata *>(k_cache), static_cast<Tdata *>(v_cache),
        static_cast<const Tdata *>(k), static_cast<const Tdata *>(v),
        static_cast<const Tidx *>(past_kv_lengths),
        info.batch_size, info.num_kv_heads, info.seq_len, info.hidden_dim,
        info.k_cache_strides_0, info.k_cache_strides_1, info.k_cache_strides_2, info.k_cache_strides_3,
        info.v_cache_strides_0, info.v_cache_strides_1, info.v_cache_strides_2, info.v_cache_strides_3,
        info.k_strides_0, info.k_strides_1, info.k_strides_2, info.k_strides_3,
        info.v_strides_0, info.v_strides_1, info.v_strides_2, info.v_strides_3);
    return INFINI_STATUS_SUCCESS;
}

} // namespace

namespace op::kv_caching::moore {

struct Descriptor::Opaque {
    std::shared_ptr<device::moore::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t k_cache,
    infiniopTensorDescriptor_t v_cache,
    infiniopTensorDescriptor_t k,
    infiniopTensorDescriptor_t v,
    infiniopTensorDescriptor_t past_kv_lengths) {
    auto handle = reinterpret_cast<device::moore::Handle *>(handle_);
    auto info = KVCachingInfo::createKVCachingInfo(k_cache, v_cache, k, v, past_kv_lengths);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        new Opaque{handle->internal()},
        info.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
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
    (void)workspace;
    (void)workspace_size;
    auto musa_stream = reinterpret_cast<musaStream_t>(stream);

#define LAUNCH_WITH_IDX(TDATA)                                                                   \
    if (_info.past_len_dtype == INFINI_DTYPE_I32) {                                              \
        return launchMooreKernel<TDATA, int32_t>(musa_stream, k_cache, v_cache, k, v, past_kv_lengths, _info); \
    }                                                                                            \
    return launchMooreKernel<TDATA, int64_t>(musa_stream, k_cache, v_cache, k, v, past_kv_lengths, _info)

    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        LAUNCH_WITH_IDX(half);
    case INFINI_DTYPE_BF16:
        LAUNCH_WITH_IDX(uint16_t);
    case INFINI_DTYPE_F32:
        LAUNCH_WITH_IDX(float);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#undef LAUNCH_WITH_IDX
}

} // namespace op::kv_caching::moore
