#include "../../../devices/nvidia/nvidia_common.cuh"
#include "kv_caching_nvidia.cuh"

#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include <cub/block/block_reduce.cuh>

#include "../../../reduce/cuda/reduce.cuh"

#include "../cuda/kernel.cuh"

template <typename Tdata>
INFINIOP_CUDA_KERNEL kvCaching(
    Tdata *k_cache,
    Tdata *v_cache,
    const Tdata *k,
    const Tdata *v,
    const int64_t *past_kv_lengths,
    int batch_size,
    int num_kv_heads,
    int max_seq_len,
    int seq_len,
    int hidden_dim,
    ptrdiff_t cache_strides_0,
    ptrdiff_t cache_strides_1,
    ptrdiff_t cache_strides_2,
    ptrdiff_t cache_strides_3) {
    kvCachingKernel<Tdata>(k_cache, v_cache, k, v, past_kv_lengths,
                           batch_size, num_kv_heads, max_seq_len, seq_len, hidden_dim,
                           cache_strides_0, cache_strides_1, cache_strides_2, cache_strides_3);
}

namespace op::kv_caching::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

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
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <unsigned int BLOCK_SIZE, typename Tdata>
infiniStatus_t launchKernel(const KVCachingInfo &info,
                            Tdata *k_cache,
                            Tdata *v_cache,
                            const Tdata *k,
                            const Tdata *v,
                            const int64_t *past_kv_lengths,
                            cudaStream_t stream, void *workspace) {

    int batch_size = static_cast<int>(info.batch_size);
    int num_kv_heads = static_cast<int>(info.num_kv_heads);
    int max_seq_len = static_cast<int>(info.max_seq_len);
    int hidden_dim = static_cast<int>(info.hidden_dim);

    int seq_len = static_cast<int>(info.seq_len);

    int total = batch_size * num_kv_heads * seq_len * hidden_dim;

    ptrdiff_t cache_strides_0 = info.cache_strides_0;
    ptrdiff_t cache_strides_1 = info.cache_strides_1;
    ptrdiff_t cache_strides_2 = info.cache_strides_2;
    ptrdiff_t cache_strides_3 = info.cache_strides_3;

    int num_blocks = min((total + BLOCK_SIZE - 1) / BLOCK_SIZE, 1024);

    kvCaching<Tdata>
        <<<num_blocks, BLOCK_SIZE, 0, stream>>>(k_cache, v_cache, k, v, past_kv_lengths,
                                                batch_size, num_kv_heads, max_seq_len, seq_len, hidden_dim,
                                                cache_strides_0, cache_strides_1, cache_strides_2, cache_strides_3);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size,
                                     void *k_cache,
                                     void *v_cache,
                                     const void *k,
                                     const void *v,
                                     const void *past_kv_lengths,
                                     void *stream_) const {
    cudaStream_t stream = (cudaStream_t)stream_;
#define CALCULATE_KV_CACHING(BLOCK_SIZE, TDATA) \
    launchKernel<BLOCK_SIZE, TDATA>(_info, (TDATA *)k_cache, (TDATA *)v_cache, (const TDATA *)k, (const TDATA *)v, (const int64_t *)past_kv_lengths, stream, workspace)
#define CALCULATE_KV_CACHING_WITH_BLOCK_SIZE(BLOCK_SIZE)            \
    {                                                               \
        if (_info.dtype == INFINI_DTYPE_F16)                        \
            return CALCULATE_KV_CACHING(BLOCK_SIZE, half);          \
        else if (_info.dtype == INFINI_DTYPE_F32)                   \
            return CALCULATE_KV_CACHING(BLOCK_SIZE, float);         \
        else if (_info.dtype == INFINI_DTYPE_BF16)                  \
            return CALCULATE_KV_CACHING(BLOCK_SIZE, __nv_bfloat16); \
        else                                                        \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;                  \
    }
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024) {
        CALCULATE_KV_CACHING_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_1024)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512) {
        CALCULATE_KV_CACHING_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_512)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_2048) {
        CALCULATE_KV_CACHING_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_2048)
    } else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096) {
        CALCULATE_KV_CACHING_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_4096)
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::kv_caching::nvidia
