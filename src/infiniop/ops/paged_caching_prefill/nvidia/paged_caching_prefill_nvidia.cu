#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/kernel.cuh"
#include "paged_caching_prefill_nvidia.cuh"

// Global Wrapper
template <typename Tdata, int NUM_THREADS>
__global__ void pagedCachingPrefill(
    Tdata *k_cache, Tdata *v_cache,
    const Tdata *k, const Tdata *v,
    const int32_t *slot_mapping,
    const size_t head_size, const size_t block_size,
    const ptrdiff_t k_src_stride, const ptrdiff_t v_src_stride,
    const ptrdiff_t k_cache_block_stride, const ptrdiff_t v_cache_block_stride) {
    op::paged_caching_prefill::cuda::pagedCachingPrefillKernel<Tdata, NUM_THREADS>(
        k_cache, v_cache, k, v, slot_mapping, head_size,
        block_size, k_src_stride, v_src_stride, k_cache_block_stride, v_cache_block_stride);
}

namespace op::paged_caching_prefill::nvidia {

struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t slot_mapping_desc) {

    auto info = PagedCachingPrefillInfo::create(k_desc, v_desc, k_cache_desc, v_cache_desc, slot_mapping_desc);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <int NUM_THREADS>
static void launch(const PagedCachingPrefillInfo &info,
                   void *k_cache, void *v_cache,
                   const void *k, const void *v,
                   const void *slot_mapping,
                   cudaStream_t stream) {
    dim3 grid(info.num_kv_heads, info.num_tokens);
    dim3 block(NUM_THREADS);

    if (info.dtype == INFINI_DTYPE_F16) {
        pagedCachingPrefill<half, NUM_THREADS><<<grid, block, 0, stream>>>(
            (half *)k_cache, (half *)v_cache, (const half *)k, (const half *)v, (const int32_t *)slot_mapping,
            info.head_size, info.block_size, info.k_src_stride, info.v_src_stride,
            info.k_cache_block_stride, info.v_cache_block_stride);
    } else if (info.dtype == INFINI_DTYPE_BF16) {
        pagedCachingPrefill<__nv_bfloat16, NUM_THREADS><<<grid, block, 0, stream>>>(
            (__nv_bfloat16 *)k_cache, (__nv_bfloat16 *)v_cache, (const __nv_bfloat16 *)k, (const __nv_bfloat16 *)v, (const int32_t *)slot_mapping,
            info.head_size, info.block_size, info.k_src_stride, info.v_src_stride,
            info.k_cache_block_stride, info.v_cache_block_stride);
    } else if (info.dtype == INFINI_DTYPE_F32) {
        pagedCachingPrefill<float, NUM_THREADS><<<grid, block, 0, stream>>>(
            (float *)k_cache, (float *)v_cache, (const float *)k, (const float *)v, (const int32_t *)slot_mapping,
            info.head_size, info.block_size, info.k_src_stride, info.v_src_stride,
            info.k_cache_block_stride, info.v_cache_block_stride);
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    const void *k, const void *v,
    void *k_cache, void *v_cache,
    const void *slot_mapping,
    void *stream_) const {

    cudaStream_t stream = (cudaStream_t)stream_;
    uint32_t max_threads = _opaque->internal->maxThreadsPerBlock();

    if (max_threads >= 1024) {
        launch<1024>(_info, k_cache, v_cache, k, v, slot_mapping, stream);
    } else if (max_threads >= 512) {
        launch<512>(_info, k_cache, v_cache, k, v, slot_mapping, stream);
    } else {
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::paged_caching_prefill::nvidia
