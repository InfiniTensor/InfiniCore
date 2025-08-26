#include "../../../devices/nvidia/nvidia_common.cuh"
#include "paged_caching_nvidia.cuh"
#include "../cuda/kernel.cuh"

// We assume some common headers from your library are available.
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

template <typename Tdata, int NUM_THREADS>
INFINIOP_CUDA_KERNEL pagedCaching(
    Tdata *k_cache, Tdata *v_cache,
    const Tdata *k, const Tdata *v,
    const int *slot_mapping,
    const int num_heads, const int head_size, const int block_size,
    const ptrdiff_t k_src_stride, const ptrdiff_t v_src_stride,
    const ptrdiff_t k_cache_block_stride, const ptrdiff_t v_cache_block_stride
    ) {
    op::paged_caching::cuda::pagedCachingKernel<Tdata, NUM_THREADS>(
        k_cache, v_cache, k, v, slot_mapping, num_heads, head_size, block_size, k_src_stride, v_src_stride, k_cache_block_stride, v_cache_block_stride);
}

namespace op::paged_caching::nvidia {
// PIMPL struct definition
struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

// Destructor implementation
Descriptor::~Descriptor() {
    delete _opaque;
}

// Static factory method implementation
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t slot_mapping_desc) {
    
    // Use the Info struct's factory to parse and validate tensor metadata.
    // NOTE: The implementation of PagedCachingInfo::create is omitted for brevity,
    // but it would extract shapes, dtypes, and strides from the descriptors.
    auto info = PagedCachingInfo::create(k_desc, v_desc, k_cache_desc, v_cache_desc, slot_mapping_desc);
    CHECK_RESULT(info);

    // Create and return the Descriptor instance.
    *desc_ptr = new Descriptor(
        new Opaque{reinterpret_cast<device::nvidia::Handle *>(handle)->internal()},
        info.take(), 0, handle->device, handle->device_id);
    
    return INFINI_STATUS_SUCCESS;
}


// The launchKernel function is a templated helper to encapsulate the CUDA kernel launch.
// It sets up grid/block dimensions and calls the device-side kernel.
template <typename Tdata, int NUM_THREADS>
void launchKernel(const PagedCachingInfo& info,
                    void *k_cache, void *v_cache,
                    const void *k, const void *v,
                    const void *slot_mapping,
                    cudaStream_t stream) {
    
    // Grid dimension is 1D, with one block per token, as we decided.
    dim3 grid(uint32_t(info.num_tokens), uint32_t(info.num_heads), 1);
    // Block dimension is 1D, using the number of threads specified at compile time.
    dim3 block(NUM_THREADS);

    // This kernel does not require dynamic shared memory.
    size_t shared_mem_size = 0;

    // Launch the device-side CUDA kernel.
    pagedCaching<Tdata, NUM_THREADS>
        <<<grid, block, shared_mem_size, stream>>>(
            (Tdata*)k_cache,
            (Tdata*)v_cache,
            (const Tdata*)k,
            (const Tdata*)v,
            (const int*)slot_mapping,
            info.num_heads,
            info.head_size,
            info.block_size,
            info.k_src_stride,
            info.v_src_stride,
            info.k_cache_block_stride,
            info.v_cache_block_stride
        );
}


// Execution method implementation
infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    const void *k, const void *v,
    void *k_cache, void *v_cache,
    const void *slot_mapping,
    void *stream_) const {
    
    cudaStream_t stream = (cudaStream_t)stream_;

    // Dispatch logic based on the GPU's maximum threads per block.
    // This allows selecting the largest, most efficient block size the hardware supports.
    if (_opaque->internal->maxThreadsPerBlock() >= CUDA_BLOCK_SIZE_1024) {
        // Dispatch based on data type for a 1024-thread block.
        if (_info.dtype == INFINI_DTYPE_F16) {
            launchKernel<half, CUDA_BLOCK_SIZE_1024>(
                _info, k_cache, v_cache, k, v, slot_mapping, stream);
        } else if (_info.dtype == INFINI_DTYPE_BF16) {
            launchKernel<__nv_bfloat16, CUDA_BLOCK_SIZE_1024>(
                _info, k_cache, v_cache, k, v, slot_mapping, stream);
        } else if (_info.dtype == INFINI_DTYPE_F32) {
            launchKernel<float, CUDA_BLOCK_SIZE_1024>(
                _info, k_cache, v_cache, k, v, slot_mapping, stream);
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE; 
        }
    } else if (_opaque->internal->maxThreadsPerBlock() >= CUDA_BLOCK_SIZE_512) {
        // Dispatch based on data type for a 512-thread block.
        if (_info.dtype == INFINI_DTYPE_F16) {
            launchKernel<half, CUDA_BLOCK_SIZE_512>(
                _info, k_cache, v_cache, k, v, slot_mapping, stream);
        } else if (_info.dtype == INFINI_DTYPE_BF16) {
            launchKernel<__nv_bfloat16, CUDA_BLOCK_SIZE_512>(
                _info, k_cache, v_cache, k, v, slot_mapping, stream);
        } else if (_info.dtype == INFINI_DTYPE_F32) {
            launchKernel<float, CUDA_BLOCK_SIZE_512>(
                _info, k_cache, v_cache, k, v, slot_mapping, stream);
        } else {
            return INFINI_STATUS_BAD_TENSOR_DTYPE; 
        }
    } else {
        // If the GPU is older and supports fewer threads, return an error.
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    }
    
    // Check for any asynchronous errors launched by the kernel.
    // return CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::paged_caching::nvidia