#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"

#include "scatter_nvidia.cuh"
#include "../cuda/kernel.cuh"
#include "../info.h"

namespace op::scatter::nvidia {

//  ---------------------- start: launchKernel: call kernel function of CUDA -----------------------
template <unsigned int BLOCK_SIZE, typename Tdata>
INFINIOP_CUDA_KERNEL launchKernel(
    Tdata * output,
    const Tdata * input,
    const int64_t * index,
    size_t ndim,
    size_t index_scatter_size,
    ptrdiff_t * output_strides,
    ptrdiff_t * input_strides,
    ptrdiff_t * index_strides,
    ptrdiff_t * contiguous_strides,
    int scatter_dim
) {
    scatterKernel<BLOCK_SIZE, Tdata>(
        output,
        input,
        index,
        ndim,
        index_scatter_size,
        output_strides,
        input_strides,
        index_strides,
        contiguous_strides,
        scatter_dim
    );
}
//  ----------------------- end: launchKernel: call kernel function of CUDA ------------------------

//  ----------------------------------- start: call launchKernel -----------------------------------
template<unsigned int BLOCK_SIZE, typename Tdata>
infiniStatus_t calculate_scatter(
    const ScatterInfo &info,
    Tdata * output,
    const Tdata * input,
    const int64_t *  index,
    cudaStream_t stream,
    void * workspace
) {
    size_t ndim = info.ndim;
    ptrdiff_t * contiguous_strides = new ptrdiff_t[ndim];
    size_t last_dim = 1, last_stride = 1;
    size_t scatter_dim = info.dim;
    for(size_t d = 0; d < ndim; d ++)
    {
        if (d == scatter_dim) 
            continue;
        contiguous_strides[d] = last_dim * last_stride;
        last_dim = info.index_shape[d];
        last_stride = contiguous_strides[d];
    }

    size_t batch_size = last_dim * last_stride;

    ptrdiff_t * contiguous_strides_cuda = reinterpret_cast<ptrdiff_t*>(workspace);
    ptrdiff_t * input_strides_cuda = contiguous_strides_cuda + ndim;
    ptrdiff_t * output_strides_cuda = input_strides_cuda + ndim;
    ptrdiff_t * index_strides_cuda = output_strides_cuda + ndim;

    CHECK_CUDA(cudaMemcpyAsync(contiguous_strides_cuda, contiguous_strides, sizeof(ptrdiff_t) * ndim, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(input_strides_cuda, info.input_strides.data(), sizeof(ptrdiff_t) * ndim, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(output_strides_cuda, info.output_strides.data(), sizeof(ptrdiff_t) * ndim, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(index_strides_cuda, info.index_strides.data(), sizeof(ptrdiff_t) * ndim, cudaMemcpyHostToDevice, stream));    

    launchKernel<BLOCK_SIZE, Tdata><<<batch_size, BLOCK_SIZE, 0, stream>>>(
        output,
        input,
        index,
        ndim,
        info.index_shape[scatter_dim],
        output_strides_cuda,
        input_strides_cuda,
        index_strides_cuda,
        contiguous_strides_cuda,
        scatter_dim
    );
    delete[] contiguous_strides;
    return INFINI_STATUS_SUCCESS;
}
//  ------------------------------------ end: call launchKernel ------------------------------------


struct Descriptor::Opaque {
    std::shared_ptr<device::nvidia::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t index_desc,
    size_t dim
) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
//  --------------------- start: check data type and calculate workspace size ----------------------
    auto dtype = output_desc->dtype();
    size_t WorkSpaceSize = sizeof(ptrdiff_t) * input_desc->ndim() * 4;
//  ---------------------- end: check data type and calculate workspace size -----------------------
    auto result = ScatterInfo::createScatterInfo(
        output_desc,
        input_desc,
        index_desc,
        dim
    );
    CHECK_RESULT(result);
    const ScatterInfo &info = result.take();
    *desc_ptr = new Descriptor(
        dtype, std::move(info), WorkSpaceSize,
        new Opaque{handle->internal()},
        handle->device, handle->device_id
    );    
    return INFINI_STATUS_SUCCESS;
}


infiniStatus_t Descriptor::calculate(
    void * workspace,
    size_t workspace_size,
    void * output,
    const void * input,
    const void * index,
    void *stream_
) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    cudaStream_t stream = (cudaStream_t)stream_;
    #define CALCULATE_SCATTER(BLOCK_SIZE, TDATA) \
        calculate_scatter<BLOCK_SIZE, TDATA>(_info, (TDATA *)output, (const TDATA *)input, (const int64_t *)index, stream, workspace)
    #define CALCULATE_SCATTER_WITH_BLOCK_SIZE(BLOCK_SIZE)             \
    switch (_info.dtype) {                                            \
        case INFINI_DTYPE_BOOL:                                       \
            return CALCULATE_SCATTER(BLOCK_SIZE, bool);               \
        case INFINI_DTYPE_U8:                                         \
            return CALCULATE_SCATTER(BLOCK_SIZE, uint8_t);            \
        case INFINI_DTYPE_U16:                                        \
            return CALCULATE_SCATTER(BLOCK_SIZE, uint16_t);           \
        case INFINI_DTYPE_U32:                                        \
            return CALCULATE_SCATTER(BLOCK_SIZE, uint32_t);           \
        case INFINI_DTYPE_U64:                                        \
            return CALCULATE_SCATTER(BLOCK_SIZE, uint64_t);           \
        case INFINI_DTYPE_I8:                                         \
            return CALCULATE_SCATTER(BLOCK_SIZE, int8_t);             \
        case INFINI_DTYPE_I16:                                        \
            return CALCULATE_SCATTER(BLOCK_SIZE, int16_t);            \
        case INFINI_DTYPE_I32:                                        \
            return CALCULATE_SCATTER(BLOCK_SIZE, int32_t);            \
        case INFINI_DTYPE_I64:                                        \
            return CALCULATE_SCATTER(BLOCK_SIZE, int64_t);            \
        case INFINI_DTYPE_F16:                                        \
            return CALCULATE_SCATTER(BLOCK_SIZE, half);               \
        case INFINI_DTYPE_F32:                                        \
            return CALCULATE_SCATTER(BLOCK_SIZE, float);              \
        case INFINI_DTYPE_BF16:                                       \
            return CALCULATE_SCATTER(BLOCK_SIZE, cuda_bfloat16);      \
        default:                                                      \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;                    \
    }    
        
    
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024)
        CALCULATE_SCATTER_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_1024)
    else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512)
        CALCULATE_SCATTER_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_512)
    else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096)
        CALCULATE_SCATTER_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_4096)
    else
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;
    
    #undef CALCULATE_SCATTER_WITH_BLOCK_SIZE
    #undef CALCULATE_SCATTER

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::scatter::nvidia
