#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "reduce_mean_nvidia.cuh"
#include "../cuda/kernel.cuh"
#include "../info.h"

namespace op::reduce_mean::nvidia {

//  ---------------------- start: launchKernel: call kernel function of CUDA -----------------------
template <typename Tdata, typename Tcompute>
INFINIOP_CUDA_KERNEL launchKernel(
    Tdata * output,
    const Tdata * input,
    int ndim,
    const size_t * input_shape,
    const ptrdiff_t * output_strides,
    const ptrdiff_t * input_strides,
    const ptrdiff_t * contiguous_strides,
    size_t dim
) {
    reduceMeanKernel<Tdata, Tcompute>(
        output,
        input,
        ndim,
        input_shape,
        output_strides,
        input_strides,
        contiguous_strides,
        dim
    );
}
//  ----------------------- end: launchKernel: call kernel function of CUDA ------------------------

//  ----------------------------------- start: call launchKernel -----------------------------------
template<unsigned int BLOCK_SIZE, typename Tdata>
infiniStatus_t calculate_reduce_mean(
    const ReduceMeanInfo &info,
    Tdata * output,
    const Tdata * input,
    cudaStream_t stream,
    void * workspace
) {
    size_t ndim = info.ndim;
    ptrdiff_t * contiguous_strides = new ptrdiff_t[ndim];

    size_t last_dim = 1, last_stride = 1;
    for(size_t d = 0; d < ndim; d ++)
    {
        if (d == info.dim)
            continue;
        contiguous_strides[d] = last_dim * last_stride;  
        last_dim = info.input_shape[d];
        last_stride = contiguous_strides[d];
    }
    size_t batch_size = last_dim * last_stride;


    ptrdiff_t * contiguous_strides_cuda = reinterpret_cast<ptrdiff_t*>(workspace);
    ptrdiff_t * input_strides_cuda = reinterpret_cast<ptrdiff_t*>(workspace) + ndim;
    ptrdiff_t * output_strides_cuda = reinterpret_cast<ptrdiff_t*>(workspace) + ndim * 2;
    size_t * input_shape_cuda = reinterpret_cast<size_t*>(workspace) + ndim * 3 * sizeof(ptrdiff_t) / sizeof(size_t);

    CHECK_CUDA(cudaMemcpyAsync(contiguous_strides_cuda, contiguous_strides, sizeof(ptrdiff_t) * ndim, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(input_strides_cuda, info.input_strides.data(), sizeof(ptrdiff_t) * ndim, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(output_strides_cuda, info.output_strides.data(), sizeof(ptrdiff_t) * ndim, cudaMemcpyHostToDevice, stream));
    CHECK_CUDA(cudaMemcpyAsync(input_shape_cuda, info.input_shape.data(), sizeof(size_t) * ndim, cudaMemcpyHostToDevice, stream));    

    launchKernel<Tdata, float><<<batch_size, 1, 0, stream>>>(
        output,
        input,
        int(info.ndim),
        input_shape_cuda,
        output_strides_cuda,
        input_strides_cuda,
        contiguous_strides_cuda,
        info.dim
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
    size_t dim
) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
//  --------------------- start: check data type and calculate workspace size ----------------------
    auto dtype = output_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    size_t WorkSpaceSize = sizeof(ptrdiff_t) * input_desc->ndim() * 4;
//  ---------------------- end: check data type and calculate workspace size -----------------------
    auto result = ReduceMeanInfo::createReduceMeanInfo(
        output_desc,
        input_desc,
        dim
    );
    CHECK_RESULT(result);
    const ReduceMeanInfo &info = result.take();
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
    void *stream_
) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    cudaStream_t stream = (cudaStream_t)stream_;

    #define CALCULATE_REDUCE_MEAN(BLOCK_SIZE, TDATA) \
        calculate_reduce_mean<BLOCK_SIZE, TDATA>(_info, (TDATA *)output, (const TDATA *)input, stream, workspace)
    #define CALCULATE_REDUCE_MEAN_WITH_BLOCK_SIZE(BLOCK_SIZE)         \
    {                                                                 \
        if (_info.dtype == INFINI_DTYPE_F16)                          \
            return CALCULATE_REDUCE_MEAN(BLOCK_SIZE, half);           \
        else if (_info.dtype == INFINI_DTYPE_F32)                     \
            return CALCULATE_REDUCE_MEAN(BLOCK_SIZE, float);          \
        else if (_info.dtype == INFINI_DTYPE_BF16)                    \
            return CALCULATE_REDUCE_MEAN(BLOCK_SIZE, __nv_bfloat16);  \
        else                                                          \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;                    \
    }
    
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024)
        CALCULATE_REDUCE_MEAN_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_1024)
    else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512)
        CALCULATE_REDUCE_MEAN_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_512)
    else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096)
        CALCULATE_REDUCE_MEAN_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_4096)
    else
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::reduce_mean::nvidia
