#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "linear_nvidia.cuh"
#include "../cuda/kernel.cuh"
#include "../info.h"

namespace op::linear::nvidia {

//  ---------------------- start: launchKernel: call kernel function of CUDA -----------------------
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
INFINIOP_CUDA_KERNEL launchKernel(
    Tdata * y,
    const Tdata * x,
    const Tdata * w,
    const Tdata * b,
	size_t in_features,
	size_t out_features,
	ptrdiff_t y_stride,
	ptrdiff_t x_stride,
	ptrdiff_t w_stride_out,
    ptrdiff_t w_stride_in,
	ptrdiff_t b_stride,
    bool bias
) {
    linearKernel<BLOCK_SIZE, Tdata, Tcompute>(
		y,
		x,
		w,
		b,
        in_features,
        out_features,
        y_stride,
        x_stride,
        w_stride_out,
        w_stride_in,
        b_stride,
        bias
    );
}
//  ----------------------- end: launchKernel: call kernel function of CUDA ------------------------

//  ----------------------------------- start: call launchKernel -----------------------------------
template<unsigned int BLOCK_SIZE, typename Tdata>
infiniStatus_t calculate_linear(
    const LinearInfo &info,
    Tdata * y,
    const Tdata * x,
    const Tdata * w,
    const Tdata * b,
    cudaStream_t stream
) {

    launchKernel<1, Tdata, float><<<info.out_features, 1, 0, stream>>>(
        y,
        x,
        w,
        b,
        info.in_features,
        info.out_features,
        info.y_stride,
        info.x_stride,
        info.w_stride_out,
        info.w_stride_in,
        info.b_stride,
        info.bias        
    );
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
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t w_desc,
    infiniopTensorDescriptor_t b_desc
) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
//  --------------------- start: check data type and calculate workspace size ----------------------
    auto dtype = y_desc->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    size_t WorkSpaceSize = 0;
    //size_t workspace_size = reinterpret_cast<op::linear::nvidia::Descriptor *>(y_desc)->workspaceSize();
//  ---------------------- end: check data type and calculate workspace size -----------------------
    auto result = LinearInfo::createLinearInfo(
        y_desc,
        x_desc,
        w_desc,
        b_desc
    );
    CHECK_RESULT(result);
    const LinearInfo &info = result.take();
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
    void * y,
    const void * x,
    const void * w,
    const void * b,
    void *stream_
) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    cudaStream_t stream = (cudaStream_t)stream_;

    #define CALCULATE_LINEAR(BLOCK_SIZE, TDATA) \
        calculate_linear<BLOCK_SIZE, TDATA>(_info, (TDATA *)y, (const TDATA *)x, (const TDATA *)w, (const TDATA *)b, stream)
    #define CALCULATE_LINEAR_WITH_BLOCK_SIZE(BLOCK_SIZE)              \
    {                                                                 \
        if (_info.dtype == INFINI_DTYPE_F16)                          \
            return CALCULATE_LINEAR(BLOCK_SIZE, half);                \
        else if (_info.dtype == INFINI_DTYPE_F32)                     \
            return CALCULATE_LINEAR(BLOCK_SIZE, float);               \
        else if (_info.dtype == INFINI_DTYPE_BF16)                    \
            return CALCULATE_LINEAR(BLOCK_SIZE, __nv_bfloat16);       \
        else                                                          \
            return INFINI_STATUS_BAD_TENSOR_DTYPE;                    \
    }
    
    if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_1024)
        CALCULATE_LINEAR_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_1024)
    else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_512)
        CALCULATE_LINEAR_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_512)
    else if (_opaque->internal->maxThreadsPerBlock() == CUDA_BLOCK_SIZE_4096)
        CALCULATE_LINEAR_WITH_BLOCK_SIZE(CUDA_BLOCK_SIZE_4096)
    else
        return INFINI_STATUS_DEVICE_ARCHITECTURE_NOT_SUPPORTED;

    return INFINI_STATUS_SUCCESS;

    #undef CALCULATE_LINEAR_WITH_BLOCK_SIZE
    #undef CALCULATE_LINEAR
}
} // namespace op::linear::nvidia
