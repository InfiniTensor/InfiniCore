#include "../../../devices/nvidia/nvidia_handle.cuh"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "triu_nvidia.cuh"
#include "../cuda/kernel.cuh"
#include "../info.h"

namespace op::triu::nvidia {

template<unsigned int BLOCK_SIZE, typename Tdata>
INFINIOP_CUDA_KERNEL launchKernel(
    Tdata * output,
    const Tdata * input,
    int column_size,
    int diagonal
) {
    triuKernel<BLOCK_SIZE, Tdata>(
        output,
        input,
        column_size,
        diagonal
    );
}


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
    int diagonal
) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
//  --------------------- start: check data type and calculate workspace size ----------------------
    auto dtype = output_desc->dtype();
    size_t WorkSpaceSize = 0;
//  ---------------------- end: check data type and calculate workspace size -----------------------
    auto result = TriuInfo::createTriuInfo(
        output_desc,
        input_desc,
        diagonal
    );
    CHECK_RESULT(result);
    const TriuInfo &info = result.take();
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

    #define CALCULATE_TRIU(TDATA)                                     \
    launchKernel<256, TDATA><<<_info.row_size, 256, 0, stream>>>(     \
        (TDATA *)output,                                              \
        (const TDATA *)input,                                         \
        _info.column_size,                                            \
        _info.diagonal                                                \
    )
switch (_info.dtype) {
        case INFINI_DTYPE_BOOL:
            CALCULATE_TRIU(bool);
            break;
        case INFINI_DTYPE_U8:
            CALCULATE_TRIU(uint8_t);
            break;
        case INFINI_DTYPE_U16:
            CALCULATE_TRIU(uint16_t);
            break;
        case INFINI_DTYPE_U32:
            CALCULATE_TRIU(uint32_t);
            break;
        case INFINI_DTYPE_U64:
            CALCULATE_TRIU(uint64_t);
            break;
        case INFINI_DTYPE_I8:
            CALCULATE_TRIU(int8_t);
            break;
        case INFINI_DTYPE_I16:
            CALCULATE_TRIU(int16_t);
            break;
        case INFINI_DTYPE_I32:
            CALCULATE_TRIU(int32_t);
            break;
        case INFINI_DTYPE_I64:
            CALCULATE_TRIU(int64_t);
            break;
        case INFINI_DTYPE_F16:
            CALCULATE_TRIU(half);
            break;
        case INFINI_DTYPE_F32:
            CALCULATE_TRIU(float);
            break;
        case INFINI_DTYPE_BF16:
            CALCULATE_TRIU(cuda_bfloat16);
            break;
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::triu::nvidia
