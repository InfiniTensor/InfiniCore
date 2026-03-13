#include "../info.h"
#include "../cuda/kernel.cuh"
#include "eye_nvidia.cuh"
#include "../../../../utils/result.hpp"
#include "../../../devices/nvidia/nvidia_common.cuh"

namespace op::eye::nvidia {

infiniStatus_t Descriptor::create(infiniopHandle_t handle_,
                                  Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t y_desc) {
    (void)handle_;
    auto info_result = EyeInfo::create(y_desc);
    CHECK_RESULT(info_result);
    *desc_ptr = new Descriptor(info_result.take(), INFINI_DEVICE_NVIDIA, 0);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace,
                                    size_t workspace_size,
                                    void *y,
                                    void *stream) const {
    (void)workspace;
    (void)workspace_size;

    size_t n = _info.n;
    size_t m = _info.m;
    size_t total = n * m;

    if (total == 0) {
        return INFINI_STATUS_SUCCESS;
    }

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    constexpr uint32_t block_size = 256;
    uint32_t grid_size = static_cast<uint32_t>((total + block_size - 1) / block_size);

    switch (_info.dtype) {
    case INFINI_DTYPE_F16:
        cuda::eyeKernel<half><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<half *>(y), n, m);
        break;
    case INFINI_DTYPE_F32:
        cuda::eyeKernel<float><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<float *>(y), n, m);
        break;
    case INFINI_DTYPE_F64:
        cuda::eyeKernel<double><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<double *>(y), n, m);
        break;
    case INFINI_DTYPE_BF16:
        cuda::eyeKernel<cuda_bfloat16><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<cuda_bfloat16 *>(y), n, m);
        break;
    case INFINI_DTYPE_I32:
        cuda::eyeKernel<int32_t><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<int32_t *>(y), n, m);
        break;
    case INFINI_DTYPE_I64:
        cuda::eyeKernel<int64_t><<<grid_size, block_size, 0, cuda_stream>>>(
            static_cast<int64_t *>(y), n, m);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return cudaGetLastError() == cudaSuccess ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
}

} // namespace op::eye::nvidia
