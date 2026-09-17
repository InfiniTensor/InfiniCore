#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../cuda/launch.cuh"
#include "mamba2_scan_nvidia.cuh"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace op::mamba2_scan::nvidia {
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t x_desc, infiniopTensorDescriptor_t dt_desc,
    infiniopTensorDescriptor_t b_desc, infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t a_desc, infiniopTensorDescriptor_t d_desc,
    infiniopTensorDescriptor_t dt_bias_desc, infiniopTensorDescriptor_t state_desc,
    infiniopTensorDescriptor_t offsets_desc, infiniopTensorDescriptor_t initial_indices_desc,
    infiniopTensorDescriptor_t final_indices_desc) {
    auto result = Mamba2ScanInfo::create(out_desc, x_desc, dt_desc, b_desc, c_desc, a_desc, d_desc, dt_bias_desc, state_desc, offsets_desc, initial_indices_desc, final_indices_desc);
    CHECK_RESULT(result);
    auto info = result.take();
    *desc_ptr = new Descriptor(info, info.workspace_bytes(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(void *workspace, size_t workspace_size, void *out,
                                     const void *x, const void *dt, const void *b, const void *c,
                                     const void *a, const void *d, const void *dt_bias, void *state,
                                     const void *offsets, const void *initial_indices,
                                     const void *final_indices, void *stream) const {
    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    if (_workspace_size && workspace == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }
    for (const void *pointer : {static_cast<const void *>(out), x, dt, b, c, a, d, dt_bias, static_cast<const void *>(state), offsets, initial_indices, final_indices}) {
        if (pointer == nullptr) {
            return INFINI_STATUS_NULL_POINTER;
        }
    }
#define LAUNCH(T) cuda::launch<T>(_info, workspace, out, x, dt, b, c, a, d, dt_bias, state, offsets, initial_indices, final_indices, static_cast<cudaStream_t>(stream))
    switch (_info.dtype) {
    case INFINI_DTYPE_F32:
        LAUNCH(float);
        break;
    case INFINI_DTYPE_F16:
        LAUNCH(half);
        break;
    case INFINI_DTYPE_BF16:
        LAUNCH(__nv_bfloat16);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
#undef LAUNCH
    CHECK_CUDA(cudaGetLastError());
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::mamba2_scan::nvidia
