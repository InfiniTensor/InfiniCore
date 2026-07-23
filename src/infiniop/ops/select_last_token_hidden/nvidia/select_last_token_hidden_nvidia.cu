#include "../../../../utils.h"
#include "../../../devices/nvidia/nvidia_common.cuh"
#include "../../../devices/nvidia/nvidia_kernel_common.cuh"
#include "../../../tensor.h"
#include "select_last_token_hidden_nvidia.cuh"
#include <cstdint>
#include <cuda_runtime.h>

template <typename CopyT>
INFINIOP_CUDA_KERNEL selectLastTokenHiddenKernel(
    CopyT *__restrict__ output,
    const CopyT *__restrict__ hidden_states,
    const int32_t *__restrict__ input_offsets,
    size_t row_width,
    size_t total_tokens) {
    const size_t request = blockIdx.x;
    const int32_t row = input_offsets[request + 1] - 1;
    if (row < 0 || static_cast<size_t>(row) >= total_tokens) {
        return;
    }
    const CopyT *src = hidden_states + static_cast<size_t>(row) * row_width;
    CopyT *dst = output + request * row_width;
    for (size_t column = threadIdx.x; column < row_width; column += blockDim.x) {
        dst[column] = src[column];
    }
}

namespace op::select_last_token_hidden::nvidia {

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t hidden_states_desc,
    infiniopTensorDescriptor_t input_offsets_desc) {
    const auto output_shape = output_desc->shape();
    const auto hidden_shape = hidden_states_desc->shape();
    const auto offsets_shape = input_offsets_desc->shape();

    CHECK_OR_RETURN(output_shape.size() == 3 && hidden_shape.size() == 3 && offsets_shape.size() == 1,
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(offsets_shape[0] >= 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
    const size_t num_requests = offsets_shape[0] - 1;
    CHECK_OR_RETURN(output_shape[0] == 1 && output_shape[1] == num_requests
                        && output_shape[2] == hidden_shape[2],
                    INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(output_desc->isContiguous() && hidden_states_desc->isContiguous()
                        && input_offsets_desc->isContiguous(),
                    INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(input_offsets_desc->dtype() == INFINI_DTYPE_I32,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    const auto hidden_dtype = hidden_states_desc->dtype();
    CHECK_OR_RETURN(hidden_dtype == INFINI_DTYPE_F16 || hidden_dtype == INFINI_DTYPE_BF16
                        || hidden_dtype == INFINI_DTYPE_F32,
                    INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(output_desc->dtype() == hidden_dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);

    const size_t total_tokens = hidden_shape[0] * hidden_shape[1];
    CHECK_OR_RETURN(total_tokens > 0 && hidden_shape[2] > 0, INFINI_STATUS_BAD_TENSOR_SHAPE);
    *desc_ptr = new Descriptor(
        num_requests,
        total_tokens,
        hidden_shape[2] * infiniSizeOf(hidden_dtype),
        handle->device,
        handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *output,
    const void *hidden_states,
    const void *input_offsets,
    void *stream) const {
    if (_num_requests == 0 || _row_bytes == 0) {
        return INFINI_STATUS_SUCCESS;
    }
    constexpr size_t block_size = 256;
    const auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (_row_bytes % sizeof(uint4) == 0) {
        selectLastTokenHiddenKernel<uint4><<<_num_requests, block_size, 0, cuda_stream>>>(
            reinterpret_cast<uint4 *>(output),
            reinterpret_cast<const uint4 *>(hidden_states),
            reinterpret_cast<const int32_t *>(input_offsets),
            _row_bytes / sizeof(uint4),
            _total_tokens);
    } else {
        selectLastTokenHiddenKernel<uint8_t><<<_num_requests, block_size, 0, cuda_stream>>>(
            reinterpret_cast<uint8_t *>(output),
            reinterpret_cast<const uint8_t *>(hidden_states),
            reinterpret_cast<const int32_t *>(input_offsets),
            _row_bytes,
            _total_tokens);
    }
    return cudaGetLastError() == cudaSuccess ? INFINI_STATUS_SUCCESS : INFINI_STATUS_INTERNAL_ERROR;
}

} // namespace op::select_last_token_hidden::nvidia
