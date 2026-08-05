#include "select_last_token_hidden_cpu.h"
#include "../../../../utils.h"
#include "../../../handle.h"
#include "../../../tensor.h"
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace op::select_last_token_hidden::cpu {

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
    (void)stream;
    auto *out = reinterpret_cast<std::byte *>(output);
    const auto *hidden = reinterpret_cast<const std::byte *>(hidden_states);
    const auto *offsets = reinterpret_cast<const int32_t *>(input_offsets);
    for (size_t request = 0; request < _num_requests; ++request) {
        const int32_t row = offsets[request + 1] - 1;
        CHECK_OR_RETURN(row >= 0 && static_cast<size_t>(row) < _total_tokens,
                        INFINI_STATUS_BAD_PARAM);
        std::memcpy(
            out + request * _row_bytes,
            hidden + static_cast<size_t>(row) * _row_bytes,
            _row_bytes);
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::select_last_token_hidden::cpu
