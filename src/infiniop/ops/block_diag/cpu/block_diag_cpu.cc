#include "block_diag_cpu.h"
#include "../../../utils.h"
#include <cstring>

namespace op::block_diag::cpu {

utils::Result<BlockDiagInfo> BlockDiagInfo::create(
    infiniopTensorDescriptor_t *input_descs,
    size_t num_inputs,
    infiniopTensorDescriptor_t y_desc) {

    if (num_inputs == 0) {
        return INFINI_STATUS_BAD_PARAM;
    }

    BlockDiagInfo info;
    info.num_inputs = num_inputs;
    info.input_shapes.resize(num_inputs);
    info.row_offsets.resize(num_inputs);
    info.col_offsets.resize(num_inputs);

    size_t total_rows = 0;
    size_t total_cols = 0;

    // Process each input matrix
    for (size_t i = 0; i < num_inputs; ++i) {
        auto shape = input_descs[i]->shape();
        if (shape.size() != 2) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        info.input_shapes[i] = shape;
        info.row_offsets[i] = total_rows;
        info.col_offsets[i] = total_cols;
        total_rows += shape[0];
        total_cols += shape[1];
    }

    // Check output shape
    auto y_shape = y_desc->shape();
    if (y_shape.size() != 2 || y_shape[0] != total_rows || y_shape[1] != total_cols) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    info.output_shape = y_shape;
    info.output_size = y_desc->numel();

    return utils::Result<BlockDiagInfo>(std::move(info));
}

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t *input_descs,
    size_t num_inputs) {

    if (num_inputs == 0) {
        return INFINI_STATUS_BAD_PARAM;
    }

    auto dtype = input_descs[0]->dtype();
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    // Check all inputs have same dtype
    for (size_t i = 1; i < num_inputs; ++i) {
        if (input_descs[i]->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    }

    auto info_result = BlockDiagInfo::create(input_descs, num_inputs, y_desc);
    CHECK_RESULT(info_result);

    *desc_ptr = new Descriptor(dtype, info_result.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

template <typename T>
void block_diag_impl(
    const BlockDiagInfo &info,
    T *y,
    const T **inputs) {

    // Initialize output to zero
    std::memset(y, 0, info.output_size * sizeof(T));

    // Place each input matrix at its diagonal position
    for (size_t i = 0; i < info.num_inputs; ++i) {
        size_t rows = info.input_shapes[i][0];
        size_t cols = info.input_shapes[i][1];
        size_t row_offset = info.row_offsets[i];
        size_t col_offset = info.col_offsets[i];
        const T *input = reinterpret_cast<const T *>(inputs[i]);

        // Copy input matrix to output at diagonal position
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                size_t out_row = row_offset + r;
                size_t out_col = col_offset + c;
                size_t out_idx = out_row * info.output_shape[1] + out_col;
                size_t in_idx = r * cols + c;
                y[out_idx] = input[in_idx];
            }
        }
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *y,
    const void **inputs,
    void *stream) const {

    switch (_dtype) {
    case INFINI_DTYPE_F16: {
        const fp16_t **typed_inputs = reinterpret_cast<const fp16_t **>(inputs);
        block_diag_impl<fp16_t>(_info, reinterpret_cast<fp16_t *>(y), typed_inputs);
        break;
    }
    case INFINI_DTYPE_BF16: {
        const bf16_t **typed_inputs = reinterpret_cast<const bf16_t **>(inputs);
        block_diag_impl<bf16_t>(_info, reinterpret_cast<bf16_t *>(y), typed_inputs);
        break;
    }
    case INFINI_DTYPE_F32: {
        const float **typed_inputs = reinterpret_cast<const float **>(inputs);
        block_diag_impl<float>(_info, reinterpret_cast<float *>(y), typed_inputs);
        break;
    }
    case INFINI_DTYPE_F64: {
        const double **typed_inputs = reinterpret_cast<const double **>(inputs);
        block_diag_impl<double>(_info, reinterpret_cast<double *>(y), typed_inputs);
        break;
    }
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::block_diag::cpu
