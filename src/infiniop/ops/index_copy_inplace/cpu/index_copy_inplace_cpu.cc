#include "index_copy_inplace_cpu.h"
#include "../../../../utils.h"
#include "../../../tensor.h"
#include <cstring>

namespace op::index_copy_inplace::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t input_desc,
    infiniopTensorDescriptor_t output_desc,
    int dim,
    infiniopTensorDescriptor_t index_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = input_desc->dtype();

    // Check data types - 支持所有合法类型
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16,
                INFINI_DTYPE_I8, INFINI_DTYPE_I16, INFINI_DTYPE_I32, INFINI_DTYPE_I64,
                INFINI_DTYPE_U8, INFINI_DTYPE_U16, INFINI_DTYPE_U32, INFINI_DTYPE_U64,
                INFINI_DTYPE_BOOL);

    // Check that input and output have same dtype
    if (input_desc->dtype() != output_desc->dtype()) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    // Check that index is integer type
    auto index_dtype = index_desc->dtype();
    if (index_dtype != INFINI_DTYPE_I32 && index_dtype != INFINI_DTYPE_I64) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    // Check dimension bounds
    auto input_shape = input_desc->shape();
    auto output_shape = output_desc->shape();
    if (dim < 0 || dim >= static_cast<int>(input_shape.size())) {
        return INFINI_STATUS_BAD_PARAM;
    }
    if (dim < 0 || dim >= static_cast<int>(output_shape.size())) {
        return INFINI_STATUS_BAD_PARAM;
    }

    // Check that input and output have same shape except possibly at dim
    if (input_shape.size() != output_shape.size()) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (i != static_cast<size_t>(dim) && input_shape[i] != output_shape[i]) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
    }

    auto desc = new Descriptor();
    desc->_input_desc = input_desc;
    desc->_output_desc = output_desc;
    desc->_index_desc = index_desc;
    desc->_dim = dim;
    desc->_handle = handle;
    desc->device_type = INFINI_DEVICE_CPU;
    desc->device_id = handle->device_id;

    *desc_ptr = desc;
    return INFINI_STATUS_SUCCESS;
}

template<typename T, typename IndexT>
static void index_copy_inplace_kernel(
    const T *input_data,
    T *output_data,
    const IndexT *index_data,
    const std::vector<size_t> &input_shape,
    const std::vector<size_t> &output_shape,
    const std::vector<size_t> &index_shape,
    const std::vector<ptrdiff_t> &input_strides,
    const std::vector<ptrdiff_t> &output_strides,
    const std::vector<ptrdiff_t> &index_strides,
    int dim) {

    // Calculate total number of elements to process
    size_t input_size = 1;
    for (size_t s : input_shape) {
        input_size *= s;
    }

    // Process each element in the input tensor
    for (size_t i = 0; i < input_size; ++i) {
        // Calculate input coordinates
        std::vector<size_t> in_coords(input_shape.size());
        size_t temp = i;
        for (int d = input_shape.size() - 1; d >= 0; --d) {
            in_coords[d] = temp % input_shape[d];
            temp /= input_shape[d];
        }

        // Get the index value for this position in the specified dimension
        size_t idx_pos = in_coords[dim];
        if (idx_pos >= index_shape[0]) {
            // Skip if index position is out of bounds
            continue;
        }
        
        IndexT target_idx = index_data[idx_pos];
        
        // Check bounds for target index
        if (target_idx < 0 || target_idx >= static_cast<IndexT>(output_shape[dim])) {
            // Skip out of bounds indices
            continue;
        }

        // Calculate output coordinates
        std::vector<size_t> out_coords = in_coords;
        out_coords[dim] = static_cast<size_t>(target_idx);

        // Calculate input offset
        size_t in_offset = 0;
        for (size_t d = 0; d < in_coords.size(); ++d) {
            in_offset += in_coords[d] * input_strides[d];
        }

        // Calculate output offset
        size_t out_offset = 0;
        for (size_t d = 0; d < out_coords.size(); ++d) {
            out_offset += out_coords[d] * output_strides[d];
        }

        // Copy the value from input to output at the indexed position
        output_data[out_offset] = input_data[in_offset];
    }
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *input,
    void *output,
    const void *index,
    void *stream) const {

    auto input_shape = _input_desc->shape();
    auto output_shape = _output_desc->shape();
    auto index_shape = _index_desc->shape();
    auto input_strides = _input_desc->strides();
    auto output_strides = _output_desc->strides();
    auto index_strides = _index_desc->strides();
    auto dtype = _input_desc->dtype();
    auto index_dtype = _index_desc->dtype();

    // Dispatch based on data type and index type
    if (index_dtype == INFINI_DTYPE_I32) {
        switch (dtype) {
            case INFINI_DTYPE_F16:
                index_copy_inplace_kernel<uint16_t, int32_t>(
                    static_cast<const uint16_t*>(input),
                    static_cast<uint16_t*>(output),
                    static_cast<const int32_t*>(index),
                    input_shape, output_shape, index_shape,
                    input_strides, output_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_F32:
                index_copy_inplace_kernel<float, int32_t>(
                    static_cast<const float*>(input),
                    static_cast<float*>(output),
                    static_cast<const int32_t*>(index),
                    input_shape, output_shape, index_shape,
                    input_strides, output_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_F64:
                index_copy_inplace_kernel<double, int32_t>(
                    static_cast<const double*>(input),
                    static_cast<double*>(output),
                    static_cast<const int32_t*>(index),
                    input_shape, output_shape, index_shape,
                    input_strides, output_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_BF16:
                index_copy_inplace_kernel<uint16_t, int32_t>(
                    static_cast<const uint16_t*>(input),
                    static_cast<uint16_t*>(output),
                    static_cast<const int32_t*>(index),
                    input_shape, output_shape, index_shape,
                    input_strides, output_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_I8:
                index_copy_inplace_kernel<int8_t, int32_t>(
                    static_cast<const int8_t*>(input),
                    static_cast<int8_t*>(output),
                    static_cast<const int32_t*>(index),
                    input_shape, output_shape, index_shape,
                    input_strides, output_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_I16:
                index_copy_inplace_kernel<int16_t, int32_t>(
                    static_cast<const int16_t*>(input),
                    static_cast<int16_t*>(output),
                    static_cast<const int32_t*>(index),
                    input_shape, output_shape, index_shape,
                    input_strides, output_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_I32:
                index_copy_inplace_kernel<int32_t, int32_t>(
                    static_cast<const int32_t*>(input),
                    static_cast<int32_t*>(output),
                    static_cast<const int32_t*>(index),
                    input_shape, output_shape, index_shape,
                    input_strides, output_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_I64:
                index_copy_inplace_kernel<int64_t, int32_t>(
                    static_cast<const int64_t*>(input),
                    static_cast<int64_t*>(output),
                    static_cast<const int32_t*>(index),
                    input_shape, output_shape, index_shape,
                    input_strides, output_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_U8:
                index_copy_inplace_kernel<uint8_t, int32_t>(
                    static_cast<const uint8_t*>(input),
                    static_cast<uint8_t*>(output),
                    static_cast<const int32_t*>(index),
                    input_shape, output_shape, index_shape,
                    input_strides, output_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_U16:
                index_copy_inplace_kernel<uint16_t, int32_t>(
                    static_cast<const uint16_t*>(input),
                    static_cast<uint16_t*>(output),
                    static_cast<const int32_t*>(index),
                    input_shape, output_shape, index_shape,
                    input_strides, output_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_U32:
                index_copy_inplace_kernel<uint32_t, int32_t>(
                    static_cast<const uint32_t*>(input),
                    static_cast<uint32_t*>(output),
                    static_cast<const int32_t*>(index),
                    input_shape, output_shape, index_shape,
                    input_strides, output_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_U64:
                index_copy_inplace_kernel<uint64_t, int32_t>(
                    static_cast<const uint64_t*>(input),
                    static_cast<uint64_t*>(output),
                    static_cast<const int32_t*>(index),
                    input_shape, output_shape, index_shape,
                    input_strides, output_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_BOOL:
                index_copy_inplace_kernel<bool, int32_t>(
                    static_cast<const bool*>(input),
                    static_cast<bool*>(output),
                    static_cast<const int32_t*>(index),
                    input_shape, output_shape, index_shape,
                    input_strides, output_strides, index_strides,
                    _dim);
                break;
            default:
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (index_dtype == INFINI_DTYPE_I64) {
        switch (dtype) {
            case INFINI_DTYPE_F16:
                index_copy_inplace_kernel<uint16_t, int64_t>(
                    static_cast<const uint16_t*>(input),
                    static_cast<uint16_t*>(output),
                    static_cast<const int64_t*>(index),
                    input_shape, output_shape, index_shape,
                    input_strides, output_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_F32:
                index_copy_inplace_kernel<float, int64_t>(
                    static_cast<const float*>(input),
                    static_cast<float*>(output),
                    static_cast<const int64_t*>(index),
                    input_shape, output_shape, index_shape,
                    input_strides, output_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_F64:
                index_copy_inplace_kernel<double, int64_t>(
                    static_cast<const double*>(input),
                    static_cast<double*>(output),
                    static_cast<const int64_t*>(index),
                    input_shape, output_shape, index_shape,
                    input_strides, output_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_BF16:
                index_copy_inplace_kernel<uint16_t, int64_t>(
                    static_cast<const uint16_t*>(input),
                    static_cast<uint16_t*>(output),
                    static_cast<const int64_t*>(index),
                    input_shape, output_shape, index_shape,
                    input_strides, output_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_I8:
                index_copy_inplace_kernel<int8_t, int64_t>(
                    static_cast<const int8_t*>(input),
                    static_cast<int8_t*>(output),
                    static_cast<const int64_t*>(index),
                    input_shape, output_shape, index_shape,
                    input_strides, output_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_I16:
                index_copy_inplace_kernel<int16_t, int64_t>(
                    static_cast<const int16_t*>(input),
                    static_cast<int16_t*>(output),
                    static_cast<const int64_t*>(index),
                    input_shape, output_shape, index_shape,
                    input_strides, output_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_I32:
                index_copy_inplace_kernel<int32_t, int64_t>(
                    static_cast<const int32_t*>(input),
                    static_cast<int32_t*>(output),
                    static_cast<const int64_t*>(index),
                    input_shape, output_shape, index_shape,
                    input_strides, output_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_I64:
                index_copy_inplace_kernel<int64_t, int64_t>(
                    static_cast<const int64_t*>(input),
                    static_cast<int64_t*>(output),
                    static_cast<const int64_t*>(index),
                    input_shape, output_shape, index_shape,
                    input_strides, output_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_U8:
                index_copy_inplace_kernel<uint8_t, int64_t>(
                    static_cast<const uint8_t*>(input),
                    static_cast<uint8_t*>(output),
                    static_cast<const int64_t*>(index),
                    input_shape, output_shape, index_shape,
                    input_strides, output_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_U16:
                index_copy_inplace_kernel<uint16_t, int64_t>(
                    static_cast<const uint16_t*>(input),
                    static_cast<uint16_t*>(output),
                    static_cast<const int64_t*>(index),
                    input_shape, output_shape, index_shape,
                    input_strides, output_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_U32:
                index_copy_inplace_kernel<uint32_t, int64_t>(
                    static_cast<const uint32_t*>(input),
                    static_cast<uint32_t*>(output),
                    static_cast<const int64_t*>(index),
                    input_shape, output_shape, index_shape,
                    input_strides, output_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_U64:
                index_copy_inplace_kernel<uint64_t, int64_t>(
                    static_cast<const uint64_t*>(input),
                    static_cast<uint64_t*>(output),
                    static_cast<const int64_t*>(index),
                    input_shape, output_shape, index_shape,
                    input_strides, output_strides, index_strides,
                    _dim);
                break;
            case INFINI_DTYPE_BOOL:
                index_copy_inplace_kernel<bool, int64_t>(
                    static_cast<const bool*>(input),
                    static_cast<bool*>(output),
                    static_cast<const int64_t*>(index),
                    input_shape, output_shape, index_shape,
                    input_strides, output_strides, index_strides,
                    _dim);
                break;
            default:
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::index_copy_inplace::cpu