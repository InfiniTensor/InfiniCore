#include "gather_cpu.h"
#include "../../../../utils.h"
#include "../../../tensor.h"
#include <cstring>
#include <omp.h>

namespace op::gather::cpu {

template<typename T, typename IndexT>
static infiniStatus_t calculate_gather(
    const std::vector<size_t> &output_shape,
    const std::vector<size_t> &input_shape,
    const std::vector<ptrdiff_t> &output_strides,
    const std::vector<ptrdiff_t> &input_strides,
    const std::vector<ptrdiff_t> &index_strides,
    size_t dim,
    T * output,
    const T * input,
    const IndexT * index
) {
    // 计算总元素数量
    size_t total_elements = 1;
    for (size_t d = 0; d < output_shape.size(); d++) {
        total_elements *= output_shape[d];
    }
    
    // 使用串行处理避免并行化可能导致的问题
    for(size_t linear_idx = 0; linear_idx < total_elements; linear_idx++) {
        
        // 将线性索引转换为多维坐标
        std::vector<size_t> coords(output_shape.size());
        size_t temp_idx = linear_idx;
        for (int d = output_shape.size() - 1; d >= 0; d--) {
            coords[d] = temp_idx % output_shape[d];
            temp_idx /= output_shape[d];
        }
        
        // 计算输出偏移量
        size_t output_offset = 0;
        for (size_t d = 0; d < output_shape.size(); d++) {
            output_offset += coords[d] * output_strides[d];
        }
        
        // 计算索引偏移量（以字节为单位，与其他strides保持一致）
        size_t index_offset = 0;
        for (size_t d = 0; d < output_shape.size(); d++) {
            index_offset += coords[d] * index_strides[d];
        }
        
        // 获取索引值并进行边界检查
        IndexT gather_index = *((const IndexT*)((const char*)index + index_offset));
        if (gather_index < 0 || gather_index >= static_cast<IndexT>(input_shape[dim])) {
            return INFINI_STATUS_BAD_PARAM; // 返回错误状态
        }
        
        // 计算输入偏移量 - 对于gather操作，在dim维度使用gather_index，其他维度使用输出坐标
        size_t input_offset = 0;
        for (size_t d = 0; d < input_shape.size(); d++) {
            if (d == dim) {
                input_offset += gather_index * input_strides[d];
            } else {
                input_offset += coords[d] * input_strides[d];
            }
        }
        
        // 复制数据（按字节复制，避免类型转换问题）
        std::memcpy((char*)output + output_offset, (const char*)input + input_offset, sizeof(T));
    }
    
    return INFINI_STATUS_SUCCESS;
}

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
    auto index_shape = index_desc->shape();
    
    if (dim < 0 || dim >= static_cast<int>(input_shape.size())) {
        return INFINI_STATUS_BAD_PARAM;
    }

    // Check that input, output and index have same number of dimensions
    if (input_shape.size() != output_shape.size() || input_shape.size() != index_shape.size()) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    // Check that output shape equals index shape (torch.gather requirement)
    for (size_t i = 0; i < output_shape.size(); ++i) {
        if (output_shape[i] != index_shape[i]) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
    }

    // Check that all dimensions except dim have same size between input and index
    for (size_t i = 0; i < input_shape.size(); ++i) {
        if (i != static_cast<size_t>(dim) && input_shape[i] != index_shape[i]) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
    }

    auto desc = new Descriptor();
    desc->_input_desc = input_desc;
    desc->_output_desc = output_desc;
    desc->_index_desc = index_desc;
    desc->_dim = dim;
    desc->_input_shape = input_shape;
    desc->_output_shape = output_shape;
    desc->_input_strides = input_desc->getByteStrides();
    desc->_output_strides = output_desc->getByteStrides();
    desc->_index_strides = index_desc->getByteStrides(); // 改为使用字节strides保持一致
    desc->_dtype = input_desc->dtype();
    desc->_index_dtype = index_desc->dtype();
    desc->_handle = handle;
    desc->device_type = INFINI_DEVICE_CPU;
    desc->device_id = handle->device_id;

    *desc_ptr = desc;
    return INFINI_STATUS_SUCCESS;
}

// gather_kernel function removed - replaced by calculate_gather

// gather_kernel函数已被移除，直接使用calculate_gather函数

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    const void *index,
    void *stream) const {

    auto output_shape = _output_shape;
    auto input_shape = _input_shape;
    auto index_shape = _index_desc->shape();
    auto input_strides = _input_strides;
    auto output_strides = _output_strides;
    auto index_strides = _index_strides;
    auto dtype = _dtype;
    auto index_dtype = _index_dtype;
    
    // 计算输出总元素数量
    size_t output_size = 1;
    for (size_t d = 0; d < output_shape.size(); d++) {
        output_size *= output_shape[d];
    }

    // 根据数据类型调用相应的kernel
    if (index_dtype == INFINI_DTYPE_I32) {
        switch (dtype) {
            case INFINI_DTYPE_F16:
                return calculate_gather<fp16_t, int32_t>(
                     output_shape, input_shape,
                     output_strides, input_strides, index_strides,
                     static_cast<size_t>(_dim),
                     static_cast<fp16_t*>(output),
                     static_cast<const fp16_t*>(input),
                     static_cast<const int32_t*>(index));
                break;
            case INFINI_DTYPE_F32:
                return calculate_gather<float, int32_t>(
                     output_shape, input_shape,
                     output_strides, input_strides, index_strides,
                     static_cast<size_t>(_dim),
                     static_cast<float*>(output),
                     static_cast<const float*>(input),
                     static_cast<const int32_t*>(index));
                 break;
             case INFINI_DTYPE_F64:
                 return calculate_gather<double, int32_t>(
                      output_shape, input_shape,
                      output_strides, input_strides, index_strides,
                      static_cast<size_t>(_dim),
                      static_cast<double*>(output),
                      static_cast<const double*>(input),
                      static_cast<const int32_t*>(index));
                 break;
             case INFINI_DTYPE_BF16:
                 return calculate_gather<bf16_t, int32_t>(
                      output_shape, input_shape,
                      output_strides, input_strides, index_strides,
                      static_cast<size_t>(_dim),
                      static_cast<bf16_t*>(output),
                      static_cast<const bf16_t*>(input),
                      static_cast<const int32_t*>(index));
                 break;
             case INFINI_DTYPE_I8:
                 return calculate_gather<int8_t, int32_t>(
                      output_shape, input_shape,
                      output_strides, input_strides, index_strides,
                      static_cast<size_t>(_dim),
                      static_cast<int8_t*>(output),
                      static_cast<const int8_t*>(input),
                      static_cast<const int32_t*>(index));
                 break;
             case INFINI_DTYPE_I16:
                 return calculate_gather<int16_t, int32_t>(
                      output_shape, input_shape,
                      output_strides, input_strides, index_strides,
                      static_cast<size_t>(_dim),
                      static_cast<int16_t*>(output),
                      static_cast<const int16_t*>(input),
                      static_cast<const int32_t*>(index));
                 break;
             case INFINI_DTYPE_I32:
                 return calculate_gather<int32_t, int32_t>(
                      output_shape, input_shape,
                      output_strides, input_strides, index_strides,
                      static_cast<size_t>(_dim),
                      static_cast<int32_t*>(output),
                      static_cast<const int32_t*>(input),
                      static_cast<const int32_t*>(index));
                 break;
             case INFINI_DTYPE_I64:
                 return calculate_gather<int64_t, int32_t>(
                      output_shape, input_shape,
                      output_strides, input_strides, index_strides,
                      static_cast<size_t>(_dim),
                      static_cast<int64_t*>(output),
                      static_cast<const int64_t*>(input),
                      static_cast<const int32_t*>(index));
                 break;
             case INFINI_DTYPE_U8:
                 return calculate_gather<uint8_t, int32_t>(
                      output_shape, input_shape,
                      output_strides, input_strides, index_strides,
                      static_cast<size_t>(_dim),
                      static_cast<uint8_t*>(output),
                      static_cast<const uint8_t*>(input),
                      static_cast<const int32_t*>(index));
                 break;
             case INFINI_DTYPE_U16:
                 return calculate_gather<uint16_t, int32_t>(
                      output_shape, input_shape,
                      output_strides, input_strides, index_strides,
                      static_cast<size_t>(_dim),
                      static_cast<uint16_t*>(output),
                      static_cast<const uint16_t*>(input),
                      static_cast<const int32_t*>(index));
                 break;
             case INFINI_DTYPE_U32:
                 return calculate_gather<uint32_t, int32_t>(
                      output_shape, input_shape,
                      output_strides, input_strides, index_strides,
                      static_cast<size_t>(_dim),
                      static_cast<uint32_t*>(output),
                      static_cast<const uint32_t*>(input),
                      static_cast<const int32_t*>(index));
                 break;
             case INFINI_DTYPE_U64:
                 return calculate_gather<uint64_t, int32_t>(
                      output_shape, input_shape,
                      output_strides, input_strides, index_strides,
                      static_cast<size_t>(_dim),
                      static_cast<uint64_t*>(output),
                      static_cast<const uint64_t*>(input),
                      static_cast<const int32_t*>(index));
                 break;
             case INFINI_DTYPE_BOOL:
                 return calculate_gather<bool, int32_t>(
                      output_shape, input_shape,
                      output_strides, input_strides, index_strides,
                      static_cast<size_t>(_dim),
                      static_cast<bool*>(output),
                      static_cast<const bool*>(input),
                      static_cast<const int32_t*>(index));
                 break;
            default:
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else if (index_dtype == INFINI_DTYPE_I64) {
        switch (dtype) {
            case INFINI_DTYPE_F16:
                return calculate_gather<fp16_t, int64_t>(
                    output_shape, input_shape,
                    output_strides, input_strides, index_strides,
                    static_cast<size_t>(_dim),
                    static_cast<fp16_t*>(output),
                    static_cast<const fp16_t*>(input),
                    static_cast<const int64_t*>(index));
                break;
            case INFINI_DTYPE_F32:
                return calculate_gather<float, int64_t>(
                    output_shape, input_shape,
                    output_strides, input_strides, index_strides,
                    static_cast<size_t>(_dim),
                    static_cast<float*>(output),
                    static_cast<const float*>(input),
                    static_cast<const int64_t*>(index));
                break;
            case INFINI_DTYPE_F64:
                return calculate_gather<double, int64_t>(
                    output_shape, input_shape,
                    output_strides, input_strides, index_strides,
                    static_cast<size_t>(_dim),
                    static_cast<double*>(output),
                    static_cast<const double*>(input),
                    static_cast<const int64_t*>(index));
                break;
            case INFINI_DTYPE_BF16:
                return calculate_gather<bf16_t, int64_t>(
                    output_shape, input_shape,
                    output_strides, input_strides, index_strides,
                    static_cast<size_t>(_dim),
                    static_cast<bf16_t*>(output),
                    static_cast<const bf16_t*>(input),
                    static_cast<const int64_t*>(index));
                break;
            case INFINI_DTYPE_I8:
                return calculate_gather<int8_t, int64_t>(
                    output_shape, input_shape,
                    output_strides, input_strides, index_strides,
                    static_cast<size_t>(_dim),
                    static_cast<int8_t*>(output),
                    static_cast<const int8_t*>(input),
                    static_cast<const int64_t*>(index));
                break;
            case INFINI_DTYPE_I16:
                return calculate_gather<int16_t, int64_t>(
                    output_shape, input_shape,
                    output_strides, input_strides, index_strides,
                    static_cast<size_t>(_dim),
                    static_cast<int16_t*>(output),
                    static_cast<const int16_t*>(input),
                    static_cast<const int64_t*>(index));
                break;
            case INFINI_DTYPE_I32:
                return calculate_gather<int32_t, int64_t>(
                    output_shape, input_shape,
                    output_strides, input_strides, index_strides,
                    static_cast<size_t>(_dim),
                    static_cast<int32_t*>(output),
                    static_cast<const int32_t*>(input),
                    static_cast<const int64_t*>(index));
                break;
            case INFINI_DTYPE_I64:
                return calculate_gather<int64_t, int64_t>(
                    output_shape, input_shape,
                    output_strides, input_strides, index_strides,
                    static_cast<size_t>(_dim),
                    static_cast<int64_t*>(output),
                    static_cast<const int64_t*>(input),
                    static_cast<const int64_t*>(index));
                break;
            case INFINI_DTYPE_U8:
                return calculate_gather<uint8_t, int64_t>(
                    output_shape, input_shape,
                    output_strides, input_strides, index_strides,
                    static_cast<size_t>(_dim),
                    static_cast<uint8_t*>(output),
                    static_cast<const uint8_t*>(input),
                    static_cast<const int64_t*>(index));
                break;
            case INFINI_DTYPE_U16:
                return calculate_gather<uint16_t, int64_t>(
                    output_shape, input_shape,
                    output_strides, input_strides, index_strides,
                    static_cast<size_t>(_dim),
                    static_cast<uint16_t*>(output),
                    static_cast<const uint16_t*>(input),
                    static_cast<const int64_t*>(index));
                break;
            case INFINI_DTYPE_U32:
                return calculate_gather<uint32_t, int64_t>(
                    output_shape, input_shape,
                    output_strides, input_strides, index_strides,
                    static_cast<size_t>(_dim),
                    static_cast<uint32_t*>(output),
                    static_cast<const uint32_t*>(input),
                    static_cast<const int64_t*>(index));
                break;
            case INFINI_DTYPE_U64:
                return calculate_gather<uint64_t, int64_t>(
                    output_shape, input_shape,
                    output_strides, input_strides, index_strides,
                    static_cast<size_t>(_dim),
                    static_cast<uint64_t*>(output),
                    static_cast<const uint64_t*>(input),
                    static_cast<const int64_t*>(index));
                break;
            case INFINI_DTYPE_BOOL:
                return calculate_gather<bool, int64_t>(
                    output_shape, input_shape,
                    output_strides, input_strides, index_strides,
                    static_cast<size_t>(_dim),
                    static_cast<bool*>(output),
                    static_cast<const bool*>(input),
                    static_cast<const int64_t*>(index));
                break;
            default:
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    } else {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::gather::cpu