#include "equal_cpu.h"
#include "../../../devices/cpu/cpu_handle.h"
#include "../../../../utils/custom_types.h"
#include <cmath>
#include <cstdio>

namespace op::equal::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &a_desc = input_desc_vec.at(0);
    const auto &b_desc = input_desc_vec.at(1);
    const auto &c_shape = out_desc->shape();
    const auto &a_shape = a_desc->shape();
    const auto &b_shape = b_desc->shape();

    // Equal算子支持所有合法类型，输出为bool类型
    // Check if input dtypes are supported
    if (a_desc->dtype() != b_desc->dtype()) {
        return INFINI_STATUS_BAD_PARAM;
    }
    
    // 输出必须是bool类型且为标量（torch.equal返回单个bool值）
    if (dtype != INFINI_DTYPE_BOOL) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    
    // 输出必须是标量（shape为空或者所有维度为1）
    if (c_shape.size() > 0) {
        bool is_scalar = true;
        for (auto dim : c_shape) {
            if (dim != 1) {
                is_scalar = false;
                break;
            }
        }
        if (!is_scalar) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
    }

    // 输入张量形状必须相同
    if (a_shape.size() != b_shape.size()) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }
    for (size_t i = 0; i < a_shape.size(); i++) {
        if (a_shape[i] != b_shape[i]) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
    }

    *desc_ptr = new Descriptor(
        a_desc->dtype(),
        a_desc->shape(),
        a_desc->strides(),
        b_desc->strides(),
        handle->device,
        handle->device_id
    );

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {
    
    const void *a_data = inputs[0];
    const void *b_data = inputs[1];
    bool *result = static_cast<bool *>(output);

    // 计算张量的总元素数量
    size_t total_elements = 1;
    for (auto dim : _shape) {
        total_elements *= dim;
    }
    
    

    // 根据数据类型进行比较
    switch (_dtype) {
    case INFINI_DTYPE_F16:
        *result = compareArraysCpu<fp16_t>(a_data, b_data, total_elements, _a_strides, _b_strides);
        break;
    case INFINI_DTYPE_F32:
        *result = compareArraysCpu<float>(a_data, b_data, total_elements, _a_strides, _b_strides);
        printf("[DEBUG] F32 comparison result: %s\n", *result ? "true" : "false");
        break;
    case INFINI_DTYPE_F64:
        *result = compareArraysCpu<double>(a_data, b_data, total_elements, _a_strides, _b_strides);
        break;
    case INFINI_DTYPE_BF16:
        *result = compareArraysCpu<bf16_t>(a_data, b_data, total_elements, _a_strides, _b_strides);
        break;
    case INFINI_DTYPE_I8:
        *result = compareArraysCpu<int8_t>(a_data, b_data, total_elements, _a_strides, _b_strides);
        break;
    case INFINI_DTYPE_I16:
        *result = compareArraysCpu<int16_t>(a_data, b_data, total_elements, _a_strides, _b_strides);
        break;
    case INFINI_DTYPE_I32:
        *result = compareArraysCpu<int32_t>(a_data, b_data, total_elements, _a_strides, _b_strides);
        break;
    case INFINI_DTYPE_I64:
        *result = compareArraysCpu<int64_t>(a_data, b_data, total_elements, _a_strides, _b_strides);
        break;
    case INFINI_DTYPE_U8:
        *result = compareArraysCpu<uint8_t>(a_data, b_data, total_elements, _a_strides, _b_strides);
        break;
    case INFINI_DTYPE_U16:
        *result = compareArraysCpu<uint16_t>(a_data, b_data, total_elements, _a_strides, _b_strides);
        break;
    case INFINI_DTYPE_U32:
        *result = compareArraysCpu<uint32_t>(a_data, b_data, total_elements, _a_strides, _b_strides);
        break;
    case INFINI_DTYPE_U64:
        *result = compareArraysCpu<uint64_t>(a_data, b_data, total_elements, _a_strides, _b_strides);
        break;
    case INFINI_DTYPE_BOOL:
        *result = compareArraysCpu<bool>(a_data, b_data, total_elements, _a_strides, _b_strides);
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }


    return INFINI_STATUS_SUCCESS;
}

template <typename T>
bool Descriptor::compareArraysCpu(
    const void *a_data,
    const void *b_data,
    size_t total_elements,
    const std::vector<ptrdiff_t> &a_strides,
    const std::vector<ptrdiff_t> &b_strides) const {
    
    const T *a_ptr = static_cast<const T *>(a_data);
    const T *b_ptr = static_cast<const T *>(b_data);
    

    
    // Check if arrays are contiguous
    bool a_contiguous = true, b_contiguous = true;
    size_t expected_stride = sizeof(T);
    for (int i = _shape.size() - 1; i >= 0; i--) {
        if (a_strides[i] != static_cast<ptrdiff_t>(expected_stride)) a_contiguous = false;
        if (b_strides[i] != static_cast<ptrdiff_t>(expected_stride)) b_contiguous = false;
        expected_stride *= _shape[i];
    }
    
    if (a_contiguous && b_contiguous) {
        // Fast path for contiguous arrays
        printf("[DEBUG] Using contiguous path\n");
        for (size_t i = 0; i < total_elements; i++) {
            bool are_equal;
            if constexpr (std::is_same_v<T, fp16_t>) {
                // For fp16, compare the underlying bits
                are_equal = (a_ptr[i]._v == b_ptr[i]._v);
            } else if constexpr (std::is_same_v<T, bf16_t>) {
                // For bf16, compare the underlying bits
                are_equal = (a_ptr[i]._v == b_ptr[i]._v);
            } else if constexpr (std::is_floating_point_v<T>) {
                // For floating point types, handle NaN according to torch.equal behavior
                // torch.equal returns False if any tensor contains NaN
                if (std::isnan(a_ptr[i]) || std::isnan(b_ptr[i])) {
                    return false;
                }
                are_equal = (a_ptr[i] == b_ptr[i]);
            } else {
                // For integer and bool types
                are_equal = (a_ptr[i] == b_ptr[i]);
            }
            
            if (!are_equal) {
                printf("[DEBUG] Found unequal elements at index %zu\n", i);
                return false;
            }
        }

    } else {
        // Slow path for non-contiguous arrays
        std::vector<size_t> indices(_shape.size(), 0);
        
        for (size_t linear_idx = 0; linear_idx < total_elements; linear_idx++) {
            // Calculate memory offsets for both arrays
            size_t a_offset = 0, b_offset = 0;
            for (size_t dim = 0; dim < _shape.size(); dim++) {
                a_offset += indices[dim] * a_strides[dim];
                b_offset += indices[dim] * b_strides[dim];
            }
            
            const T *a_elem = reinterpret_cast<const T *>(reinterpret_cast<const char *>(a_ptr) + a_offset);
            const T *b_elem = reinterpret_cast<const T *>(reinterpret_cast<const char *>(b_ptr) + b_offset);
            
            bool are_equal;
            if constexpr (std::is_same_v<T, fp16_t>) {
                are_equal = (a_elem->_v == b_elem->_v);
            } else if constexpr (std::is_same_v<T, bf16_t>) {
                are_equal = (a_elem->_v == b_elem->_v);
            } else if constexpr (std::is_floating_point_v<T>) {
                if (std::isnan(*a_elem) || std::isnan(*b_elem)) {
                    return false;
                }
                are_equal = (*a_elem == *b_elem);
            } else {
                are_equal = (*a_elem == *b_elem);
            }
            
            if (!are_equal) {

                return false;
            }
            
            // Update indices for next iteration
            for (int dim = _shape.size() - 1; dim >= 0; dim--) {
                indices[dim]++;
                if (indices[dim] < _shape[dim]) {
                    break;
                }
                indices[dim] = 0;
            }
        }

    }
    

    return true;
}

} // namespace op::equal::cpu