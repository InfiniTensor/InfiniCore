#include "all_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../../utils.h"
#include <iostream>
namespace op::all::cpu {

Descriptor::~Descriptor() {}
//  一个descriptor的create 一个AllInfo 的create
infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t output_desc,
    infiniopTensorDescriptor_t input_desc,
    size_t *dim, 
    size_t dim_size, 
    bool keepdim) {
    // auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = AllInfo::create(output_desc, input_desc, dim, dim_size, keepdim);
    CHECK_RESULT(result);
    
    *desc_ptr = new Descriptor(nullptr, result.take(), 0, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

namespace{
// template<typename Tdata>
// infiniStatus_t calculateAll(
//     const AllInfo &info,
//     bool *output,
//     const Tdata *input,
//     size_t *dim,
//     size_t dim_size,
//     bool keepdim){
//         if (info.reduce_dim_size == info.ndim){ // 规约到标量
//             bool result = true;
//             for(size_t index = 0; index < info.input_size; index++){
//                 size_t input_offset = op::common_cpu::indexToOffset(index, info.ndim, info.permuted_input_shape.data(), info.permuted_input_strides.data());
//                 result = result && input[input_offset];
//             }
//             output[0] = result;
//             return INFINI_STATUS_SUCCESS;
//         } else{
//             for (size_t i = 0; i < info.output_size; i++) {
//                 size_t output_offset = op::common_cpu::indexToOffset(i, info.output_shape.size(), info.output_shape.data(), info.output_strides.data());
//                 bool result = true;
//                 for(size_t j = 0; j < info.reduce_num; j++){
//                     size_t input_offset = op::common_cpu::indexToOffset(j + i * info.reduce_num, info.ndim, info.permuted_input_shape.data(), info.permuted_input_strides.data());
//                     result = result && input[input_offset];
//                 }
//                 output[output_offset] = result;
//             }
//             return INFINI_STATUS_SUCCESS;
//         }
//     }
// }

// _TENSOR_DTYPES = [infinicore.bool, infinicore.uint8]

template<typename Tdata>
infiniStatus_t calculateAll(
    const AllInfo &info,
    bool *output,
    const Tdata *input,
    size_t *dim,
    size_t dim_size,
    bool keepdim){
        if (info.reduce_dim_size == info.ndim){ // 规约到标量
            bool result = true;
            for(size_t index = 0; index < info.input_size; index++){
                size_t input_offset = op::common_cpu::indexToOffset(index, info.ndim, info.permuted_input_shape.data(), info.permuted_input_strides.data());
                result = result && input[input_offset];
            }
            output[0] = result;
            return INFINI_STATUS_SUCCESS;
        } else{
            std::cout << "=== DEBUG All calculateAll ===" << std::endl;
            std::cout << "output_shape: ";
            for(auto s : info.output_shape) std::cout << s << " ";
            std::cout << std::endl;
            std::cout << "output_strides: ";
            for(auto s : info.output_strides) std::cout << s << " ";
            std::cout << std::endl;
            std::cout << "permuted_input_shape: ";
            for(auto s : info.permuted_input_shape) std::cout << s << " ";
            std::cout << std::endl;
            std::cout << "permuted_input_strides: ";
            for(auto s : info.permuted_input_strides) std::cout << s << " ";
            std::cout << std::endl;
            std::cout << "reduce_num: " << info.reduce_num << std::endl;
            std::cout << "output_size: " << info.output_size << std::endl;
            
            // for (size_t i = 0; i < info.output_size; i++) {
            for (size_t i = info.output_size; i-->0;) {
                size_t output_offset = op::common_cpu::indexToOffset(i, info.output_shape.size(), info.output_shape.data(), info.output_strides.data());
                std::cout << "i=" << i << ", output_offset=" << output_offset << std::endl;
                
                bool result = true;
                for(size_t j = 0; j < info.reduce_num; j++){
                    size_t input_flat = j + i * info.reduce_num;
                    size_t input_offset = op::common_cpu::indexToOffset(input_flat, info.ndim, info.permuted_input_shape.data(), info.permuted_input_strides.data());
                    Tdata input_val = input[input_offset];
                    bool bool_val = static_cast<bool>(input_val);
                    std::cout << "  j=" << j << ", input_flat=" << input_flat << ", input_offset=" << input_offset << ", input_val=" << (input_val ? "true" : "false") << std::endl;
                    result = result && bool_val;
                }
                std::cout << "  final result=" << (result ? "true" : "false") << std::endl;
                output[output_offset] = result;
            }
            std::cout << "=== END DEBUG ===" << std::endl;
            return INFINI_STATUS_SUCCESS;
        }
    }
}
infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    const void *input,
    size_t *dim,
    size_t dim_size,
    bool keepdim,
    void *stream) const {
    switch (_info.dtype) {
    case INFINI_DTYPE_BOOL:
        return calculateAll<bool>(_info, reinterpret_cast<bool *>(output), reinterpret_cast<const bool *>(input), dim, dim_size, keepdim);
    case INFINI_DTYPE_U8:
        return calculateAll<uint8_t>(_info, reinterpret_cast<bool *>(output), reinterpret_cast<const uint8_t *>(input), dim, dim_size, keepdim);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::all::cpu
