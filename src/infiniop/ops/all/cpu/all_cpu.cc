#include "all_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include "../../../../utils.h"
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
                result = result &&input[input_offset];
            }
            output[0] = result;
            return INFINI_STATUS_SUCCESS;
        } else{
            for (size_t i = 0; i < info.output_size; i++) {
                size_t output_offset = op::common_cpu::indexToOffset(i, info.output_shape.size(), info.output_shape.data(), info.output_strides.data());
                bool result = true;
                for(size_t j = 0; j < info.reduce_num; j++){
                    size_t input_offset = op::common_cpu::indexToOffset(j + i * info.reduce_num, info.ndim, info.permuted_input_shape.data(), info.permuted_input_strides.data());
                    result = result && input[input_offset];
                }
                output[output_offset] = result;
            }
            return INFINI_STATUS_SUCCESS;
        }
    }
}

// _TENSOR_DTYPES = [infinicore.bool, infinicore.uint8]

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
        return calculateAll<bool>(_info, output, reinterpret_cast<const bool *>(input), dim, dim_size, keepdim);
    case INFINI_DTYPE_U8:
        return calculateAll<bool>(_info, output, reinterpret_cast<const bool *>(input), dim, dim_size, keepdim);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::all::cpu
