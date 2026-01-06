#include <cstdint>
#include <type_traits>

#include "equal_cpu.h"

namespace op::equal::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    
    // Equal 算子输出是 Bool/U8，我们需要知道输入是 Float 还是 Int 才能正确读取数据进行比较。
    const auto &a_desc = input_desc_vec.at(0);
    const auto &b_desc = input_desc_vec.at(1);
    auto compute_dtype = a_desc->dtype(); 
    auto out_dtype = out_desc->dtype();

    // 1. 校验两个输入类型必须一致 (例如 float vs float)
    if (compute_dtype != b_desc->dtype()) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    // 2. 校验输出类型 (必须是布尔或整型)
    // 根据底层支持，通常是 BOOL, U8, I8
    CHECK_DTYPE(out_dtype, INFINI_DTYPE_BOOL, INFINI_DTYPE_U8, INFINI_DTYPE_I8, INFINI_DTYPE_I32);

    // 3. 校验输入类型 (支持常见的数值类型)
    CHECK_DTYPE(compute_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, 
                INFINI_DTYPE_BF16, INFINI_DTYPE_I32, INFINI_DTYPE_I64);

    const auto &c_shape = out_desc->shape();
    const auto &a_shape = a_desc->shape();
    const auto &b_shape = b_desc->shape();

    // 4. 形状校验 (假设 Elementwise 框架要求形状一致或由框架处理广播)
    CHECK_SAME_SHAPE(c_shape, a_shape, b_shape);

    // 这样 _dtype 成员变量存的就是输入类型，方便在 calculate 中分发
    CREATE_ELEMENTWISE_CPU_DESCRIPTOR(handle, compute_dtype, out_desc, input_desc_vec);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    auto dispatch_by_input = [&](auto out_tag) -> infiniStatus_t {
        using Tout = std::decay_t<decltype(out_tag)>;
        switch (_dtype) {
        case INFINI_DTYPE_F16:
            return _device_info->calculate<EqualOp, Tout, fp16_t, fp16_t>(_info, output, inputs, stream);
        case INFINI_DTYPE_F32:
            return _device_info->calculate<EqualOp, Tout, float, float>(_info, output, inputs, stream);
        case INFINI_DTYPE_F64:
            return _device_info->calculate<EqualOp, Tout, double, double>(_info, output, inputs, stream);
        case INFINI_DTYPE_BF16:
            return _device_info->calculate<EqualOp, Tout, bf16_t, bf16_t>(_info, output, inputs, stream);
        case INFINI_DTYPE_I32:
            return _device_info->calculate<EqualOp, Tout, int32_t, int32_t>(_info, output, inputs, stream);
        case INFINI_DTYPE_I64:
            return _device_info->calculate<EqualOp, Tout, int64_t, int64_t>(_info, output, inputs, stream);
        default:
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
    };

    switch (_info.getOutputDtype()) {
    case INFINI_DTYPE_BOOL:
        return dispatch_by_input(bool{});
    case INFINI_DTYPE_U8:
        return dispatch_by_input(uint8_t{});
    case INFINI_DTYPE_I8:
        return dispatch_by_input(int8_t{});
    case INFINI_DTYPE_I32:
        return dispatch_by_input(int32_t{});
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::equal::cpu
