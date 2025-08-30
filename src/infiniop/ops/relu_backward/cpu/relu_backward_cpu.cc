#include "relu_backward_cpu.h"

namespace op::relu_backward::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec
) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
//  -------------------------- start: check data type and input validity ---------------------------
    auto dtype = out_desc->dtype();
    const auto &input_desc = input_desc_vec.at(0);
    const auto &grad_output_desc = input_desc_vec.at(1);
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    CHECK_SAME_SHAPE(out_desc->shape(), input_desc->shape(), grad_output_desc->shape());
//  --------------------------- end: check data type and input validity ----------------------------

    // create CPU elementwise descriptor
    CREATE_ELEMENTWISE_CPU_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    #define CALCULATE_RELU_BACKWARD_OP(TDATA) \
        return _device_info->calculate<ReLUBackwardOp, TDATA>(_info, output, inputs, stream)
    switch (_dtype) {
    case INFINI_DTYPE_F16:
        CALCULATE_RELU_BACKWARD_OP(fp16_t);
    case INFINI_DTYPE_F32:
        CALCULATE_RELU_BACKWARD_OP(float);
    case INFINI_DTYPE_BF16:
        CALCULATE_RELU_BACKWARD_OP(bf16_t);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::relu_backward::cpu
