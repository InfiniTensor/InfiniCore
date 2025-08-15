#include "or_cpu.h"

namespace op::op_or::cpu {

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
    const auto &a_desc = input_desc_vec.at(0);
    const auto &b_desc = input_desc_vec.at(1);
    CHECK_DTYPE(dtype, INFINI_DTYPE_BOOL);
    CHECK_OR_RETURN(
        a_desc->dtype() == INFINI_DTYPE_BOOL && b_desc->dtype() == INFINI_DTYPE_BOOL,
        INFINI_STATUS_BAD_TENSOR_DTYPE
    );
    CHECK_SAME_SHAPE(out_desc->shape(), a_desc->shape(), b_desc->shape());
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

    return _device_info->calculate<OrOp, bool>(_info, output, inputs, stream);

}
} // namespace op::or::cpu
