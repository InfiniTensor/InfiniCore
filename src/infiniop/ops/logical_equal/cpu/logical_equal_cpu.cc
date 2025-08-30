#include "logical_equal_cpu.h"
#include "infinicore.h"
#include <cmath>
#include <cstdint>

namespace op::logical_equal::cpu {

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

    CHECK_DTYPE(dtype, INFINI_DTYPE_BOOL, INFINI_DTYPE_I8, INFINI_DTYPE_I16,
                INFINI_DTYPE_I32, INFINI_DTYPE_I64, INFINI_DTYPE_BF16,
                INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

    CHECK_SAME_SHAPE(c_shape, a_shape, b_shape);

    // create CPU elementwise descriptor
    CREATE_ELEMENTWISE_CPU_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec)

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {
#define CASE(CASE, TYPE) \
    case CASE:           \
        return _device_info->calculate<LogicalEqualOp, TYPE>(_info, output, inputs, stream);

    switch (_dtype) {
        CASE(INFINI_DTYPE_BOOL, bool)
        CASE(INFINI_DTYPE_I8, int8_t)
        CASE(INFINI_DTYPE_I16, int16_t)
        CASE(INFINI_DTYPE_I32, int32_t)
        CASE(INFINI_DTYPE_I64, int64_t)
        CASE(INFINI_DTYPE_F16, fp16_t)
        CASE(INFINI_DTYPE_BF16, bf16_t)
        CASE(INFINI_DTYPE_F32, float)
        CASE(INFINI_DTYPE_F64, double_t)

    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::logical_equal::cpu
