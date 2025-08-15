#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"
#include "../cuda/kernel.cuh"
#include "div_nvidia.cuh"

namespace op::div::nvidia {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec
) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
//  --------------------- start: check data type and calculate workspace size ----------------------    

    auto dtype = out_desc->dtype();
    const auto &a_desc = input_desc_vec.at(0);
    const auto &b_desc = input_desc_vec.at(1);
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    CHECK_SAME_SHAPE(out_desc->shape(), a_desc->shape(), b_desc->shape());

//  ---------------------- end: check data type and calculate workspace size -----------------------
    // create CUDA elementwise descriptor
    CREATE_ELEMENTWISE_CUDA_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec)

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }

    #define CALCULATE_DIV_OP(BLOCK_SIZE, TDATA) \
        return _device_info->calculate<BLOCK_SIZE, cuda::DivOp, TDATA>(_info, workspace, output, inputs, stream)

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        CALCULATE_DIV_OP(256, half);
    case INFINI_DTYPE_F32:
        CALCULATE_DIV_OP(256, float);
    case INFINI_DTYPE_BF16:
        CALCULATE_DIV_OP(256, cuda_bfloat16);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::div::nvidia
