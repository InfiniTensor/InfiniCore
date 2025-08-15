#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"
#include "../cuda/kernel.cuh"
#include "silu_nvidia.cuh"

namespace op::silu::nvidia {

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
    const auto &input_desc = input_desc_vec.at(0);
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    CHECK_SAME_SHAPE(out_desc->shape(), input_desc->shape());


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

    #define CALCULATE_SILU_OP(BLOCK_SIZE, TDATA) \
        return _device_info->calculate<BLOCK_SIZE, cuda::SiluOp, TDATA>(_info, workspace, output, inputs, stream)

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        CALCULATE_SILU_OP(256, half);
    case INFINI_DTYPE_F32:
        CALCULATE_SILU_OP(256, float);
    case INFINI_DTYPE_BF16:
        CALCULATE_SILU_OP(256, cuda_bfloat16);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::silu::nvidia
