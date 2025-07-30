#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"

#include "../cuda/kernel.cuh"
#include "infinicore.h"
#include "silu_nvidia.cuh"

namespace op::silu::nvidia {
Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(infiniopHandle_t handle_,
                                  Descriptor **desc_ptr,
                                  infiniopTensorDescriptor_t output_desc,
                                  std::vector<infiniopTensorDescriptor_t> input_descs) {
    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = output_desc->dtype();

    const auto &a_desc = input_descs.at(0);
    const auto c_shape = output_desc->shape();
    const auto a_shape = a_desc->shape();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    CHECK_SAME_SHAPE(c_shape, a_shape);

    // create CPU elementwise descriptor
    CREATE_ELEMENTWISE_CUDA_DESCRIPTOR(handle, dtype, output_desc, input_descs)

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
    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return _device_info->calculate<256, cuda::SiluOp, half>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<256, cuda::SiluOp, cuda_bfloat16>(_info, workspace, output, inputs, stream);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<256, cuda::SiluOp, float>(_info, workspace, output, inputs, stream);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::silu::nvidia
