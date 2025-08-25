#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"

#include "../cuda/kernel.cuh"
#include "cross_entropy_loss_backward_nvidia.cuh"
#include <cstddef>
#include <cuda_device_runtime_api.h>
#include <driver_types.h>

namespace op::cross_entropy_loss_backward::nvidia {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::nvidia::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &probs_desc = input_desc_vec.at(0);
    const auto &target_desc = input_desc_vec.at(1);
    const auto &grad_logits_shape = out_desc->shape();
    const auto &probs_shape = probs_desc->shape();
    const auto &target_shape = target_desc->shape();

    CHECK_DTYPE(dtype, INFINI_DTYPE_BF16, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64);

    CHECK_SAME_SHAPE(grad_logits_shape, probs_shape, target_shape);

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

    const size_t *output_shape = _info.getOutputShape();
    const size_t dim = _info.getNdim();
    size_t N{1};
    for (size_t i = 0; i < dim - 1; ++i) {
        N *= output_shape[i];
    }

    switch (_dtype) {
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<256, cuda::CrossEntropyLossBackwardOp, cuda_bfloat16>(_info, workspace, output, inputs, stream, std::move(N));
    case INFINI_DTYPE_F16:
        return _device_info->calculate<256, cuda::CrossEntropyLossBackwardOp, half>(_info, workspace, output, inputs, stream, std::move(N));
    case INFINI_DTYPE_F32:
        return _device_info->calculate<256, cuda::CrossEntropyLossBackwardOp, float>(_info, workspace, output, inputs, stream, std::move(N));
    case INFINI_DTYPE_F64:
        return _device_info->calculate<256, cuda::CrossEntropyLossBackwardOp, double>(_info, workspace, output, inputs, stream, std::move(N));
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::cross_entropy_loss_backward::nvidia
