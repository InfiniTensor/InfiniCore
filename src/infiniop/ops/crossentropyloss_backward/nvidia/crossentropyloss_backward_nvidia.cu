#include "../../../elementwise/nvidia/elementwise_nvidia.cuh"

#include "../cuda/kernel.cuh"
#include "crossentropyloss_backward_nvidia.cuh"



namespace op::crossentropyloss_backward::nvidia {

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
    const auto &grad_input_shape = out_desc->shape();
    const auto &probs_shape = probs_desc->shape();
    const auto &target_shape = target_desc->shape();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    CHECK_SAME_SHAPE(grad_input_shape, probs_shape);
    CHECK_SAME_SHAPE(out_desc->shape(), probs_desc->shape(), target_desc->shape());
    // According to competition.md, target is one-hot tensor with same shape as logits

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

    // Calculate batch_size as the product of all dimensions except the last one (class dimension)
    // Use probs tensor shape (input 0) and consider stride=0 cases for effective shape
    size_t batch_size = 1;
    const size_t* probs_shape = _info.getInputShape(0);
    const ptrdiff_t* probs_strides = _info.getInputStrides(0);
    size_t ndim = _info.getNdim();
    for (size_t d = 0; d < ndim - 1; d++) {
        // If stride is 0, the effective size for this dimension is 1 (broadcasted)
        size_t effective_size = (probs_strides[d] == 0) ? 1 : probs_shape[d];
        batch_size *= effective_size;
    }
    
    switch (_dtype) {
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<256, cuda::CrossEntropyLossBackwardOp, cuda_bfloat16>(_info, workspace, output, inputs, stream, std::move(batch_size));
    case INFINI_DTYPE_F16:
        return _device_info->calculate<256, cuda::CrossEntropyLossBackwardOp, half>(_info, workspace, output, inputs, stream, std::move(batch_size));
    case INFINI_DTYPE_F32:
        return _device_info->calculate<256, cuda::CrossEntropyLossBackwardOp, float>(_info, workspace, output, inputs, stream, std::move(batch_size));
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::crossentropyloss_backward::nvidia