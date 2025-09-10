#include "cross_entropy_loss_backward_cpu.h"

namespace op::cross_entropy_loss_backward::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    std::vector<infiniopTensorDescriptor_t> input_desc_vec) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto dtype = out_desc->dtype();

    const auto &probs_desc = input_desc_vec.at(0);
    const auto &target_desc = input_desc_vec.at(1);
    const auto &grad_logits_shape = out_desc->shape();
    const auto &probs_shape = probs_desc->shape();
    const auto &target_shape = target_desc->shape();

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_F64, INFINI_DTYPE_BF16);

    CHECK_SAME_SHAPE(grad_logits_shape, probs_shape, target_shape);

    // creat CPU elementwise descriptor
    CREATE_ELEMENTWISE_CPU_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    // 计算实际的batch_size
    // PyTorch CrossEntropyLoss默认reduction='mean'，需要除以batch_size
    // 对于CrossEntropy，batch_size通常是除了最后一维(num_classes)之外的所有维度的乘积
    // 例如：[N, C] -> batch_size = N
    //      [N, H, W, C] -> batch_size = N * H * W
    size_t batch_size = 1;
    // size_t num_dims = probs_shape.size();
    size_t num_dims = _info.getNdim();

    // 假设最后一维是classes维度，前面所有维度都是batch维度
    for (size_t i = 0; i < num_dims - 1; i++) {
        batch_size *= _info.getOutputShape()[i];
    }

    // 防止batch_size为0的情况
    if (batch_size == 0) {
        batch_size = 1;
    }

    switch (_dtype) {
    case INFINI_DTYPE_F16:
        return _device_info->calculate<CrossEntropyLossBackwardOp, fp16_t>(_info, output, inputs, stream, batch_size);
    case INFINI_DTYPE_F32:
        return _device_info->calculate<CrossEntropyLossBackwardOp, float>(_info, output, inputs, stream, batch_size);
    case INFINI_DTYPE_F64:
        return _device_info->calculate<CrossEntropyLossBackwardOp, double>(_info, output, inputs, stream, batch_size);
    case INFINI_DTYPE_BF16:
        return _device_info->calculate<CrossEntropyLossBackwardOp, bf16_t>(_info, output, inputs, stream, batch_size);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    return INFINI_STATUS_SUCCESS;
}
} // namespace op::cross_entropy_loss_backward::cpu
