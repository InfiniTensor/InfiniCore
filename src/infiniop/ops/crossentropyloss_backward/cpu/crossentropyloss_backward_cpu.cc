#include "crossentropyloss_backward_cpu.h"

namespace op::crossentropyloss_backward::cpu {

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

    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);

    CHECK_SAME_SHAPE(grad_logits_shape, probs_shape);
    CHECK_SAME_SHAPE(grad_logits_shape, target_shape);

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

    // Create a custom operator with batch_size
    CrossEntropyLossBackwardOp op(batch_size);

    // Directly use the operator
    switch (_dtype) {
    case INFINI_DTYPE_F16: {
        auto* out_ptr = reinterpret_cast<fp16_t*>(output);
        auto* probs_ptr = reinterpret_cast<const fp16_t*>(inputs[0]);
        auto* target_ptr = reinterpret_cast<const fp16_t*>(inputs[1]);
        
        size_t output_size = _info.getOutputSize();
        #pragma omp parallel for
        for (size_t i = 0; i < output_size; ++i) {
            size_t out_idx = _info.isOutputContiguous() ? i : 
                op::common_cpu::indexToOffset(i, _info.getNdim(), _info.getOutputShape(), _info.getOutputStrides());
            
            size_t probs_idx = _info.getInputContiguous()[0] ? i : 
                (_info.getInputBroadcasted()[0] ? 
                    op::common_cpu::indexToReducedOffset(i, _info.getNdim(), _info.getOutputStrides(), _info.getInputStrides(0)) : 
                    op::common_cpu::indexToOffset(i, _info.getNdim(), _info.getInputShape(0), _info.getInputStrides(0)));
            
            size_t target_idx = _info.getInputContiguous()[1] ? i : 
                (_info.getInputBroadcasted()[1] ? 
                    op::common_cpu::indexToReducedOffset(i, _info.getNdim(), _info.getOutputStrides(), _info.getInputStrides(1)) : 
                    op::common_cpu::indexToOffset(i, _info.getNdim(), _info.getInputShape(1), _info.getInputStrides(1)));
            
            out_ptr[out_idx] = op(probs_ptr[probs_idx], target_ptr[target_idx]);
        }
        return INFINI_STATUS_SUCCESS;
    }
    case INFINI_DTYPE_F32: {
        auto* out_ptr = reinterpret_cast<float*>(output);
        auto* probs_ptr = reinterpret_cast<const float*>(inputs[0]);
        auto* target_ptr = reinterpret_cast<const float*>(inputs[1]);
        
        size_t output_size = _info.getOutputSize();
        #pragma omp parallel for
        for (size_t i = 0; i < output_size; ++i) {
            size_t out_idx = _info.isOutputContiguous() ? i : 
                op::common_cpu::indexToOffset(i, _info.getNdim(), _info.getOutputShape(), _info.getOutputStrides());
            
            size_t probs_idx = _info.getInputContiguous()[0] ? i : 
                (_info.getInputBroadcasted()[0] ? 
                    op::common_cpu::indexToReducedOffset(i, _info.getNdim(), _info.getOutputStrides(), _info.getInputStrides(0)) : 
                    op::common_cpu::indexToOffset(i, _info.getNdim(), _info.getInputShape(0), _info.getInputStrides(0)));
            
            size_t target_idx = _info.getInputContiguous()[1] ? i : 
                (_info.getInputBroadcasted()[1] ? 
                    op::common_cpu::indexToReducedOffset(i, _info.getNdim(), _info.getOutputStrides(), _info.getInputStrides(1)) : 
                    op::common_cpu::indexToOffset(i, _info.getNdim(), _info.getInputShape(1), _info.getInputStrides(1)));
            
            out_ptr[out_idx] = op(probs_ptr[probs_idx], target_ptr[target_idx]);
        }
        return INFINI_STATUS_SUCCESS;
    }
    case INFINI_DTYPE_BF16: {
        auto* out_ptr = reinterpret_cast<bf16_t*>(output);
        auto* probs_ptr = reinterpret_cast<const bf16_t*>(inputs[0]);
        auto* target_ptr = reinterpret_cast<const bf16_t*>(inputs[1]);
        
        size_t output_size = _info.getOutputSize();
        #pragma omp parallel for
        for (size_t i = 0; i < output_size; ++i) {
            size_t out_idx = _info.isOutputContiguous() ? i : 
                op::common_cpu::indexToOffset(i, _info.getNdim(), _info.getOutputShape(), _info.getOutputStrides());
            
            size_t probs_idx = _info.getInputContiguous()[0] ? i : 
                (_info.getInputBroadcasted()[0] ? 
                    op::common_cpu::indexToReducedOffset(i, _info.getNdim(), _info.getOutputStrides(), _info.getInputStrides(0)) : 
                    op::common_cpu::indexToOffset(i, _info.getNdim(), _info.getInputShape(0), _info.getInputStrides(0)));
            
            size_t target_idx = _info.getInputContiguous()[1] ? i : 
                (_info.getInputBroadcasted()[1] ? 
                    op::common_cpu::indexToReducedOffset(i, _info.getNdim(), _info.getOutputStrides(), _info.getInputStrides(1)) : 
                    op::common_cpu::indexToOffset(i, _info.getNdim(), _info.getInputShape(1), _info.getInputStrides(1)));
            
            out_ptr[out_idx] = op(probs_ptr[probs_idx], target_ptr[target_idx]);
        }
        return INFINI_STATUS_SUCCESS;
    }
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::crossentropyloss_backward::cpu