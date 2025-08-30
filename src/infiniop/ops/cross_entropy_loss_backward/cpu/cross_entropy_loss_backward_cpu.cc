#include "cross_entropy_loss_backward_cpu.h"

namespace op::cross_entropy_loss_backward::cpu {

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

    const auto &probs_desc = input_desc_vec.at(0);
    const auto &target_desc = input_desc_vec.at(1);
    CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_F32, INFINI_DTYPE_BF16);
    CHECK_SAME_SHAPE(out_desc->shape(), probs_desc->shape(), target_desc->shape());
    size_t batch_size = 1;
    for (size_t d = 0; d < out_desc->ndim() - 1; d++)
        batch_size *= out_desc->dim(d);

    // create CPU elementwise descriptor
    CREATE_CROSS_ENTROPY_LOSS_BACKWARD_ELEMENTWISE_CPU_DESCRIPTOR(handle, dtype, out_desc, input_desc_vec, batch_size);
//  --------------------------- end: check data type and input validity ----------------------------

    return INFINI_STATUS_SUCCESS;
}
/*
template<typename Tdata, size_t... Is, typename... Args>
infiniStatus_t calculate_cross_entropy_loss_backward(
    const op::elementwise::ElementwiseInfo &info,
    void *output,
    const std::vector<const void *> &inputs,
    size_t batch_size,
    std::index_sequence<Is...>
) {

    Tdata *out = reinterpret_cast<Tdata *>(output);
    std::array<const Tdata *, sizeof...(Is)> ins = {reinterpret_cast<const Tdata *>(inputs[Is])...};
    const ptrdiff_t output_size = info.getOutputSize();

#pragma omp parallel for
    for (ptrdiff_t i = 0; i < output_size; ++i) {
        size_t out_idx = info.isOutputContiguous()
                           ? i
                           : op::common_cpu::indexToOffset(i, info.getNdim(), info.getOutputShape(), info.getOutputStrides());

        auto get_input_idx = [&](size_t input_id) {
            return info.getInputContiguous()[input_id]
                     ? i
                     : (info.getInputBroadcasted()[input_id]
                            ? op::common_cpu::indexToReducedOffset(i, info.getNdim(), info.getOutputStrides(), info.getInputStrides(input_id))
                            : op::common_cpu::indexToOffset(i, info.getNdim(), info.getInputShape(input_id), info.getInputStrides(input_id)));
        };

        if constexpr (std::is_same_v<Tdata, fp16_t> || std::is_same_v<Tdata, bf16_t>) {
            out[out_idx] = utils::cast<Tdata>(CrossEntropyLossBackwardOp(batch_size)(utils::cast<float>(ins[Is][get_input_idx(Is)])...));
        } else {
            out[out_idx] = CrossEntropyLossBackwardOp(batch_size)(ins[Is][get_input_idx(Is)]...);
        }
    }
    return INFINI_STATUS_SUCCESS;
}
*/
infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *output,
    std::vector<const void *> inputs,
    void *stream) const {

    // size_t batch_size = (reinterpret_cast<const CrossEntropyLossBackInfo*>(_info))->
    // info
    // (static_cast<const CrossEntropyLossBackInfo *>(_info))->
    #define CALCULATE_CROSS_ENTROPY_LOSS_BACKWARD_OP(TDATA) \
        return _device_info->calculate<CrossEntropyLossBackwardOp, TDATA>(_info, output, inputs, stream, _batch_size)
    /*
    // #define CALCULATE_CROSS_ENTROPY_LOSS_BACKWARD_OP(TDATA) \
        // return calculate_cross_entropy_loss_backward<TDATA>(_info, output, inputs, _batch_size, std::make_index_sequence<2>{});
    */    
    switch (_dtype) {
    case INFINI_DTYPE_F16:
        CALCULATE_CROSS_ENTROPY_LOSS_BACKWARD_OP(fp16_t);
    case INFINI_DTYPE_F32:
        CALCULATE_CROSS_ENTROPY_LOSS_BACKWARD_OP(float);
    case INFINI_DTYPE_BF16:
        CALCULATE_CROSS_ENTROPY_LOSS_BACKWARD_OP(bf16_t);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
/*
    #define CALCULATE_CROSS_ENTROPY_LOSS_BACKWARD_OP(TDATA) \
        return _device_info->calculate<CrossEntropyLossBackwardOp, TDATA>(_info, output, inputs, stream)
    switch (_dtype) {
    case INFINI_DTYPE_F16:
        CALCULATE_CROSS_ENTROPY_LOSS_BACKWARD_OP(fp16_t);
    case INFINI_DTYPE_F32:
        CALCULATE_CROSS_ENTROPY_LOSS_BACKWARD_OP(float);
    case INFINI_DTYPE_BF16:
        CALCULATE_CROSS_ENTROPY_LOSS_BACKWARD_OP(bf16_t);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
*/
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::cross_entropy_loss_backward::cpu
