#ifndef DSV4_SILU_MUL_MASKED_QUANT_NVIDIA_CUH
#define DSV4_SILU_MUL_MASKED_QUANT_NVIDIA_CUH

#include "../dsv4_silu_mul_masked_quant.h"

namespace op::dsv4_silu_mul_masked_quant::nvidia {

class Descriptor final : public op::dsv4_silu_mul_masked_quant::Descriptor {
public:
    using op::dsv4_silu_mul_masked_quant::Descriptor::Descriptor;
    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t q_desc, infiniopTensorDescriptor_t scale_desc, infiniopTensorDescriptor_t gate_desc, infiniopTensorDescriptor_t up_desc, infiniopTensorDescriptor_t mask_desc);
    infiniStatus_t calculate(void *workspace, size_t workspace_size, void *q, void *scale, const void *gate, const void *up, const void *mask, void *stream) const;
};

} // namespace op::dsv4_silu_mul_masked_quant::nvidia

#endif
