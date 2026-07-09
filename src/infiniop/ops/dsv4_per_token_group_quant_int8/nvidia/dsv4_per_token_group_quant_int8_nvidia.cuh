#ifndef DSV4_PER_TOKEN_GROUP_QUANT_INT8_NVIDIA_CUH
#define DSV4_PER_TOKEN_GROUP_QUANT_INT8_NVIDIA_CUH

#include "../dsv4_per_token_group_quant_int8.h"

namespace op::dsv4_per_token_group_quant_int8::nvidia {

class Descriptor final : public op::dsv4_per_token_group_quant_int8::Descriptor {
public:
    using op::dsv4_per_token_group_quant_int8::Descriptor::Descriptor;
    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t q_desc, infiniopTensorDescriptor_t scale_desc, infiniopTensorDescriptor_t x_desc, int group_size);
    infiniStatus_t calculate(void *workspace, size_t workspace_size, void *q, void *scale, const void *x, void *stream) const;
};

} // namespace op::dsv4_per_token_group_quant_int8::nvidia

#endif
