#ifndef DSV4_SGLANG_SILU_AND_MUL_CLAMP_NVIDIA_CUH
#define DSV4_SGLANG_SILU_AND_MUL_CLAMP_NVIDIA_CUH

#include "../dsv4_sglang_silu_and_mul_clamp.h"

namespace op::dsv4_sglang_silu_and_mul_clamp::nvidia {

class Descriptor final : public op::dsv4_sglang_silu_and_mul_clamp::Descriptor {
public:
    using op::dsv4_sglang_silu_and_mul_clamp::Descriptor::Descriptor;

    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t output_desc, infiniopTensorDescriptor_t input_desc, double limit);

    infiniStatus_t calculate(void *workspace, size_t workspace_size, void *output, const void *input, void *stream) const;
};

} // namespace op::dsv4_sglang_silu_and_mul_clamp::nvidia

#endif
