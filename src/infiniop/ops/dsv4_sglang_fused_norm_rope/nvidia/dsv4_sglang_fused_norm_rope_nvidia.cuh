#ifndef DSV4_SGLANG_FUSED_NORM_ROPE_NVIDIA_CUH
#define DSV4_SGLANG_FUSED_NORM_ROPE_NVIDIA_CUH

#include "../dsv4_sglang_fused_norm_rope.h"

namespace op::dsv4_sglang_fused_norm_rope::nvidia {

class Descriptor final : public op::dsv4_sglang_fused_norm_rope::Descriptor {
public:
    using op::dsv4_sglang_fused_norm_rope::Descriptor::Descriptor;

    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t kv_desc, infiniopTensorDescriptor_t weight_desc, infiniopTensorDescriptor_t positions_desc, infiniopTensorDescriptor_t freqs_desc, double eps);

    infiniStatus_t calculate(void *workspace, size_t workspace_size, void *kv, const void *weight, const void *positions, const void *freqs, void *stream) const;
};

} // namespace op::dsv4_sglang_fused_norm_rope::nvidia

#endif
