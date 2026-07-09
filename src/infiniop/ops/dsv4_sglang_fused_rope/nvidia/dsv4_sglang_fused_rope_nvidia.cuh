#ifndef DSV4_SGLANG_FUSED_ROPE_NVIDIA_CUH
#define DSV4_SGLANG_FUSED_ROPE_NVIDIA_CUH

#include "../dsv4_sglang_fused_rope.h"

namespace op::dsv4_sglang_fused_rope::nvidia {

class Descriptor final : public op::dsv4_sglang_fused_rope::Descriptor {
public:
    using op::dsv4_sglang_fused_rope::Descriptor::Descriptor;

    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t q_desc, infiniopTensorDescriptor_t freqs_cis_desc, infiniopTensorDescriptor_t positions_desc, bool inverse);

    infiniStatus_t calculate(void *workspace, size_t workspace_size, void *q, const void *freqs_cis, const void *positions, void *stream) const;
};

} // namespace op::dsv4_sglang_fused_rope::nvidia

#endif
