#ifndef DSV4_SGLANG_MAIN_Q_NORM_ROPE_NVIDIA_CUH
#define DSV4_SGLANG_MAIN_Q_NORM_ROPE_NVIDIA_CUH

#include "../dsv4_sglang_main_q_norm_rope.h"

namespace op::dsv4_sglang_main_q_norm_rope::nvidia {

class Descriptor final : public op::dsv4_sglang_main_q_norm_rope::Descriptor {
public:
    using op::dsv4_sglang_main_q_norm_rope::Descriptor::Descriptor;

    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t output_desc, infiniopTensorDescriptor_t input_desc, infiniopTensorDescriptor_t freqs_desc, infiniopTensorDescriptor_t positions_desc, double eps);

    infiniStatus_t calculate(void *workspace, size_t workspace_size, void *output, const void *input, const void *freqs, const void *positions, void *stream) const;
};

} // namespace op::dsv4_sglang_main_q_norm_rope::nvidia

#endif
