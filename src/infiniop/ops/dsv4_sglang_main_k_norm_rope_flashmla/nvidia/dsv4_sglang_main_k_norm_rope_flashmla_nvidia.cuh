#ifndef DSV4_SGLANG_MAIN_K_NORM_ROPE_FLASHMLA_NVIDIA_CUH
#define DSV4_SGLANG_MAIN_K_NORM_ROPE_FLASHMLA_NVIDIA_CUH
#include "../dsv4_sglang_main_k_norm_rope_flashmla.h"
namespace op::dsv4_sglang_main_k_norm_rope_flashmla::nvidia {
class Descriptor final : public op::dsv4_sglang_main_k_norm_rope_flashmla::Descriptor {
public:
    using op::dsv4_sglang_main_k_norm_rope_flashmla::Descriptor::Descriptor;
    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t kv_desc, infiniopTensorDescriptor_t weight_desc, infiniopTensorDescriptor_t freqs_desc, infiniopTensorDescriptor_t positions_desc, infiniopTensorDescriptor_t out_loc_desc, infiniopTensorDescriptor_t cache_desc, double eps);
    infiniStatus_t calculate(void *workspace, size_t workspace_size, void *kv, const void *weight, const void *freqs, const void *positions, const void *out_loc, void *cache, void *stream) const;
};
} // namespace op::dsv4_sglang_main_k_norm_rope_flashmla::nvidia
#endif
