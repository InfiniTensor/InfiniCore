#ifndef DSV4_SGLANG_MASK_TOPK_IDS_NVIDIA_CUH
#define DSV4_SGLANG_MASK_TOPK_IDS_NVIDIA_CUH

#include "../dsv4_sglang_mask_topk_ids.h"

namespace op::dsv4_sglang_mask_topk_ids::nvidia {

class Descriptor final : public op::dsv4_sglang_mask_topk_ids::Descriptor {
public:
    using op::dsv4_sglang_mask_topk_ids::Descriptor::Descriptor;

    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t topk_ids_desc, infiniopTensorDescriptor_t num_token_non_padded_desc);

    infiniStatus_t calculate(void *workspace, size_t workspace_size, void *topk_ids, const void *num_token_non_padded, void *stream) const;
};

} // namespace op::dsv4_sglang_mask_topk_ids::nvidia

#endif
