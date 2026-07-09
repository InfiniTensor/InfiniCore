#ifndef DSV4_SGLANG_TOPK_V2_NVIDIA_CUH
#define DSV4_SGLANG_TOPK_V2_NVIDIA_CUH

#include "../dsv4_sglang_topk_v2.h"

namespace op::dsv4_sglang_topk_v2::nvidia {

class Descriptor final : public op::dsv4_sglang_topk_v2::Descriptor {
public:
    using op::dsv4_sglang_topk_v2::Descriptor::Descriptor;

    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t scores_desc, infiniopTensorDescriptor_t seq_lens_desc, infiniopTensorDescriptor_t page_table_desc, infiniopTensorDescriptor_t page_indices_desc, infiniopTensorDescriptor_t transform_workspace_desc, infiniopTensorDescriptor_t metadata_desc, int64_t page_size);

    infiniStatus_t calculate(void *workspace, size_t workspace_size, const void *scores, const void *seq_lens, const void *page_table, void *page_indices, void *transform_workspace, void *metadata, void *stream) const;
};

} // namespace op::dsv4_sglang_topk_v2::nvidia

#endif
