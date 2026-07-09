#ifndef DSV4_SGLANG_TOPK_TRANSFORM_NVIDIA_CUH
#define DSV4_SGLANG_TOPK_TRANSFORM_NVIDIA_CUH

#include "../dsv4_sglang_topk_transform.h"

namespace op::dsv4_sglang_topk_transform::nvidia {

class Descriptor final : public op::dsv4_sglang_topk_transform::Descriptor {
public:
    using op::dsv4_sglang_topk_transform::Descriptor::Descriptor;

    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t scores_desc, infiniopTensorDescriptor_t seq_lens_desc, infiniopTensorDescriptor_t page_table_desc, infiniopTensorDescriptor_t page_indices_desc, infiniopTensorDescriptor_t raw_indices_desc, int64_t page_size);

    infiniStatus_t calculate(void *workspace, size_t workspace_size, const void *scores, const void *seq_lens, const void *page_table, void *page_indices, void *raw_indices, void *stream) const;
};

} // namespace op::dsv4_sglang_topk_transform::nvidia

#endif
