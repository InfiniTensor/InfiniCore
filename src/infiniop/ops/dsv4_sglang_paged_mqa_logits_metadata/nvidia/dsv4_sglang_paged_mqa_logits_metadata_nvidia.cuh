#ifndef DSV4_SGLANG_PAGED_MQA_LOGITS_METADATA_NVIDIA_CUH
#define DSV4_SGLANG_PAGED_MQA_LOGITS_METADATA_NVIDIA_CUH

#include "../dsv4_sglang_paged_mqa_logits_metadata.h"

namespace op::dsv4_sglang_paged_mqa_logits_metadata::nvidia {

class Descriptor final : public op::dsv4_sglang_paged_mqa_logits_metadata::Descriptor {
public:
    using op::dsv4_sglang_paged_mqa_logits_metadata::Descriptor::Descriptor;

    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t seq_lens_desc, infiniopTensorDescriptor_t metadata_desc);

    infiniStatus_t calculate(void *workspace, size_t workspace_size, const void *seq_lens, void *metadata, void *stream) const;
};

} // namespace op::dsv4_sglang_paged_mqa_logits_metadata::nvidia

#endif
