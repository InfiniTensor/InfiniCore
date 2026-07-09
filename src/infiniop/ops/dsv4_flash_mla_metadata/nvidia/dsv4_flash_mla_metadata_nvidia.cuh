#ifndef DSV4_FLASH_MLA_METADATA_NVIDIA_CUH
#define DSV4_FLASH_MLA_METADATA_NVIDIA_CUH

#include "../dsv4_flash_mla_metadata.h"

namespace op::dsv4_flash_mla_metadata::nvidia {

class Descriptor final : public op::dsv4_flash_mla_metadata::Descriptor {
public:
    using op::dsv4_flash_mla_metadata::Descriptor::Descriptor;

    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t cache_seqlens_desc, infiniopTensorDescriptor_t tile_scheduler_metadata_desc, infiniopTensorDescriptor_t num_splits_desc, int num_heads_per_head_k, int num_heads_k);

    infiniStatus_t calculate(void *workspace, size_t workspace_size, const void *cache_seqlens, void *tile_scheduler_metadata, void *num_splits, void *stream) const;
};

} // namespace op::dsv4_flash_mla_metadata::nvidia

#endif
