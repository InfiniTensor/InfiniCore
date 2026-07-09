#ifndef DSV4_FLASH_MLA_DECODE_NVIDIA_CUH
#define DSV4_FLASH_MLA_DECODE_NVIDIA_CUH

#include "../dsv4_flash_mla_decode.h"

namespace op::dsv4_flash_mla_decode::nvidia {

class Descriptor final : public op::dsv4_flash_mla_decode::Descriptor {
public:
    using op::dsv4_flash_mla_decode::Descriptor::Descriptor;

    static infiniStatus_t create(infiniopHandle_t handle, Descriptor **desc_ptr, infiniopTensorDescriptor_t out_desc, infiniopTensorDescriptor_t lse_desc, infiniopTensorDescriptor_t q_nope_desc, infiniopTensorDescriptor_t q_pe_desc, infiniopTensorDescriptor_t k_cache_desc, infiniopTensorDescriptor_t block_table_desc, infiniopTensorDescriptor_t cache_seqlens_desc, infiniopTensorDescriptor_t tile_scheduler_metadata_desc, infiniopTensorDescriptor_t num_splits_desc, int head_dim_v, float softmax_scale, bool causal);

    infiniStatus_t calculate(void *workspace, size_t workspace_size, void *out, void *lse, const void *q_nope, const void *q_pe, const void *k_cache, const void *block_table, const void *cache_seqlens, const void *tile_scheduler_metadata, const void *num_splits, void *stream) const;
};

} // namespace op::dsv4_flash_mla_decode::nvidia

#endif
