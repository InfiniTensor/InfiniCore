#include "paged_attention_ascend.h"
#include "../../../devices/ascend/common_ascend.h"

namespace op::paged_attention::ascend {

struct Descriptor::Opaque {};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_cache_desc,
    infiniopTensorDescriptor_t v_cache_desc,
    infiniopTensorDescriptor_t block_tables_desc,
    infiniopTensorDescriptor_t cache_lens_desc,
    const std::optional<infiniopTensorDescriptor_t> &alibi_slopes_desc,
    float scale) {
    auto info = PagedAttentionInfo::create(
        out_desc, q_desc, k_cache_desc, v_cache_desc,
        block_tables_desc, cache_lens_desc, alibi_slopes_desc, scale);
    CHECK_RESULT(info);

    auto handle_ascend = reinterpret_cast<device::ascend::Handle *>(handle);
    *desc_ptr = new Descriptor(
        new Opaque{},
        info.take(),
        0,
        handle_ascend->device,
        handle_ascend->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace, size_t workspace_size,
    void *out, const void *q, const void *k_cache, const void *v_cache,
    const void *block_tables, const void *cache_lens, const void *alibi_slopes,
    void *stream) const {
    (void)workspace;
    (void)workspace_size;

    return paged_attention_kernel_launch(
        out, q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes,
        _info.dtype,
        _info.index_dtype,
        _info.num_heads,
        _info.num_seqs,
        _info.num_kv_heads,
        _info.head_size,
        _info.scale,
        _info.max_num_blocks_per_seq,
        _info.page_block_size,
        _info.q_stride,
        _info.k_batch_stride,
        _info.k_row_stride,
        _info.k_head_stride,
        _info.v_batch_stride,
        _info.v_row_stride,
        _info.v_head_stride,
        _info.o_stride,
        _info.block_table_batch_stride,
        _info.cache_lens_stride,
        stream);
}

} // namespace op::paged_attention::ascend
