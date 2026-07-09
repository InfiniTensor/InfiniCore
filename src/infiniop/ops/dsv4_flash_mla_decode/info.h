#ifndef DSV4_FLASH_MLA_DECODE_INFO_H
#define DSV4_FLASH_MLA_DECODE_INFO_H

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::dsv4_flash_mla_decode {

struct Info {
    infiniDtype_t dtype;
    size_t batch;
    size_t seqlen_q;
    size_t heads_q;
    size_t head_dim_nope;
    size_t head_dim_pe;
    size_t head_dim_v;
    size_t num_blocks;
    size_t page_block_size;
    size_t heads_kv;
    size_t max_blocks_per_seq;
    size_t tile_meta_rows;
    float softmax_scale;
    bool causal;
};

inline infiniStatus_t createInfo(Info *info,
                                 infiniopTensorDescriptor_t out_desc,
                                 infiniopTensorDescriptor_t lse_desc,
                                 infiniopTensorDescriptor_t q_nope_desc,
                                 infiniopTensorDescriptor_t q_pe_desc,
                                 infiniopTensorDescriptor_t k_cache_desc,
                                 infiniopTensorDescriptor_t block_table_desc,
                                 infiniopTensorDescriptor_t cache_seqlens_desc,
                                 infiniopTensorDescriptor_t tile_scheduler_metadata_desc,
                                 infiniopTensorDescriptor_t num_splits_desc,
                                 int head_dim_v,
                                 float softmax_scale,
                                 bool causal) {
    CHECK_OR_RETURN(info != nullptr && out_desc != nullptr && lse_desc != nullptr && q_nope_desc != nullptr && q_pe_desc != nullptr && k_cache_desc != nullptr && block_table_desc != nullptr && cache_seqlens_desc != nullptr && tile_scheduler_metadata_desc != nullptr && num_splits_desc != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_DTYPE(q_nope_desc->dtype(), INFINI_DTYPE_F16, INFINI_DTYPE_BF16);
    CHECK_OR_RETURN(q_pe_desc->dtype() == q_nope_desc->dtype() && k_cache_desc->dtype() == q_nope_desc->dtype() && out_desc->dtype() == q_nope_desc->dtype(), INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(lse_desc->dtype() == INFINI_DTYPE_F32, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(block_table_desc->dtype() == INFINI_DTYPE_I32 && cache_seqlens_desc->dtype() == INFINI_DTYPE_I32 && tile_scheduler_metadata_desc->dtype() == INFINI_DTYPE_I32 && num_splits_desc->dtype() == INFINI_DTYPE_I32, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(q_nope_desc->ndim() == 4 && q_pe_desc->ndim() == 4 && k_cache_desc->ndim() == 4 && out_desc->ndim() == 4, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(block_table_desc->ndim() == 2 && tile_scheduler_metadata_desc->ndim() == 2 && cache_seqlens_desc->ndim() == 1 && num_splits_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(q_nope_desc->isContiguous() && q_pe_desc->isContiguous() && k_cache_desc->isContiguous() && out_desc->isContiguous() && block_table_desc->isContiguous() && cache_seqlens_desc->isContiguous() && tile_scheduler_metadata_desc->isContiguous() && num_splits_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);

    size_t batch = q_nope_desc->dim(0);
    size_t seqlen_q = q_nope_desc->dim(1);
    size_t heads_q = q_nope_desc->dim(2);
    size_t head_dim_nope = q_nope_desc->dim(3);
    size_t head_dim_pe = q_pe_desc->dim(3);
    CHECK_OR_RETURN(batch > 0 && seqlen_q == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(q_pe_desc->dim(0) == batch && q_pe_desc->dim(1) == seqlen_q && q_pe_desc->dim(2) == heads_q, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(head_dim_nope == 512 && head_dim_pe == 64, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(head_dim_v > 0 && head_dim_v % 32 == 0, INFINI_STATUS_BAD_PARAM);
    CHECK_OR_RETURN(out_desc->dim(0) == batch && out_desc->dim(1) == seqlen_q && out_desc->dim(2) == heads_q && out_desc->dim(3) == static_cast<size_t>(head_dim_v), INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(lse_desc->ndim() == 3 && lse_desc->dim(0) == batch && lse_desc->dim(1) == heads_q && lse_desc->dim(2) == seqlen_q, INFINI_STATUS_BAD_TENSOR_SHAPE);

    size_t num_blocks = k_cache_desc->dim(0);
    size_t page_block_size = k_cache_desc->dim(1);
    size_t heads_kv = k_cache_desc->dim(2);
    CHECK_OR_RETURN(heads_kv == 1 && k_cache_desc->dim(3) == head_dim_nope + head_dim_pe, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(heads_q % heads_kv == 0, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(block_table_desc->dim(0) == batch, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(cache_seqlens_desc->dim(0) == batch, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(tile_scheduler_metadata_desc->dim(1) == 8, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(num_splits_desc->dim(0) == batch + 1, INFINI_STATUS_BAD_TENSOR_SHAPE);

    *info = Info{q_nope_desc->dtype(), batch, seqlen_q, heads_q, head_dim_nope, head_dim_pe, static_cast<size_t>(head_dim_v), num_blocks, page_block_size, heads_kv, block_table_desc->dim(1), tile_scheduler_metadata_desc->dim(0), softmax_scale, causal};
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::dsv4_flash_mla_decode

#endif
