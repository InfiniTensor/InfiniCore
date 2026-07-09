#ifndef DSV4_FLASH_MLA_METADATA_INFO_H
#define DSV4_FLASH_MLA_METADATA_INFO_H

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::dsv4_flash_mla_metadata {

struct Info {
    size_t batch;
    size_t tile_meta_rows;
    int num_heads_per_head_k;
    int num_heads_k;
};

inline infiniStatus_t createInfo(Info *info,
                                 infiniopTensorDescriptor_t cache_seqlens_desc,
                                 infiniopTensorDescriptor_t tile_scheduler_metadata_desc,
                                 infiniopTensorDescriptor_t num_splits_desc,
                                 int num_heads_per_head_k,
                                 int num_heads_k) {
    CHECK_OR_RETURN(info != nullptr && cache_seqlens_desc != nullptr && tile_scheduler_metadata_desc != nullptr && num_splits_desc != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(cache_seqlens_desc->dtype() == INFINI_DTYPE_I32 && tile_scheduler_metadata_desc->dtype() == INFINI_DTYPE_I32 && num_splits_desc->dtype() == INFINI_DTYPE_I32, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(cache_seqlens_desc->ndim() == 1 && tile_scheduler_metadata_desc->ndim() == 2 && num_splits_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(cache_seqlens_desc->isContiguous() && tile_scheduler_metadata_desc->isContiguous() && num_splits_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(num_heads_per_head_k > 0 && num_heads_k > 0, INFINI_STATUS_BAD_PARAM);

    size_t batch = cache_seqlens_desc->dim(0);
    CHECK_OR_RETURN(batch > 0, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(tile_scheduler_metadata_desc->dim(0) > 0 && tile_scheduler_metadata_desc->dim(1) == 8, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(num_splits_desc->dim(0) == batch + 1, INFINI_STATUS_BAD_TENSOR_SHAPE);

    *info = Info{batch, tile_scheduler_metadata_desc->dim(0), num_heads_per_head_k, num_heads_k};
    return INFINI_STATUS_SUCCESS;
}

} // namespace op::dsv4_flash_mla_metadata

#endif
