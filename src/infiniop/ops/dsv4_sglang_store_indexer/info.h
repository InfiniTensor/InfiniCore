#ifndef DSV4_SGLANG_STORE_INDEXER_INFO_H
#define DSV4_SGLANG_STORE_INDEXER_INFO_H
#include "../../../utils.h"
#include "../../tensor.h"
namespace op::dsv4_sglang_store_indexer {
struct Info {
    size_t tokens;
    size_t head_dim;
    size_t cache_rows;
    size_t cache_cols;
};
inline infiniStatus_t createInfo(Info *info, infiniopTensorDescriptor_t input_desc, infiniopTensorDescriptor_t cache_desc, infiniopTensorDescriptor_t indices_desc) {
    CHECK_OR_RETURN(info && input_desc && cache_desc && indices_desc, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(input_desc->dtype() == INFINI_DTYPE_BF16 && cache_desc->dtype() == INFINI_DTYPE_U8 && indices_desc->dtype() == INFINI_DTYPE_I32, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(input_desc->ndim() == 2 && cache_desc->ndim() == 2 && indices_desc->ndim() == 1, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(input_desc->isContiguous() && cache_desc->isContiguous() && indices_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(indices_desc->dim(0) == input_desc->dim(0) && input_desc->dim(1) > 0 && cache_desc->dim(0) > 0 && cache_desc->dim(1) > 0, INFINI_STATUS_BAD_TENSOR_SHAPE);
    *info = Info{input_desc->dim(0), input_desc->dim(1), cache_desc->dim(0), cache_desc->dim(1)};
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::dsv4_sglang_store_indexer
#endif
