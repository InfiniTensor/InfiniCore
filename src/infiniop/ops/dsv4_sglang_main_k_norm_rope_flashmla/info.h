#ifndef DSV4_SGLANG_MAIN_K_NORM_ROPE_FLASHMLA_INFO_H
#define DSV4_SGLANG_MAIN_K_NORM_ROPE_FLASHMLA_INFO_H
#include "../../../utils.h"
#include "../../tensor.h"
namespace op::dsv4_sglang_main_k_norm_rope_flashmla {
struct Info {
    size_t tokens;
    size_t freqs_rows;
    size_t cache_rows;
    size_t cache_cols;
    double eps;
};
inline infiniStatus_t createInfo(Info *info, infiniopTensorDescriptor_t kv_desc, infiniopTensorDescriptor_t weight_desc, infiniopTensorDescriptor_t freqs_desc, infiniopTensorDescriptor_t positions_desc, infiniopTensorDescriptor_t out_loc_desc, infiniopTensorDescriptor_t cache_desc, double eps) {
    CHECK_OR_RETURN(info && kv_desc && weight_desc && freqs_desc && positions_desc && out_loc_desc && cache_desc, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(kv_desc->dtype() == INFINI_DTYPE_BF16 && weight_desc->dtype() == INFINI_DTYPE_BF16 && freqs_desc->dtype() == INFINI_DTYPE_F32 && positions_desc->dtype() == INFINI_DTYPE_I32 && out_loc_desc->dtype() == INFINI_DTYPE_I32 && cache_desc->dtype() == INFINI_DTYPE_U8, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(kv_desc->ndim() == 2 && weight_desc->ndim() == 1 && freqs_desc->ndim() == 2 && positions_desc->ndim() == 1 && out_loc_desc->ndim() == 1 && cache_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(kv_desc->isContiguous() && weight_desc->isContiguous() && freqs_desc->isContiguous() && positions_desc->isContiguous() && out_loc_desc->isContiguous() && cache_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(kv_desc->dim(1) == 512 && weight_desc->dim(0) == 512 && freqs_desc->dim(1) == 64 && positions_desc->dim(0) == kv_desc->dim(0) && out_loc_desc->dim(0) == kv_desc->dim(0), INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(eps >= 0.0, INFINI_STATUS_BAD_PARAM);
    *info = Info{kv_desc->dim(0), freqs_desc->dim(0), cache_desc->dim(0), cache_desc->dim(1), eps};
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::dsv4_sglang_main_k_norm_rope_flashmla
#endif
