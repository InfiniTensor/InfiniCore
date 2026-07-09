#ifndef DSV4_SGLANG_PAGED_MQA_LOGITS_METADATA_INFO_H
#define DSV4_SGLANG_PAGED_MQA_LOGITS_METADATA_INFO_H
#include "../../../utils.h"
#include "../../tensor.h"
namespace op::dsv4_sglang_paged_mqa_logits_metadata {
struct Info {
    size_t batch;
    size_t metadata_rows;
};
inline infiniStatus_t createInfo(Info *info, infiniopTensorDescriptor_t seq_lens_desc, infiniopTensorDescriptor_t metadata_desc) {
    CHECK_OR_RETURN(info != nullptr && seq_lens_desc != nullptr && metadata_desc != nullptr, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(seq_lens_desc->dtype() == INFINI_DTYPE_I32 && metadata_desc->dtype() == INFINI_DTYPE_I32, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(seq_lens_desc->ndim() == 1 && metadata_desc->ndim() == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(seq_lens_desc->isContiguous() && metadata_desc->isContiguous(), INFINI_STATUS_BAD_TENSOR_STRIDES);
    CHECK_OR_RETURN(metadata_desc->dim(0) >= 1 && metadata_desc->dim(1) == 2, INFINI_STATUS_BAD_TENSOR_SHAPE);
    *info = Info{seq_lens_desc->dim(0), metadata_desc->dim(0)};
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::dsv4_sglang_paged_mqa_logits_metadata
#endif
