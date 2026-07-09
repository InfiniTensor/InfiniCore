#ifndef DSV4_SWA_PREFILL_INDICES_INFO_H
#define DSV4_SWA_PREFILL_INDICES_INFO_H
#include "../../../utils.h"
#include "../../tensor.h"
namespace op::dsv4_swa_prefill_indices {
struct Info {
    size_t batch, seq_len;
    int window_size;
};
inline infiniStatus_t createInfo(Info *info, infiniopTensorDescriptor_t indices, int window_size) {
    CHECK_OR_RETURN(info && indices, INFINI_STATUS_NULL_POINTER);
    CHECK_OR_RETURN(indices->dtype() == INFINI_DTYPE_I32, INFINI_STATUS_BAD_TENSOR_DTYPE);
    CHECK_OR_RETURN(indices->ndim() == 2 && indices->isContiguous(), INFINI_STATUS_BAD_TENSOR_SHAPE);
    CHECK_OR_RETURN(window_size > 0, INFINI_STATUS_BAD_PARAM);
    *info = Info{indices->dim(0), indices->dim(1), window_size};
    return INFINI_STATUS_SUCCESS;
}
} // namespace op::dsv4_swa_prefill_indices
#endif
