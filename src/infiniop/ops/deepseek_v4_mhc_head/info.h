#ifndef __DEEPSEEK_V4_MHC_HEAD_INFO_H__
#define __DEEPSEEK_V4_MHC_HEAD_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

#include <cstddef>

namespace op::deepseek_v4_mhc_head {

inline bool is_contiguous(infiniopTensorDescriptor_t desc) {
    ptrdiff_t expected = 1;
    for (ptrdiff_t i = static_cast<ptrdiff_t>(desc->ndim()) - 1; i >= 0; --i) {
        if (desc->stride(static_cast<size_t>(i)) != expected) {
            return false;
        }
        expected *= static_cast<ptrdiff_t>(desc->dim(static_cast<size_t>(i)));
    }
    return true;
}

inline bool is_activation_dtype(infiniDtype_t dtype) {
    return dtype == INFINI_DTYPE_F16 || dtype == INFINI_DTYPE_BF16 || dtype == INFINI_DTYPE_F32;
}

struct DeepseekV4MHCHeadCollapseInfo {
    size_t batch_size;
    size_t seq_len;
    size_t hc_mult;
    size_t hidden_size;
    size_t token_count;
    float epsilon;
    infiniDtype_t dtype;
    infiniDtype_t mixes_dtype;

    static utils::Result<DeepseekV4MHCHeadCollapseInfo> create(
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t mixes_desc,
        infiniopTensorDescriptor_t base_desc,
        infiniopTensorDescriptor_t scale_desc,
        float epsilon) {
        const auto dtype = x_desc->dtype();
        const auto mixes_dtype = mixes_desc->dtype();
        if (!is_activation_dtype(dtype) || !is_activation_dtype(mixes_dtype)
            || y_desc->dtype() != dtype || base_desc->dtype() != INFINI_DTYPE_F32
            || scale_desc->dtype() != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (x_desc->ndim() != 4 || y_desc->ndim() != 3 || mixes_desc->ndim() != 3
            || base_desc->ndim() != 1 || scale_desc->ndim() != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        const size_t batch_size = x_desc->shape()[0];
        const size_t seq_len = x_desc->shape()[1];
        const size_t hc_mult = x_desc->shape()[2];
        const size_t hidden_size = x_desc->shape()[3];
        if (batch_size == 0 || seq_len == 0 || hc_mult == 0 || hidden_size == 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (y_desc->shape()[0] != batch_size || y_desc->shape()[1] != seq_len
            || y_desc->shape()[2] != hidden_size
            || mixes_desc->shape()[0] != batch_size || mixes_desc->shape()[1] != seq_len
            || mixes_desc->shape()[2] != hc_mult
            || base_desc->shape()[0] != hc_mult || scale_desc->shape()[0] < 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (!is_contiguous(y_desc) || !is_contiguous(x_desc) || !is_contiguous(mixes_desc)
            || !is_contiguous(base_desc) || !is_contiguous(scale_desc)) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        return utils::Result<DeepseekV4MHCHeadCollapseInfo>(DeepseekV4MHCHeadCollapseInfo{
            batch_size,
            seq_len,
            hc_mult,
            hidden_size,
            batch_size * seq_len,
            epsilon,
            dtype,
            mixes_dtype,
        });
    }
};

} // namespace op::deepseek_v4_mhc_head

#endif
