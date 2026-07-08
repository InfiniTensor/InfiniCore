#ifndef __DEEPSEEK_V4_MHC_POST_INFO_H__
#define __DEEPSEEK_V4_MHC_POST_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

#include <cstddef>

namespace op::deepseek_v4_mhc_post {

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

struct DeepseekV4MHCPostInfo {
    size_t batch_size;
    size_t seq_len;
    size_t token_count;
    size_t hc_mult;
    size_t hidden_size;
    infiniDtype_t dtype;
    infiniDtype_t coeff_dtype;

    static utils::Result<DeepseekV4MHCPostInfo> create(
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t new_x_desc,
        infiniopTensorDescriptor_t residual_desc,
        infiniopTensorDescriptor_t post_desc,
        infiniopTensorDescriptor_t comb_desc) {
        const auto dtype = new_x_desc->dtype();
        const auto coeff_dtype = post_desc->dtype();
        if (!is_activation_dtype(dtype) || !is_activation_dtype(coeff_dtype)
            || residual_desc->dtype() != dtype || y_desc->dtype() != dtype
            || comb_desc->dtype() != coeff_dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (new_x_desc->ndim() != 3 || residual_desc->ndim() != 4 || y_desc->ndim() != 4
            || post_desc->ndim() != 3 || comb_desc->ndim() != 4) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        const size_t batch_size = residual_desc->shape()[0];
        const size_t seq_len = residual_desc->shape()[1];
        const size_t hc_mult = residual_desc->shape()[2];
        const size_t hidden_size = residual_desc->shape()[3];
        if (batch_size == 0 || seq_len == 0 || hc_mult == 0 || hidden_size == 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (new_x_desc->shape()[0] != batch_size || new_x_desc->shape()[1] != seq_len
            || new_x_desc->shape()[2] != hidden_size
            || y_desc->shape()[0] != batch_size || y_desc->shape()[1] != seq_len
            || y_desc->shape()[2] != hc_mult || y_desc->shape()[3] != hidden_size
            || post_desc->shape()[0] != batch_size || post_desc->shape()[1] != seq_len
            || post_desc->shape()[2] != hc_mult
            || comb_desc->shape()[0] != batch_size || comb_desc->shape()[1] != seq_len
            || comb_desc->shape()[2] != hc_mult || comb_desc->shape()[3] != hc_mult) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (!is_contiguous(y_desc) || !is_contiguous(new_x_desc) || !is_contiguous(residual_desc)
            || !is_contiguous(post_desc) || !is_contiguous(comb_desc)) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        return utils::Result<DeepseekV4MHCPostInfo>(DeepseekV4MHCPostInfo{
            batch_size,
            seq_len,
            batch_size * seq_len,
            hc_mult,
            hidden_size,
            dtype,
            coeff_dtype,
        });
    }
};

} // namespace op::deepseek_v4_mhc_post

#endif
