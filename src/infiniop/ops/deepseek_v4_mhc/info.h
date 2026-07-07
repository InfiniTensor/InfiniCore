#ifndef __DEEPSEEK_V4_MHC_INFO_H__
#define __DEEPSEEK_V4_MHC_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

#include <cstddef>

namespace op::deepseek_v4_mhc {

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

struct DeepseekV4MHCParamsInfo {
    size_t batch_size;
    size_t seq_len;
    size_t hc_mult;
    size_t mix_hc;
    size_t token_count;
    size_t sinkhorn_iters;
    float epsilon;
    infiniDtype_t dtype;

    static utils::Result<DeepseekV4MHCParamsInfo> create(
        infiniopTensorDescriptor_t pre_desc,
        infiniopTensorDescriptor_t post_desc,
        infiniopTensorDescriptor_t comb_desc,
        infiniopTensorDescriptor_t mixes_desc,
        infiniopTensorDescriptor_t base_desc,
        infiniopTensorDescriptor_t scale_desc,
        size_t sinkhorn_iters,
        float epsilon) {
        const auto dtype = mixes_desc->dtype();
        if (!is_activation_dtype(dtype) || pre_desc->dtype() != dtype || post_desc->dtype() != dtype
            || comb_desc->dtype() != dtype || base_desc->dtype() != INFINI_DTYPE_F32
            || scale_desc->dtype() != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (pre_desc->ndim() != 3 || post_desc->ndim() != 3 || comb_desc->ndim() != 4
            || mixes_desc->ndim() != 3 || base_desc->ndim() != 1 || scale_desc->ndim() != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        const size_t batch_size = mixes_desc->shape()[0];
        const size_t seq_len = mixes_desc->shape()[1];
        const size_t mix_hc = mixes_desc->shape()[2];
        if (mix_hc == 0 || sinkhorn_iters == 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        size_t hc_mult = 0;
        for (size_t h = 1; h <= 16; ++h) {
            if ((2 + h) * h == mix_hc) {
                hc_mult = h;
                break;
            }
        }
        if (hc_mult == 0) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (pre_desc->shape()[0] != batch_size || pre_desc->shape()[1] != seq_len || pre_desc->shape()[2] != hc_mult
            || post_desc->shape()[0] != batch_size || post_desc->shape()[1] != seq_len || post_desc->shape()[2] != hc_mult
            || comb_desc->shape()[0] != batch_size || comb_desc->shape()[1] != seq_len || comb_desc->shape()[2] != hc_mult
            || comb_desc->shape()[3] != hc_mult || base_desc->shape()[0] != mix_hc || scale_desc->shape()[0] < 3) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (!is_contiguous(pre_desc) || !is_contiguous(post_desc) || !is_contiguous(comb_desc)
            || !is_contiguous(mixes_desc) || !is_contiguous(base_desc) || !is_contiguous(scale_desc)) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        return utils::Result<DeepseekV4MHCParamsInfo>(DeepseekV4MHCParamsInfo{
            batch_size,
            seq_len,
            hc_mult,
            mix_hc,
            batch_size * seq_len,
            sinkhorn_iters,
            epsilon,
            dtype,
        });
    }
};

} // namespace op::deepseek_v4_mhc

#endif
