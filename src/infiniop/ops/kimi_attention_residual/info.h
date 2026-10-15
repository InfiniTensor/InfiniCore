#ifndef __KIMI_ATTENTION_RESIDUAL_INFO_H__
#define __KIMI_ATTENTION_RESIDUAL_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

#include <vector>

namespace op::kimi_attention_residual {

class KimiAttentionResidualInfo {
    KimiAttentionResidualInfo() = default;

public:
    infiniDtype_t dtype;
    size_t num_tokens;
    size_t num_sources;
    size_t hidden_size;
    float epsilon;
    std::vector<ptrdiff_t> output_strides;
    std::vector<ptrdiff_t> values_strides;
    std::vector<ptrdiff_t> norm_weight_strides;
    std::vector<ptrdiff_t> projection_weight_strides;

    static utils::Result<KimiAttentionResidualInfo> create(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t values_desc,
        infiniopTensorDescriptor_t norm_weight_desc,
        infiniopTensorDescriptor_t projection_weight_desc,
        float epsilon) {
        if (output_desc == nullptr || values_desc == nullptr
            || norm_weight_desc == nullptr || projection_weight_desc == nullptr) {
            return INFINI_STATUS_NULL_POINTER;
        }
        const auto dtype = values_desc->dtype();
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);
        if (output_desc->dtype() != dtype || norm_weight_desc->dtype() != dtype
            || projection_weight_desc->dtype() != dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (values_desc->ndim() != 3 || output_desc->ndim() != 2
            || norm_weight_desc->ndim() != 1
            || (projection_weight_desc->ndim() != 1
                && projection_weight_desc->ndim() != 2)) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        const size_t num_tokens = values_desc->dim(0);
        const size_t num_sources = values_desc->dim(1);
        const size_t hidden_size = values_desc->dim(2);
        if (num_sources == 0 || num_sources > 16 || hidden_size == 0
            || output_desc->dim(0) != num_tokens
            || output_desc->dim(1) != hidden_size
            || norm_weight_desc->dim(0) != hidden_size) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (projection_weight_desc->ndim() == 1) {
            if (projection_weight_desc->dim(0) != hidden_size) {
                return INFINI_STATUS_BAD_TENSOR_SHAPE;
            }
        } else if (projection_weight_desc->dim(0) != 1
                   || projection_weight_desc->dim(1) != hidden_size) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        if (output_desc->stride(1) != 1 || values_desc->stride(2) != 1
            || norm_weight_desc->stride(0) != 1
            || projection_weight_desc->stride(projection_weight_desc->ndim() - 1) != 1) {
            return INFINI_STATUS_BAD_TENSOR_STRIDES;
        }

        return utils::Result<KimiAttentionResidualInfo>(KimiAttentionResidualInfo{
            dtype,
            num_tokens,
            num_sources,
            hidden_size,
            epsilon,
            output_desc->strides(),
            values_desc->strides(),
            norm_weight_desc->strides(),
            projection_weight_desc->strides(),
        });
    }
};

} // namespace op::kimi_attention_residual

#endif
