#ifndef __FUSED_GATED_DELTA_NET_GATING_INFO_H__
#define __FUSED_GATED_DELTA_NET_GATING_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

#include <vector>

namespace op::fused_gated_delta_net_gating {

class FusedGatedDeltaNetGatingInfo {
    FusedGatedDeltaNetGatingInfo() = default;

public:
    infiniDtype_t input_dtype;
    infiniDtype_t parameter_dtype;
    size_t batch_size;
    size_t seq_len;
    size_t hidden;
    std::vector<ptrdiff_t> g_strides;
    std::vector<ptrdiff_t> beta_output_strides;
    std::vector<ptrdiff_t> A_log_strides;
    std::vector<ptrdiff_t> a_strides;
    std::vector<ptrdiff_t> b_strides;
    std::vector<ptrdiff_t> dt_bias_strides;
    float beta;
    float threshold;

    size_t numel() const {
        return batch_size * seq_len * hidden;
    }

    static utils::Result<FusedGatedDeltaNetGatingInfo> create(
        infiniopTensorDescriptor_t g_desc,
        infiniopTensorDescriptor_t beta_output_desc,
        infiniopTensorDescriptor_t A_log_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc,
        infiniopTensorDescriptor_t dt_bias_desc,
        float beta,
        float threshold,
        bool allow_fp32_parameters = false) {

        if (g_desc->dtype() != INFINI_DTYPE_F32 || beta_output_desc->dtype() != INFINI_DTYPE_F32) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        auto input_dtype = a_desc->dtype();
        if (input_dtype != INFINI_DTYPE_F32 && input_dtype != INFINI_DTYPE_F16 && input_dtype != INFINI_DTYPE_BF16) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        if (b_desc->dtype() != input_dtype) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }
        auto parameter_dtype = A_log_desc->dtype();
        if (dt_bias_desc->dtype() != parameter_dtype
            || (parameter_dtype != input_dtype
                && !(allow_fp32_parameters && parameter_dtype == INFINI_DTYPE_F32))) {
            return INFINI_STATUS_BAD_TENSOR_DTYPE;
        }

        if (g_desc->ndim() != 3 || beta_output_desc->ndim() != 3 || a_desc->ndim() != 3 || b_desc->ndim() != 3) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }
        if (A_log_desc->ndim() != 1 || dt_bias_desc->ndim() != 1) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        const auto &shape = a_desc->shape();
        if (shape != b_desc->shape() || shape != g_desc->shape() || shape != beta_output_desc->shape()) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        size_t hidden = shape[2];
        if (A_log_desc->shape()[0] != hidden || dt_bias_desc->shape()[0] != hidden) {
            return INFINI_STATUS_BAD_TENSOR_SHAPE;
        }

        FusedGatedDeltaNetGatingInfo info;
        info.input_dtype = input_dtype;
        info.parameter_dtype = parameter_dtype;
        info.batch_size = shape[0];
        info.seq_len = shape[1];
        info.hidden = hidden;
        info.g_strides = g_desc->strides();
        info.beta_output_strides = beta_output_desc->strides();
        info.A_log_strides = A_log_desc->strides();
        info.a_strides = a_desc->strides();
        info.b_strides = b_desc->strides();
        info.dt_bias_strides = dt_bias_desc->strides();
        info.beta = beta;
        info.threshold = threshold;
        return utils::Result<FusedGatedDeltaNetGatingInfo>(info);
    }
};

} // namespace op::fused_gated_delta_net_gating

#endif
