#ifndef __LINEAR_MXFP4_INFO_H__
#define __LINEAR_MXFP4_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::linear_mxfp4 {

class LinearMxfp4Info {
    LinearMxfp4Info() = default;

public:
    infiniDtype_t dtype;
    size_t M;
    size_t N;
    size_t K;
    float alpha;
    bool has_bias;

    static utils::Result<LinearMxfp4Info> create(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc,
        infiniopTensorDescriptor_t packed_weight_desc,
        infiniopTensorDescriptor_t weight_scale_desc,
        infiniopTensorDescriptor_t bias_desc,
        float alpha) {
        CHECK_OR_RETURN(output_desc != nullptr && input_desc != nullptr
                            && packed_weight_desc != nullptr && weight_scale_desc != nullptr,
                        INFINI_STATUS_NULL_POINTER);

        const auto dtype = input_desc->dtype();
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        CHECK_OR_RETURN(output_desc->dtype() == dtype,
                        INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(packed_weight_desc->dtype(), INFINI_DTYPE_U8);
        CHECK_DTYPE(weight_scale_desc->dtype(), INFINI_DTYPE_U8);
        if (bias_desc != nullptr) {
            CHECK_OR_RETURN(bias_desc->dtype() == dtype,
                            INFINI_STATUS_BAD_TENSOR_DTYPE);
        }

        CHECK_OR_RETURN(input_desc->ndim() >= 2
                            && output_desc->ndim() == input_desc->ndim()
                            && packed_weight_desc->ndim() == 2
                            && weight_scale_desc->ndim() == 2
                            && (bias_desc == nullptr || bias_desc->ndim() == 1),
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(input_desc->isContiguous() && output_desc->isContiguous()
                            && packed_weight_desc->isContiguous()
                            && weight_scale_desc->isContiguous()
                            && (bias_desc == nullptr || bias_desc->isContiguous()),
                        INFINI_STATUS_BAD_TENSOR_STRIDES);

        const size_t input_last = input_desc->ndim() - 1;
        const size_t output_last = output_desc->ndim() - 1;
        const size_t K = input_desc->dim(input_last);
        const size_t N = packed_weight_desc->dim(0);
        CHECK_OR_RETURN(K > 0 && K % 32 == 0 && N > 0,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(packed_weight_desc->dim(1) == K / 2
                            && weight_scale_desc->dim(0) == N
                            && weight_scale_desc->dim(1) == K / 32
                            && output_desc->dim(output_last) == N
                            && (bias_desc == nullptr || bias_desc->dim(0) == N),
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        for (size_t i = 0; i < input_last; ++i) {
            CHECK_OR_RETURN(input_desc->dim(i) == output_desc->dim(i),
                            INFINI_STATUS_BAD_TENSOR_SHAPE);
        }
        const size_t M = input_desc->numel() / K;
        CHECK_OR_RETURN(output_desc->numel() == M * N,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        LinearMxfp4Info info;
        info.dtype = dtype;
        info.M = M;
        info.N = N;
        info.K = K;
        info.alpha = alpha;
        info.has_bias = bias_desc != nullptr;
        return utils::Result<LinearMxfp4Info>(info);
    }
};

} // namespace op::linear_mxfp4

#endif
