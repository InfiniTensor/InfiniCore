#ifndef __LINEAR_W4A8_INFO_H__
#define __LINEAR_W4A8_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::linear_w4a8 {

inline size_t alignWorkspace(size_t size) {
    constexpr size_t alignment = 256;
    return (size + alignment - 1) / alignment * alignment;
}

class LinearW4A8Info {
    LinearW4A8Info() = default;

public:
    infiniDtype_t dtype;
    size_t M;
    size_t N;
    size_t K;
    float alpha;
    bool has_bias;

    size_t quantizedInputBytes() const { return alignWorkspace(M * K); }
    size_t inputScaleBytes() const { return alignWorkspace(M * sizeof(float)); }
    size_t workspaceSize() const { return quantizedInputBytes() + inputScaleBytes(); }

    static utils::Result<LinearW4A8Info> create(
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
        CHECK_OR_RETURN(output_desc->dtype() == dtype
                            && packed_weight_desc->dtype() == INFINI_DTYPE_I8
                            && weight_scale_desc->dtype() == INFINI_DTYPE_F32,
                        INFINI_STATUS_BAD_TENSOR_DTYPE);
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
        CHECK_OR_RETURN(K > 0 && K % 2 == 0 && N > 0,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(packed_weight_desc->dim(1) == K / 2
                            && weight_scale_desc->dim(0) == N
                            && weight_scale_desc->dim(1) == 1
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
        return utils::Result<LinearW4A8Info>(
            LinearW4A8Info{dtype, M, N, K, alpha, bias_desc != nullptr});
    }
};

} // namespace op::linear_w4a8

#endif
