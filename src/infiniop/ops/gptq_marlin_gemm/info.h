#ifndef __GPTQ_MARLIN_GEMM_INFO_H__
#define __GPTQ_MARLIN_GEMM_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

#include <cassert>

namespace op::gptq_marlin_gemm {

class GptqMarlinGemmInfo {
    GptqMarlinGemmInfo() = default;

public:
    infiniDtype_t dtype;
    size_t M, K, N, b_q_size_1;
    int num_groups;
    ptrdiff_t a_stride_0;

    static utils::Result<GptqMarlinGemmInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc,
        infiniopTensorDescriptor_t b_scales_desc,
        infiniopTensorDescriptor_t global_scale_desc,
        infiniopTensorDescriptor_t b_zeros_desc,
        infiniopTensorDescriptor_t g_idx_desc,
        infiniopTensorDescriptor_t perm_desc) {
        CHECK_OR_RETURN(
            out_desc != nullptr && a_desc != nullptr && b_desc != nullptr && b_scales_desc != nullptr,
            INFINI_STATUS_NULL_POINTER);
        const infiniDtype_t dtype = a_desc->dtype();
        size_t M = out_desc->dim(0);
        size_t N = out_desc->dim(1);
        size_t K = a_desc->dim(1);
        size_t b_q_size_1 = b_desc->dim(1);
        int num_groups = static_cast<int>(b_scales_desc->dim(0));
        ptrdiff_t a_stride_0 = a_desc->strides()[0];

        auto ndim = out_desc->ndim();
        CHECK_OR_RETURN(ndim == 2
                            && a_desc->ndim() == ndim
                            && b_desc->ndim() == ndim
                            && b_scales_desc->ndim() == ndim,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        CHECK_OR_RETURN(b_scales_desc->shape()[1] == N
                            && a_stride_0 % 8 == 0,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        return utils::Result<GptqMarlinGemmInfo>(
            GptqMarlinGemmInfo{dtype, M, K, N, b_q_size_1, num_groups, a_stride_0});
    }
};

} // namespace op::gptq_marlin_gemm

#endif // __GPTQ_MARLIN_GEMM_INFO_H__
