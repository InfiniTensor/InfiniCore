#ifndef __AWQ_MARLIN_GEMM_INFO_H__
#define __AWQ_MARLIN_GEMM_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include "marlin/marlin.cuh"
#include <vector>

#include <cassert>

namespace op::awq_marlin_gemm {

class AwqMarlinGemmInfo {
    AwqMarlinGemmInfo() = default;

public:
    infiniDtype_t a_dtype, b_dtype, c_dtype, s_dtype;
    size_t size_m, size_k, size_n;
    int num_groups;
    size_t b_q_size_0, b_q_size_1, b_zeros_size_1;
    ptrdiff_t a_stride_0;

    static utils::Result<AwqMarlinGemmInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc,
        infiniopTensorDescriptor_t b_bias_desc,
        infiniopTensorDescriptor_t b_scales_desc,
        infiniopTensorDescriptor_t a_scales_desc,
        infiniopTensorDescriptor_t global_scales_desc,
        infiniopTensorDescriptor_t b_zeros_desc,
        infiniopTensorDescriptor_t g_idx_desc,
        infiniopTensorDescriptor_t perm_desc) {
        CHECK_OR_RETURN(
            out_desc != nullptr && a_desc != nullptr && b_desc != nullptr && b_scales_desc != nullptr,
            INFINI_STATUS_NULL_POINTER);
        const infiniDtype_t a_dtype = a_desc->dtype();
        const infiniDtype_t b_dtype = b_desc->dtype();
        const infiniDtype_t c_dtype = out_desc->dtype();
        const infiniDtype_t s_dtype = b_scales_desc->dtype();

        size_t size_m = a_desc->dim(0);
        size_t size_k = a_desc->dim(1);
        size_t size_n = out_desc->dim(1);

        int num_groups = static_cast<int>(b_scales_desc->dim(0));
        size_t b_q_size_0 = b_desc->dim(0);
        size_t b_q_size_1 = b_desc->dim(1);
        size_t b_zeros_size_1 = b_zeros_desc->dim(1);
        ptrdiff_t a_stride_0 = a_desc->strides()[0];

        return utils::Result<AwqMarlinGemmInfo>(
            AwqMarlinGemmInfo{a_dtype, b_dtype, c_dtype, s_dtype,
                              size_m, size_k, size_n,
                              num_groups, b_q_size_0, b_q_size_1, b_zeros_size_1, a_stride_0});
    }
};

} // namespace op::awq_marlin_gemm

#endif // __AWQ_MARLIN_GEMM_INFO_H__
