#ifndef __GPTQ_QYBLAS_GEMM_INFO_H__
#define __GPTQ_QYBLAS_GEMM_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <optional>
#include <vector>

namespace op::gptq_qyblas_gemm {

class GptqQyblasGemmInfo {
    GptqQyblasGemmInfo() = default;

public:
    infiniDtype_t dtype, weight_dtype, scales_dtype, zeros_dtype;
    size_t M, K, N, scales_size_0, scales_size_1;
    ptrdiff_t lda, ldb, result_ld;
    bool transpose_mat_1, transpose_mat_2, transpose_result;

    static utils::Result<GptqQyblasGemmInfo> createGptqQyblasGemmInfo(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc,
        infiniopTensorDescriptor_t b_scales_desc,
        infiniopTensorDescriptor_t b_zeros_desc) {

        auto dtype = a_desc->dtype();

        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16);
        CHECK_DTYPE(dtype, out_desc->dtype());

        const infiniDtype_t weight_dtype = b_desc->dtype();
        CHECK_DTYPE(weight_dtype, INFINI_DTYPE_F8, INFINI_DTYPE_U8, INFINI_DTYPE_I8);

        const infiniDtype_t scales_dtype = b_scales_desc->dtype();
        const infiniDtype_t zeros_dtype = b_zeros_desc->dtype();

        size_t M = out_desc->shape()[0];
        size_t N = out_desc->shape()[1];
        size_t K = a_desc->shape()[1];

        size_t scales_size_0 = b_scales_desc->shape()[0];
        size_t scales_size_1 = b_scales_desc->shape()[1];

        auto ndim = out_desc->ndim();
        CHECK_OR_RETURN(ndim == 2
                            && a_desc->ndim() == ndim
                            && b_desc->ndim() == ndim
                            && b_scales_desc->ndim() == ndim
                            && b_zeros_desc->ndim() == ndim,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        bool transpose_result = false;
        if (out_desc->strides()[0] == 1 && out_desc->strides()[1] >= std::max<int64_t>(1, out_desc->shape()[0])) {
            transpose_result = true;
        } else if (out_desc->strides()[1] == 1 && out_desc->strides()[0] >= std::max<int64_t>(1, out_desc->shape()[1])) {
            transpose_result = false;
        } else {
            transpose_result = false;
        }
        bool transpose_mat_1 = false;
        if (a_desc->strides()[0] == 1 && a_desc->strides()[1] >= std::max<int64_t>(1, a_desc->shape()[0])) {
            transpose_mat_1 = true;
        } else if (a_desc->strides()[1] == 1 && a_desc->strides()[0] >= std::max<int64_t>(1, a_desc->shape()[1])) {
            transpose_mat_1 = false;
        } else {
            transpose_mat_1 = false;
        }
        bool transpose_mat_2 = false;
        if (b_desc->strides()[0] == 1 && b_desc->strides()[1] >= std::max<int64_t>(1, b_desc->shape()[0])) {
            transpose_mat_2 = true;
        } else if (b_desc->strides()[1] == 1 && b_desc->strides()[0] >= std::max<int64_t>(1, b_desc->shape()[1])) {
            transpose_mat_2 = false;
        } else {
            transpose_mat_2 = false;
        }

        ptrdiff_t lda = a_desc->strides()[transpose_mat_1 ? 1 : 0];
        ptrdiff_t ldb = b_desc->strides()[transpose_mat_2 ? 1 : 0];
        ptrdiff_t result_ld = out_desc->strides()[transpose_result ? 1 : 0];

        return utils::Result<GptqQyblasGemmInfo>(GptqQyblasGemmInfo{
            dtype, weight_dtype, scales_dtype, zeros_dtype,
            M, K, N, scales_size_0, scales_size_1,
            lda, ldb, result_ld,
            transpose_mat_1, transpose_mat_2, transpose_result});
    }
};

} // namespace op::gptq_qyblas_gemm

#endif // __GPTQ_QYBLAS_GEMM_INFO_H__
