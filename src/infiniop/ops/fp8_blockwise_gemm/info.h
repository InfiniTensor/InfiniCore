#ifndef __FP8_BLOCKWISE_GEMM_INFO_H__
#define __FP8_BLOCKWISE_GEMM_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::fp8_blockwise_gemm {

class Fp8BlockwiseGemmInfo {
    Fp8BlockwiseGemmInfo() = default;

public:
    infiniDtype_t dtype;
    size_t M, N, K;
    size_t block_n;
    size_t block_k;
    size_t scales_cols;

    static utils::Result<Fp8BlockwiseGemmInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t q_desc,
        infiniopTensorDescriptor_t scales_desc) {
        CHECK_OR_RETURN(out_desc != nullptr && a_desc != nullptr && q_desc != nullptr && scales_desc != nullptr,
                        INFINI_STATUS_NULL_POINTER);
        const infiniDtype_t dtype = out_desc->dtype();
        CHECK_DTYPE(dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        CHECK_OR_RETURN(a_desc->dtype() == dtype, INFINI_STATUS_BAD_TENSOR_DTYPE);
        CHECK_DTYPE(q_desc->dtype(), INFINI_DTYPE_F8);
        CHECK_DTYPE(scales_desc->dtype(), INFINI_DTYPE_F32);
        CHECK_OR_RETURN(out_desc->ndim() == 2 && a_desc->ndim() == 2 && q_desc->ndim() == 2 && scales_desc->ndim() == 2,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(out_desc->isContiguous() && a_desc->isContiguous() && q_desc->isContiguous() && scales_desc->isContiguous(),
                        INFINI_STATUS_BAD_TENSOR_STRIDES);

        const size_t M = out_desc->dim(0);
        const size_t N = out_desc->dim(1);
        const size_t K = a_desc->dim(1);
        CHECK_OR_RETURN(M > 0 && N > 0 && K > 0, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(a_desc->dim(0) == M && q_desc->dim(0) == N && q_desc->dim(1) == K,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        const size_t scales_rows = scales_desc->dim(0);
        const size_t scales_cols = scales_desc->dim(1);
        CHECK_OR_RETURN(scales_rows > 0 && scales_cols > 0 && N % scales_rows == 0 && K % scales_cols == 0,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        const size_t block_n = N / scales_rows;
        const size_t block_k = K / scales_cols;
        // The kernel addresses scales per 128-wide K chunk and requires
        // 4-byte vectorizable rows.
        CHECK_OR_RETURN(block_k % 128 == 0 && K % 128 == 0, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(block_n % 16 == 0 && K % 4 == 0, INFINI_STATUS_BAD_TENSOR_SHAPE);

        return utils::Result<Fp8BlockwiseGemmInfo>(Fp8BlockwiseGemmInfo{
            dtype, M, N, K, block_n, block_k, scales_cols});
    }
};

} // namespace op::fp8_blockwise_gemm

#endif
