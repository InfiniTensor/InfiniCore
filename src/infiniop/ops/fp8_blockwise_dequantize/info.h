#ifndef __FP8_BLOCKWISE_DEQUANTIZE_INFO_H__
#define __FP8_BLOCKWISE_DEQUANTIZE_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::fp8_blockwise_dequantize {

class Fp8BlockwiseDequantizeInfo {
    Fp8BlockwiseDequantizeInfo() = default;

public:
    infiniDtype_t output_dtype;
    size_t rows;
    size_t cols;
    size_t block_rows;
    size_t block_cols;
    size_t scales_cols;

    static utils::Result<Fp8BlockwiseDequantizeInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t q_desc,
        infiniopTensorDescriptor_t scales_desc) {
        CHECK_OR_RETURN(out_desc != nullptr && q_desc != nullptr && scales_desc != nullptr,
                        INFINI_STATUS_NULL_POINTER);
        CHECK_DTYPE(out_desc->dtype(), INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        CHECK_DTYPE(q_desc->dtype(), INFINI_DTYPE_F8);
        CHECK_DTYPE(scales_desc->dtype(), INFINI_DTYPE_F32);
        CHECK_OR_RETURN(out_desc->ndim() == 2 && q_desc->ndim() == 2 && scales_desc->ndim() == 2,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(out_desc->isContiguous() && q_desc->isContiguous() && scales_desc->isContiguous(),
                        INFINI_STATUS_BAD_TENSOR_STRIDES);

        const size_t rows = q_desc->dim(0);
        const size_t cols = q_desc->dim(1);
        CHECK_OR_RETURN(rows > 0 && cols > 0 && out_desc->dim(0) == rows && out_desc->dim(1) == cols,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        const size_t scales_rows = scales_desc->dim(0);
        const size_t scales_cols = scales_desc->dim(1);
        CHECK_OR_RETURN(scales_rows > 0 && scales_cols > 0 && rows % scales_rows == 0 && cols % scales_cols == 0,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        return utils::Result<Fp8BlockwiseDequantizeInfo>(Fp8BlockwiseDequantizeInfo{
            out_desc->dtype(), rows, cols, rows / scales_rows, cols / scales_cols, scales_cols});
    }
};

} // namespace op::fp8_blockwise_dequantize

#endif
