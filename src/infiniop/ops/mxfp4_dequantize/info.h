#ifndef __MXFP4_DEQUANTIZE_INFO_H__
#define __MXFP4_DEQUANTIZE_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"

namespace op::mxfp4_dequantize {

class Mxfp4DequantizeInfo {
    Mxfp4DequantizeInfo() = default;

public:
    infiniDtype_t output_dtype;
    size_t rows;
    size_t logical_width;
    size_t packed_numel;

    static utils::Result<Mxfp4DequantizeInfo> create(
        infiniopTensorDescriptor_t out_desc,
        infiniopTensorDescriptor_t packed_desc,
        infiniopTensorDescriptor_t scales_desc) {
        CHECK_OR_RETURN(out_desc != nullptr && packed_desc != nullptr && scales_desc != nullptr,
                        INFINI_STATUS_NULL_POINTER);
        CHECK_DTYPE(out_desc->dtype(), INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        CHECK_DTYPE(packed_desc->dtype(), INFINI_DTYPE_U8);
        CHECK_DTYPE(scales_desc->dtype(), INFINI_DTYPE_U8);
        CHECK_OR_RETURN(out_desc->ndim() > 0 && out_desc->ndim() == packed_desc->ndim() && out_desc->ndim() == scales_desc->ndim(),
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(out_desc->isContiguous() && packed_desc->isContiguous() && scales_desc->isContiguous(),
                        INFINI_STATUS_BAD_TENSOR_STRIDES);

        const size_t last = out_desc->ndim() - 1;
        const size_t logical_width = out_desc->dim(last);
        CHECK_OR_RETURN(logical_width > 0 && logical_width % 32 == 0,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(packed_desc->dim(last) == logical_width / 2 && scales_desc->dim(last) == logical_width / 32,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        for (size_t i = 0; i < last; ++i) {
            CHECK_OR_RETURN(out_desc->dim(i) == packed_desc->dim(i) && out_desc->dim(i) == scales_desc->dim(i),
                            INFINI_STATUS_BAD_TENSOR_SHAPE);
        }

        const size_t rows = out_desc->numel() / logical_width;
        CHECK_OR_RETURN(packed_desc->numel() == rows * logical_width / 2 && scales_desc->numel() == rows * logical_width / 32,
                        INFINI_STATUS_BAD_TENSOR_SHAPE);

        return utils::Result<Mxfp4DequantizeInfo>(Mxfp4DequantizeInfo{
            out_desc->dtype(), rows, logical_width, packed_desc->numel()});
    }
};

} // namespace op::mxfp4_dequantize

#endif
