

#ifndef __PIXEL_SHUFFLE_INFO_H__
#define __PIXEL_SHUFFLE_INFO_H__

#include "../../../utils.h"
#include "../../operator.h"
#include "../../tensor.h"

namespace op::pixel_shuffle {

class PixelShuffleInfo {
private:
    PixelShuffleInfo() = default;

public:
    infiniDtype_t dtype;
    int r;

    int B;
    int C_out;
    int H_out;
    int W_out;

    // strides in ELEMENTS
    ptrdiff_t input_b_stride;
    ptrdiff_t input_c_stride;
    ptrdiff_t input_h_stride;
    ptrdiff_t input_w_stride;

    ptrdiff_t output_b_stride;
    ptrdiff_t output_c_stride;
    ptrdiff_t output_h_stride;
    ptrdiff_t output_w_stride;

    static utils::Result<PixelShuffleInfo> createPixelShuffleInfo(
        infiniopTensorDescriptor_t output_desc,
        infiniopTensorDescriptor_t input_desc,
        int upscale_factor)
    {
        CHECK_OR_RETURN(upscale_factor > 0, INFINI_STATUS_BAD_PARAM);

        size_t ndim = input_desc->ndim();
        CHECK_OR_RETURN(ndim >= 3, INFINI_STATUS_BAD_TENSOR_SHAPE);

        size_t input_c = input_desc->dim(ndim - 3);
        size_t input_h = input_desc->dim(ndim - 2);
        size_t input_w = input_desc->dim(ndim - 1);

        CHECK_OR_RETURN(
            input_c % (upscale_factor * upscale_factor) == 0,
            INFINI_STATUS_BAD_TENSOR_SHAPE
        );

        size_t output_c = input_c / (upscale_factor * upscale_factor);
        size_t output_h = input_h * upscale_factor;
        size_t output_w = input_w * upscale_factor;

        CHECK_OR_RETURN(output_desc->dim(ndim - 3) == output_c, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(output_desc->dim(ndim - 2) == output_h, INFINI_STATUS_BAD_TENSOR_SHAPE);
        CHECK_OR_RETURN(output_desc->dim(ndim - 1) == output_w, INFINI_STATUS_BAD_TENSOR_SHAPE);

        // batch size = product of leading dims
        size_t B = 1;
        for (size_t i = 0; i + 3 < ndim; ++i) {
            B *= input_desc->dim(i);
        }

        auto in_strides  = input_desc->strides();
        auto out_strides = output_desc->strides();

        // 根据 ndim 正确获取 strides
        // 对于 3D: (C, H, W)，没有 batch 维度，batch_stride = 0
        // 对于 4D+: (..., B, C, H, W)，batch_stride 在 ndim-4 位置
        ptrdiff_t input_b_stride, input_c_stride, input_h_stride, input_w_stride;
        ptrdiff_t output_b_stride, output_c_stride, output_h_stride, output_w_stride;

        if (ndim == 3) {
            // 3D tensor: (C, H, W)
            input_b_stride = 0;
            input_c_stride = in_strides[0];
            input_h_stride = in_strides[1];
            input_w_stride = in_strides[2];
            
            output_b_stride = 0;
            output_c_stride = out_strides[0];
            output_h_stride = out_strides[1];
            output_w_stride = out_strides[2];
        } else {
            // 4D+ tensor: (..., B, C, H, W)
            // 最后 4 个维度是 B, C, H, W
            input_b_stride = in_strides[ndim - 4];
            input_c_stride = in_strides[ndim - 3];
            input_h_stride = in_strides[ndim - 2];
            input_w_stride = in_strides[ndim - 1];
            
            output_b_stride = out_strides[ndim - 4];
            output_c_stride = out_strides[ndim - 3];
            output_h_stride = out_strides[ndim - 2];
            output_w_stride = out_strides[ndim - 1];
        }

        return utils::Result<PixelShuffleInfo>(PixelShuffleInfo{
            output_desc->dtype(),
            upscale_factor,
            (int)B,
            (int)output_c,
            (int)output_h,
            (int)output_w,
            input_b_stride,
            input_c_stride,
            input_h_stride,
            input_w_stride,
            output_b_stride,
            output_c_stride,
            output_h_stride,
            output_w_stride
        });
    }
};

} // namespace op::pixel_shuffle

#endif
