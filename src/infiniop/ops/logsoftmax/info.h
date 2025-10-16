#ifndef __LOGSOFTMAX_INFO_H__
#define __LOGSOFTMAX_INFO_H__

#include "../../../utils.h"
#include "../../tensor.h"
#include <vector>

namespace op::logsoftmax {

class LogSoftmaxInfo {
    LogSoftmaxInfo() = default;

public:
    infiniDtype_t x_dtype;
    infiniDtype_t y_dtype;
    size_t batch_size;
    size_t probs_size;

    // Original tensor dimensions for 3D support
    size_t ndim;
    size_t seq_len; // Only used for 3D tensors

    // Flattened strides for CPU iteration
    ptrdiff_t y_stride_b;
    ptrdiff_t y_stride_p;
    ptrdiff_t x_stride_b;
    ptrdiff_t x_stride_p;

    // Original 3D strides for correct memory access
    ptrdiff_t y_stride_0, y_stride_1, y_stride_2;
    ptrdiff_t x_stride_0, x_stride_1, x_stride_2;

    static utils::Result<LogSoftmaxInfo> create(infiniopTensorDescriptor_t y_desc, infiniopTensorDescriptor_t x_desc) {
        auto x_dtype = x_desc->dtype();
        auto y_dtype = y_desc->dtype();

        CHECK_DTYPE(x_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);
        // Check the output data type, and any dtype is allowed to output fp32.
        CHECK_DTYPE(y_dtype, INFINI_DTYPE_F16, INFINI_DTYPE_BF16, INFINI_DTYPE_F32);

        auto x_shape = x_desc->shape();
        auto y_shape = y_desc->shape();
        CHECK_SAME_SHAPE(x_shape, y_shape);

        auto ndim = x_desc->ndim();
        if (ndim < 2 || ndim > 3) {
            CHECK_STATUS(INFINI_STATUS_BAD_TENSOR_SHAPE);
        }

        size_t batch_size, probs_size, seq_len = 0;
        if (ndim == 2) {
            batch_size = x_shape[0];
            probs_size = x_shape[1];
        } else { // ndim == 3
            batch_size = x_shape[0] * x_shape[1];
            probs_size = x_shape[2];
            seq_len = x_shape[1];
        }

        // Store original strides for all dimensions
        ptrdiff_t y_stride_0 = 0, y_stride_1 = 0, y_stride_2 = 0;
        ptrdiff_t x_stride_0 = 0, x_stride_1 = 0, x_stride_2 = 0;

        if (ndim == 2) {
            y_stride_0 = y_desc->stride(0); // First dimension
            y_stride_1 = y_desc->stride(1); // Second dimension
            x_stride_0 = x_desc->stride(0);
            x_stride_1 = x_desc->stride(1);
        } else if (ndim == 3) {
            y_stride_0 = y_desc->stride(0); // First dimension (batch)
            y_stride_1 = y_desc->stride(1); // Second dimension (seq)
            y_stride_2 = y_desc->stride(2); // Third dimension (prob)
            x_stride_0 = x_desc->stride(0);
            x_stride_1 = x_desc->stride(1);
            x_stride_2 = x_desc->stride(2);
        }

        ptrdiff_t y_stride_b, y_stride_p, x_stride_b, x_stride_p;
        if (ndim == 2) {
            y_stride_b = y_desc->stride(0);
            y_stride_p = y_desc->stride(1);
            x_stride_b = x_desc->stride(0);
            x_stride_p = x_desc->stride(1);
        } else { // ndim == 3
            // For 3D tensors, flat the first two dimensions
            // The CPU implementation expects to iterate through batch_size elements
            // where each batch contains probs_size elements
            // For flattened iteration, we need stride between consecutive sequences
            y_stride_b = y_desc->stride(1); // stride between sequences (20*512 -> 512)
            y_stride_p = y_desc->stride(2); // stride within probability dimension
            x_stride_b = x_desc->stride(1); // stride between sequences
            x_stride_p = x_desc->stride(2); // stride within probability dimension
        }

        return utils::Result<LogSoftmaxInfo>(LogSoftmaxInfo{
            x_dtype,
            y_dtype,
            batch_size,
            probs_size,
            ndim,
            seq_len,
            y_stride_b,
            y_stride_p,
            x_stride_b,
            x_stride_p,
            y_stride_0,
            y_stride_1,
            y_stride_2,
            x_stride_0,
            x_stride_1,
            x_stride_2});
    }
};

} // namespace op::logsoftmax

#endif // __LOGSOFTMAX_INFO_H__
