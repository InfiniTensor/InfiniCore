#ifndef __LINEAR_BACKWARD_KERNEL_CUH__
#define __LINEAR_BACKWARD_KERNEL_CUH__
//  ------------------------------- start: perform operator on CUDA --------------------------------
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ void linearBackwardKernel(
    Tdata * grad_x,
    Tdata * grad_w,
    Tdata * grad_b,
    const Tdata * grad_y,
    const Tdata * x,
    const Tdata * w,
    size_t out_features,
    ptrdiff_t grad_x_stride,
    ptrdiff_t grad_w_stride_out,
    ptrdiff_t grad_w_stride_in,
    ptrdiff_t grad_b_stride,
    ptrdiff_t grad_y_stride,
    ptrdiff_t x_stride,
    ptrdiff_t w_stride_out,
    ptrdiff_t w_stride_in,
    bool bias
) {
    size_t in_index = blockIdx.x;

    auto w_ptr = w + in_index * w_stride_in;
    auto grad_w_ptr = grad_w + in_index * grad_w_stride_in;
    Tcompute grad_x_sum = 0.;
    Tcompute x_value = *(x + in_index * x_stride);
    for (size_t j = 0; j < out_features; j ++)
    {
        Tcompute grad_y_value = *(grad_y + j * grad_y_stride);
        grad_x_sum += grad_y_value * Tcompute(*(w_ptr + j * w_stride_out));
        (*(grad_w_ptr + j * grad_w_stride_out)) = x_value * grad_y_value;
        if (bias && blockIdx.x == 0)
            (*(grad_b + j * grad_b_stride)) = grad_y_value;
    }
    (*(grad_x + in_index * grad_x_stride)) = grad_x_sum;
}
//  -------------------------------- end: perform operator on CUDA ---------------------------------

#endif // __LINEAR_BACKWARD_KERNEL_CUH__
