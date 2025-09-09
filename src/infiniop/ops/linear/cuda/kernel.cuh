#ifndef __LINEAR_KERNEL_CUH__
#define __LINEAR_KERNEL_CUH__
//  ------------------------------- start: perform operator on CUDA --------------------------------
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ void linearKernel(
	Tdata * y,
	const Tdata * x,
	const Tdata * w,
	const Tdata * b,
	size_t in_features,
	size_t out_features,
	ptrdiff_t y_stride,
	ptrdiff_t x_stride,
	ptrdiff_t w_stride_out,
    ptrdiff_t w_stride_in,
	ptrdiff_t b_stride,
    bool bias  
) {
    size_t y_index = blockIdx.x;
    auto y_ptr = y + y_index * y_stride;
    auto w_ptr = w + y_index * w_stride_out;

    Tcompute y_value = bias ? (Tcompute(*(b + y_index * b_stride))) : Tcompute(0);
    for(size_t i = 0; i < in_features; i ++)
    {
        y_value += Tcompute(*(x + i * x_stride)) * Tcompute(*(w_ptr + i * w_stride_in));
    }
    *y_ptr = y_value;
}
//  -------------------------------- end: perform operator on CUDA ---------------------------------

#endif // __LINEAR_KERNEL_CUH__
