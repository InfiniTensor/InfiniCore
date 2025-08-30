#ifndef __REDUCE_MAX_KERNEL_CUH__
#define __REDUCE_MAX_KERNEL_CUH__

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ void ReduceMaxKernel(
    Tdata *y_, const Tdata *x_,
    size_t batch, size_t channels, size_t height, size_t width,
    ptrdiff_t y_stride_b, ptrdiff_t y_stride_c, ptrdiff_t y_stride_h,
    ptrdiff_t x_stride_b, ptrdiff_t x_stride_c, ptrdiff_t x_stride_h, ptrdiff_t x_stride_w) {

    Tdata *y = y_ + blockIdx.x * y_stride_b + blockIdx.y * y_stride_c + blockIdx.z * y_stride_h;
    const Tdata *x = x_ + blockIdx.x * x_stride_b + blockIdx.y * x_stride_c + blockIdx.z * x_stride_h;

    // [Reduce] Find the max of each updated row and store in shared memory
    Tcompute max_0 = op::common_cuda::reduce_op::max<BLOCK_SIZE, Tdata>(x, width, x_stride_w);
    if (threadIdx.x == 0) {
        *y = max_0;
    }
}

#endif // __REDUCE_MAX_KERNEL_CUH__
