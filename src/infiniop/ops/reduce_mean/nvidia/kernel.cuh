#ifndef __REDUCE_MEAN_KERNEL_CUH__
#define __REDUCE_MEAN_KERNEL_CUH__

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ void ReduceMeanKernel(
    Tdata *y_, const Tdata *x_,
    size_t batch, size_t channels, size_t height, size_t width,
    ptrdiff_t y_stride_b, ptrdiff_t y_stride_c, ptrdiff_t y_stride_h,
    ptrdiff_t x_stride_b, ptrdiff_t x_stride_c, ptrdiff_t x_stride_h, ptrdiff_t x_stride_w) {

    Tdata *y = y_ + blockIdx.x * y_stride_b + blockIdx.y * y_stride_c + blockIdx.z * y_stride_h;
    const Tdata *x = x_ + blockIdx.x * x_stride_b + blockIdx.y * x_stride_c + blockIdx.z * x_stride_h;

    // [Reduce] Find the sum of each updated row and store in shared memory
    Tcompute sum_0 = op::common_cuda::reduce_op::sum<BLOCK_SIZE, Tdata, Tcompute>(x, width, x_stride_w);
    if (threadIdx.x == 0) {
        // mean_ = sum_0/width;
        *y = sum_0 / width;
    }
    // __syncthreads();

    // [Elementwise] Divide each element by the sum and store in shared memory
    // *y = mean_;
}

#endif // __REDUCE_MEAN_KERNEL_CUH__
