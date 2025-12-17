#ifndef __RMS_NORM_CUDA_KERNEL_H__
#define __RMS_NORM_CUDA_KERNEL_H__

template <unsigned int BLOCK_SIZE, typename Tcompute, typename Tdata, typename Tweight>
__device__ void rmsnormBlock(
    Tdata *__restrict__ y,
    ptrdiff_t stride_y_batch,
    ptrdiff_t stride_y_nhead,
    const Tdata *__restrict__ x1,
    ptrdiff_t stride_x1_batch,
    ptrdiff_t stride_x1_nhead,
    const Tdata *__restrict__ x2,
    ptrdiff_t stride_x2_batch,
    ptrdiff_t stride_x2_nhead,
    const Tweight *__restrict__ w,
    size_t nhead,
    size_t dim,
    float epsilon) {
    // Each block takes care of one head in one batch
    // Each thread deals with every block_size element in the row
    size_t batch_idx = blockIdx.x / nhead;
    size_t head_idx = blockIdx.x % nhead;

    auto y_ptr = y + batch_idx * stride_y_batch + head_idx * stride_y_nhead;
    auto x1_ptr = x1 + batch_idx * stride_x1_batch + head_idx * stride_x1_nhead;
    auto x2_ptr = x2 + batch_idx * stride_x2_batch + head_idx * stride_x2_nhead;
    auto w_ptr = w;

    // Block-reduce sum of x^2
    Tcompute ss = op::common_cuda::reduce_op::sumBinomialSquare<BLOCK_SIZE, Tdata, Tcompute>(x1_ptr, x2_ptr, dim);

    // Thread_0 computes RMS=1/sqrt(ss/dim+epsilon) and stores in shared memory
    __shared__ Tcompute rms;
    if (threadIdx.x == 0) {
        rms = Tcompute(rsqrtf(ss / Tcompute(dim) + epsilon));
    }
    __syncthreads();

    for (size_t i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        y_ptr[i] = Tdata(Tcompute(x1_ptr[i]) * Tcompute(x2_ptr[i]) * Tcompute(w_ptr[i]) * rms);
    }
}

#endif
