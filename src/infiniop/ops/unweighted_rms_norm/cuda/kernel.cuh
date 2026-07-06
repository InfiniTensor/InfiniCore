#ifndef __UNWEIGHTED_RMS_NORM_CUDA_KERNEL_H__
#define __UNWEIGHTED_RMS_NORM_CUDA_KERNEL_H__

template <unsigned int BLOCK_SIZE, typename Tcompute, typename Tdata>
__device__ void unweightedRMSNormBlock(
    Tdata *__restrict__ y,
    const Tdata *__restrict__ x,
    size_t dim,
    float epsilon) {
    const size_t row = blockIdx.x;
    auto y_ptr = y + row * dim;
    auto x_ptr = x + row * dim;

    Tcompute ss = op::common_cuda::reduce_op::sumSquared<BLOCK_SIZE, Tdata, Tcompute>(x_ptr, dim);

    __shared__ Tcompute rms;
    if (threadIdx.x == 0) {
        rms = Tcompute(rsqrtf(ss / Tcompute(dim) + epsilon));
    }
    __syncthreads();

    for (size_t i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        y_ptr[i] = Tdata(Tcompute(x_ptr[i]) * rms);
    }
}

#endif
