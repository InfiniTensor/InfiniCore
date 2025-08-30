#ifndef __RMS_NORM_BACKWARD_KERNEL_CUH__
#define __RMS_NORM_BACKWARD_KERNEL_CUH__
//  ------------------------------- start: perform operator on CUDA --------------------------------
template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ void rmsNormBackwardKernel(
    Tdata * grad_x,
    Tdata * grad_w_cuda,
    const Tdata * grad_y,
    const Tdata * x,
    const Tdata * w,
    size_t ndim,
    size_t batch_size,
    size_t norm_size,
    const ptrdiff_t *__restrict__ grad_x_strides,
    const ptrdiff_t *__restrict__ grad_y_strides,
    const ptrdiff_t *__restrict__ x_strides,
    const ptrdiff_t *__restrict__ contiguous_strides,
    ptrdiff_t  w_stride
) {
    auto grad_x_ptr = grad_x;
    auto grad_y_ptr = grad_y;
    auto x_ptr = x;
    
    size_t batch_index = blockIdx.x;
    size_t rem = batch_index;
    for (int d = ndim - 2; d >= 0; d --)
    {
        size_t dim_index = rem / contiguous_strides[d];
        rem = rem % contiguous_strides[d];
        grad_x_ptr += dim_index * grad_x_strides[d];
        grad_y_ptr += dim_index * grad_y_strides[d];
        x_ptr += dim_index * x_strides[d];
    }
    auto grad_w_ptr = grad_w_cuda + batch_index;

    Tcompute ss = op::common_cuda::reduce_op::sumSquared<BLOCK_SIZE, Tdata, Tcompute>(
        x_ptr, norm_size
    );
    Tcompute norm_size_f = Tcompute(norm_size);

    __shared__ Tcompute rms;
    __shared__ Tcompute sum_grad_y_times_y;
    if (threadIdx.x == 0) {
        
        rms = rsqrtf(ss / norm_size_f);
        sum_grad_y_times_y = 0.;
        for (size_t c = 0; c < norm_size; c++) {
            Tcompute grad_y_times_normed_x = Tcompute(grad_y_ptr[c]) * Tcompute(x_ptr[c]) * rms;
            *(grad_w_ptr + c * batch_size) = grad_y_times_normed_x;
            sum_grad_y_times_y = sum_grad_y_times_y + grad_y_times_normed_x * Tcompute(*(w + c * w_stride));
        }
    }
    __syncthreads();

    for (size_t c = threadIdx.x; c < norm_size; c += BLOCK_SIZE) {
        grad_x_ptr[c] = Tdata(rms * (
            Tcompute(*(w + c * w_stride)) * Tcompute(grad_y_ptr[c]) - rms * sum_grad_y_times_y * Tcompute(x_ptr[c]) / norm_size_f
        ));
    }
}

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__device__ void sumUpGradWKernel(
    Tdata * grad_w,
    Tdata * grad_w_cuda,
    size_t batch_size,
    ptrdiff_t grad_w_stride
) {
    size_t norm_index = blockIdx.x;
    
    Tcompute sum_grad_w = op::common_cuda::reduce_op::sum<BLOCK_SIZE, Tdata, Tcompute>(
        grad_w_cuda + norm_index * batch_size, batch_size
    );
    if (threadIdx.x == 0)
        *(grad_w + norm_index * grad_w_stride) = Tdata(sum_grad_w);
}
//  -------------------------------- end: perform operator on CUDA ---------------------------------

#endif // __RMS_NORM_BACKWARD_KERNEL_CUH__
