#ifndef __INFINIOP_AVG_POOL1D_CUDA_KERNEL_CUH__
#define __INFINIOP_AVG_POOL1D_CUDA_KERNEL_CUH__

template <typename T>
__device__ void avgPool1dKernel(
    T *y,
    const T *x,
    size_t batch,
    size_t channels,
    size_t in_width,
    size_t out_width,
    size_t kernel_size,
    size_t stride,
    size_t padding,
    // Strides need to be passed explicitly to device
    ptrdiff_t y_stride_batch,
    ptrdiff_t y_stride_channel,
    ptrdiff_t y_stride_width,
    ptrdiff_t x_stride_batch,
    ptrdiff_t x_stride_channel,
    ptrdiff_t x_stride_width) {

    // Grid Strategy: One thread per output pixel
    // Total threads needed: Batch * Channel * OutWidth
    size_t total_elements = batch * channels * out_width;

    // Grid-Stride Loop
    for (size_t idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < total_elements; 
         idx += gridDim.x * blockDim.x) {

        // 1. Reconstruct indices (b, c, ow) from flat index
        size_t ow = idx % out_width;
        size_t temp = idx / out_width;
        size_t c = temp % channels;
        size_t b = temp / channels;

        // 2. Calculate Output Offset
        size_t y_offset = b * y_stride_batch + c * y_stride_channel + ow * y_stride_width;

        // 3. Calculate Input Window
        long long start_w = static_cast<long long>(ow * stride) - padding;
        
        // 4. Stencil Summation
        T sum = 0;
        // Optimization: Convert T to float for accumulation precision if needed, 
        // but here we stick to T to match generic template
        
        for (size_t k = 0; k < kernel_size; ++k) {
            long long iw = start_w + k;
            
            // Boundary check
            if (iw >= 0 && iw < static_cast<long long>(in_width)) {
                size_t x_offset = b * x_stride_batch + c * x_stride_channel + iw * x_stride_width;
                sum += x[x_offset];
            }
        }

        // 5. Average and Write back
        // Using static_cast to handle half/bf16 division properly if operators are overloaded
        y[y_offset] = sum / static_cast<T>(kernel_size);
    }
}

#endif // __INFINIOP_AVG_POOL1D_CUDA_KERNEL_CUH__