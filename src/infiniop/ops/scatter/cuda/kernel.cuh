#ifndef __SCATTER_KERNEL_CUH__
#define __SCATTER_KERNEL_CUH__
//  ------------------------------- start: perform operator on CUDA --------------------------------
template <unsigned int BLOCK_SIZE, typename Tdata>
__device__ void scatterKernel(
    Tdata * output,
    const Tdata * input,
    const int64_t * index,
    size_t ndim,
    size_t index_scatter_size,
    ptrdiff_t * output_strides,
    ptrdiff_t * input_strides,
    ptrdiff_t * index_strides,
    ptrdiff_t * contiguous_strides,
    int scatter_dim
) {
        auto output_ptr = output;
        auto input_ptr = input;
        auto index_ptr = index;        
        size_t rem = blockIdx.x;
        for(int d = ndim - 1; d >= 0; d --) {
            if (d == scatter_dim)
                continue;
            size_t dim_index = rem / contiguous_strides[d];
            rem = rem % contiguous_strides[d];
            output_ptr += dim_index * output_strides[d];
            input_ptr += dim_index * input_strides[d];
            index_ptr += dim_index * index_strides[d];
        }
        for (size_t c = threadIdx.x; c < index_scatter_size; c += BLOCK_SIZE) {
            int64_t scatter_number = *(index_ptr + c * index_strides[scatter_dim]);
            *(output_ptr + scatter_number * output_strides[scatter_dim]) = \
                *(input_ptr + c * input_strides[scatter_dim]);
        }
}
//  -------------------------------- end: perform operator on CUDA ---------------------------------

#endif // __SCATTER_KERNEL_CUH__
