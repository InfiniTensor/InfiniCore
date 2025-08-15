#ifndef __EQUAL_KERNEL_CUH__
#define __EQUAL_KERNEL_CUH__
//  ------------------------------- start: perform operator on CUDA --------------------------------
template <unsigned int BLOCK_SIZE, typename Tdata>
__device__ void equalKernel(
    bool * c,
    const Tdata * a,
    const Tdata * b,
    size_t ndim,
    size_t total_size,
    ptrdiff_t* contiguous_strides,
    ptrdiff_t* a_strides,
    ptrdiff_t* b_strides
) {
    if (threadIdx.x == 0)
    {
        *c = true;
    }
    __syncthreads();
    for(size_t i = threadIdx.x; i < total_size; i += BLOCK_SIZE) {
        auto a_ptr = a;
        auto b_ptr = b;
        size_t rem = i;
        for(int d = ndim - 1; d >= 0; d --) {
            size_t dim_index = rem / contiguous_strides[d];
            rem = rem % contiguous_strides[d];
            a_ptr += dim_index * a_strides[d];
            b_ptr += dim_index * b_strides[d];
        }
        if ((*a_ptr != *b_ptr)  && (*c == true)) {
            *c = false;
        }

    }
}
//  -------------------------------- end: perform operator on CUDA ---------------------------------

#endif // __EQUAL_KERNEL_CUH__
