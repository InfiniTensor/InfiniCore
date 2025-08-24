#ifndef __REDUCE_MEAN_KERNEL_CUH__
#define __REDUCE_MEAN_KERNEL_CUH__
//  ------------------------------- start: perform operator on CUDA --------------------------------
template <typename Tdata, typename Tcompute>
__device__ __forceinline__ void reduceMeanKernel(
    Tdata *output,
    const Tdata *input,
    int ndim,
    const size_t * input_shape,
    const ptrdiff_t * output_strides,
    const ptrdiff_t * input_strides,
    const ptrdiff_t * contiguous_strides,
    size_t dim
) {
    auto output_ptr = output;
    auto input_ptr = input;
    int rem = blockIdx.x;
    for(int d = ndim - 1; d >= 0; d --)
    {
        if(d == dim)
            continue;
        size_t dim_index = rem / contiguous_strides[d];
        rem = rem % contiguous_strides[d];
        output_ptr += dim_index * output_strides[d];
        input_ptr += dim_index * input_strides[d]; 
    }
    Tcompute sum = 0.;
    Tcompute count = 0.;
    for (size_t i = 0; i < input_shape[dim]; i++) {
        sum += Tcompute(*(input_ptr + i * input_strides[dim]));
        count += 1.;
    }
    *output_ptr = sum / count;
}
//  -------------------------------- end: perform operator on CUDA ---------------------------------

#endif // __REDUCE_MEAN_KERNEL_CUH__
