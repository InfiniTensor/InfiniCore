#ifndef __TRIU_KERNEL_CUH__
#define __TRIU_KERNEL_CUH__
//  ------------------------------- start: perform operator on CUDA --------------------------------
template<unsigned int BLOCK_SIZE, typename Tdata>
__device__ void triuKernel(
    Tdata * output,
    const Tdata * input,
    int column_size,
    int diagonal
) {
    int row = blockIdx.x;
    auto output_ptr = output + row * column_size;
    auto input_ptr = input + row * column_size;
    for (int i = threadIdx.x; i < column_size; i += BLOCK_SIZE)
        *(output_ptr + i) = (i < row + diagonal) ? (Tdata)0 : *(input_ptr + i);
}
//  -------------------------------- end: perform operator on CUDA ---------------------------------

#endif // __TRIU_KERNEL_CUH__
