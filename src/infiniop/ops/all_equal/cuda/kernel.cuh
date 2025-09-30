#ifndef __ALL_EQUAL_KERNEL_CUH__
#define __ALL_EQUAL_KERNEL_CUH__
//  ------------------------------- start: perform operator on CUDA --------------------------------
template <unsigned int BLOCK_SIZE, typename Tdata>
__device__ void allEqualKernel(
    bool *c,
    const Tdata *a,
    const Tdata *b,
    size_t ndim,
    size_t total_size,
    ptrdiff_t *contiguous_strides,
    ptrdiff_t *a_strides,
    ptrdiff_t *b_strides) {
    // 使用共享内存来避免竞态条件
    __shared__ bool block_result;

    if (threadIdx.x == 0) {
        block_result = true;
    }
    __syncthreads();

    // 每个线程检查自己负责的元素
    bool thread_result = true;
    for (size_t i = threadIdx.x; i < total_size; i += BLOCK_SIZE) {
        auto a_ptr = a;
        auto b_ptr = b;
        size_t rem = i;
        for (int d = ndim - 1; d >= 0; d--) {
            size_t dim_index = rem / contiguous_strides[d];
            rem = rem % contiguous_strides[d];
            a_ptr += dim_index * a_strides[d];
            b_ptr += dim_index * b_strides[d];
        }
        if (*a_ptr != *b_ptr) {
            thread_result = false;
            break; // 发现不匹配，提前退出
        }
    }

    // 使用原子操作来安全地更新结果
    if (!thread_result) {
        atomicAnd((int *)&block_result, 0);
    }

    __syncthreads();

    // 只有第一个线程写入最终结果
    if (threadIdx.x == 0) {
        *c = block_result;
    }
}
//  -------------------------------- end: perform operator on CUDA ---------------------------------

#endif // __ALL_EQUAL_KERNEL_CUH__
