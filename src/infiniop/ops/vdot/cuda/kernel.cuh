#ifndef __VDOT_CUDA_KERNEL_CUH__
#define __VDOT_CUDA_KERNEL_CUH__

#include <cuda_runtime.h>
#include <cstddef>

namespace op::vdot::cuda {

template <unsigned int BLOCK_SIZE, typename Tdata, typename Tcompute>
__global__ void vdotKernel(Tcompute *out, const Tdata *a, const Tdata *b,
                           size_t length, ptrdiff_t a_stride,
                           ptrdiff_t b_stride) {

    // 每个线程计算部分点积
    Tcompute local_sum = 0;
    for (size_t i = threadIdx.x; i < length; i += BLOCK_SIZE) {
        Tcompute a_val = static_cast<Tcompute>(a[i * a_stride]);
        Tcompute b_val = static_cast<Tcompute>(b[i * b_stride]);
        local_sum += a_val * b_val;
    }

    // 使用共享内存进行 block 内归约（不依赖 CUB）
    __shared__ Tcompute sdata[BLOCK_SIZE];
    sdata[threadIdx.x] = local_sum;
    __syncthreads();

    // 标准的二分归约算法
    for (unsigned int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }

    // Thread 0 写入结果
    if (threadIdx.x == 0) {
        *out = sdata[0];
    }
}

} // namespace op::vdot::cuda

#endif // __VDOT_CUDA_KERNEL_CUH__
