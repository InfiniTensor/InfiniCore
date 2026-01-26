#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>

// =========================================================
// 标准 CUDA 错误检查宏
// =========================================================
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d code=%d(%s)\n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// =========================================================
// 辅助宏：计算 Grid 大小
// =========================================================
#define DIVUP(x, y) (((x) + (y) - 1) / (y))

// =========================================================
// 辅助函数：Warp 级归约 (Reduce)
// 很多 Reduce Kernel 会用到这个
// =========================================================
template <typename T>
__inline__ __device__ T warpReduceSum(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

#endif // CUDA_UTILS_H