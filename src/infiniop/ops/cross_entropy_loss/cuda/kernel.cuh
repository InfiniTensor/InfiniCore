#ifndef __CROSS_ENTROPY_KERNEL_CUH__
#define __CROSS_ENTROPY_KERNEL_CUH__

#include <cstdint>
#include <hpcc_fp16.h>
#include <math.h>

__device__ __forceinline__ float to_float(float val) { return val; }

__device__ __forceinline__ float to_float(half val) {
    return __half2float(val);
}

__device__ __forceinline__ float to_float(__hpcc_bfloat16 val) {
    return __bfloat162float(val);
}

template <typename T_in, typename T_out>
__global__ void cross_entropy_loss_kernel(T_out *loss, const T_in *logits,
                                          const int64_t *target, int N, int C,
                                          long long inner_size,
                                          int64_t ignore_index) {

    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * inner_size;
    if (idx >= total) {
        return;
    }

    int n = (int)(idx / inner_size);
    int inner = (int)(idx % inner_size);

    int64_t t = target[idx];

    if (t == ignore_index) {
        loss[idx] = (T_out)0.0f;
        return;
    }
    if (t < 0 || t >= C) {
        loss[idx] = (T_out)0.0f;
        return;
    }

    const long long base_offset = ((long long)n * C * inner_size) + inner;

    // 1. 找到 logits 中的最大值
    float max_val = -HUGE_VALF; // 使用浮点数的最大负值
    for (int c = 0; c < C; ++c) {
        long long offset = base_offset + (long long)c * inner_size;
        max_val = fmaxf(max_val, to_float(logits[offset]));
    }

    // 2. 计算 sum(exp(x - max_val))
    float sum_exp = 0.0f;
    for (int c = 0; c < C; ++c) {
        long long offset = base_offset + (long long)c * inner_size;
        sum_exp += expf(to_float(logits[offset]) - max_val);
    }

    // 3. 计算最终 loss
    long long target_offset = base_offset + (long long)t * inner_size;
    float logit_tgt = to_float(logits[target_offset]);

    loss[idx] = (T_out)(logf(sum_exp) + max_val - logit_tgt);
}

#endif // __CROSS_ENTROPY_KERNEL_CUH__
