#ifndef __FUSED_MOE_KERNEL_CUH__
#define __FUSED_MOE_KERNEL_CUH__

#include <math_constants.h>
#include <stdint.h>

template <typename T>
__device__ inline float moeToFloat(T v) {
    return static_cast<float>(v);
}

template <>
__device__ inline float moeToFloat<half>(half v) {
    return __half2float(v);
}

template <>
__device__ inline float moeToFloat<cuda_bfloat16>(cuda_bfloat16 v) {
    return __bfloat162float(v);
}

template <typename T>
__device__ inline T moeFromFloat(float v) {
    return static_cast<T>(v);
}

template <>
__device__ inline half moeFromFloat<half>(float v) {
    return __float2half(v);
}

template <>
__device__ inline cuda_bfloat16 moeFromFloat<cuda_bfloat16>(float v) {
    return __float2bfloat16(v);
}

__device__ inline float moeSilu(float x) {
    return x / (1.0f + expf(-x));
}

template <typename T>
__global__ void fusedMoeW1Kernel(
    T *w1_out,
    const T *input,
    const int32_t *selected_experts,
    const T *w1,
    const T *b1,
    size_t route_count,
    size_t hidden_size,
    size_t topk,
    size_t w1_cols,
    size_t num_experts) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = route_count * w1_cols;
    if (idx >= total) {
        return;
    }

    size_t route_id = idx / w1_cols;
    size_t col = idx % w1_cols;
    size_t token = route_id / topk;
    int expert = selected_experts[route_id];
    if (expert < 0 || static_cast<size_t>(expert) >= num_experts) {
        w1_out[idx] = moeFromFloat<T>(0.0f);
        return;
    }

    const T *x = input + token * hidden_size;
    const T *w = w1 + (static_cast<size_t>(expert) * w1_cols + col) * hidden_size;
    float acc = 0.0f;
    for (size_t h = 0; h < hidden_size; ++h) {
        acc += moeToFloat(w[h]) * moeToFloat(x[h]);
    }
    if (b1 != nullptr) {
        acc += moeToFloat(b1[static_cast<size_t>(expert) * w1_cols + col]);
    }
    w1_out[idx] = moeFromFloat<T>(acc);
}

template <typename T>
__global__ void fusedMoeActivationKernel(
    T *activated,
    const T *w1_out,
    size_t route_count,
    size_t inter_size,
    size_t w1_cols,
    int activation) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = route_count * inter_size;
    if (idx >= total) {
        return;
    }

    size_t route_id = idx / inter_size;
    size_t j = idx % inter_size;
    const T *row = w1_out + route_id * w1_cols;
    float gate = moeToFloat(row[j]);
    float act;
    if (activation == INFINIOP_FUSED_MOE_ACT_SWIGLU) {
        float up = moeToFloat(row[j + inter_size]);
        act = moeSilu(gate) * up;
    } else {
        act = moeSilu(gate);
    }
    activated[idx] = moeFromFloat<T>(act);
}

template <typename T>
__global__ void fusedMoeW2ScatterKernel(
    float *out_accum,
    const T *activated,
    const int32_t *selected_experts,
    const float *final_scales,
    const T *w2,
    const T *b2,
    size_t route_count,
    size_t hidden_size,
    size_t inter_size,
    size_t topk,
    size_t num_experts) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = route_count * hidden_size;
    if (idx >= total) {
        return;
    }

    size_t route_id = idx / hidden_size;
    size_t out_h = idx % hidden_size;
    size_t token = route_id / topk;
    int expert = selected_experts[route_id];
    if (expert < 0 || static_cast<size_t>(expert) >= num_experts) {
        return;
    }

    const T *act = activated + route_id * inter_size;
    const T *w = w2 + (static_cast<size_t>(expert) * hidden_size + out_h) * inter_size;
    float acc = 0.0f;
    for (size_t j = 0; j < inter_size; ++j) {
        acc += moeToFloat(w[j]) * moeToFloat(act[j]);
    }
    if (b2 != nullptr) {
        acc += moeToFloat(b2[static_cast<size_t>(expert) * hidden_size + out_h]);
    }
    float scaled = final_scales[route_id] * acc;
    atomicAdd(out_accum + token * hidden_size + out_h, scaled);
}

template <typename T>
__global__ void fusedMoeCastKernel(T *out, const float *out_accum, size_t total) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        out[idx] = moeFromFloat<T>(out_accum[idx]);
    }
}

#endif // __FUSED_MOE_KERNEL_CUH__
