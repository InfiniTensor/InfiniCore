#ifndef __FUSED_MOE_KERNEL_CUH__
#define __FUSED_MOE_KERNEL_CUH__

#include "infiniop/ops/fused_moe.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

// Inspired by TensorRT-LLM fused MoE post-routing contract:
// cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h and
// cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/moe_kernels.cu.
// TensorRT-LLM is licensed under Apache-2.0. This file is an InfiniCore
// implementation and intentionally does not depend on TensorRT-LLM symbols.

template <typename T>
__device__ inline float moeToFloat(T v) {
    return static_cast<float>(v);
}

template <>
__device__ inline float moeToFloat<half>(half v) {
    return __half2float(v);
}

template <>
__device__ inline float moeToFloat<__nv_bfloat16>(__nv_bfloat16 v) {
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
__device__ inline __nv_bfloat16 moeFromFloat<__nv_bfloat16>(float v) {
    return __float2bfloat16(v);
}

__device__ inline float moeSilu(float x) {
    return x / (1.0f + expf(-x));
}

template <typename T>
__global__ void fusedMoeKernel(
    T *out,
    const T *input,
    const int32_t *selected_experts,
    const float *final_scales,
    const T *w1,
    const T *w2,
    const T *b1,
    const T *b2,
    size_t N,
    size_t hidden_size,
    size_t inter_size,
    size_t num_experts,
    size_t topk,
    size_t w1_cols,
    int activation) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total = N * hidden_size;
    if (idx >= total) {
        return;
    }

    size_t token = idx / hidden_size;
    size_t out_h = idx % hidden_size;
    float accum = 0.0f;

    for (size_t route = 0; route < topk; ++route) {
        int expert = selected_experts[token * topk + route];
        if (expert < 0 || static_cast<size_t>(expert) >= num_experts) {
            continue;
        }
        float route_scale = final_scales[token * topk + route];
        float expert_out = 0.0f;
        for (size_t j = 0; j < inter_size; ++j) {
            float gate = 0.0f;
            float up = 0.0f;
            size_t gate_row = j;
            size_t up_row = activation == INFINIOP_FUSED_MOE_ACT_SWIGLU ? (j + inter_size) : j;
            const T *w1_expert = w1 + (static_cast<size_t>(expert) * w1_cols * hidden_size);
            for (size_t h = 0; h < hidden_size; ++h) {
                float x = moeToFloat(input[token * hidden_size + h]);
                gate += moeToFloat(w1_expert[gate_row * hidden_size + h]) * x;
                if (activation == INFINIOP_FUSED_MOE_ACT_SWIGLU) {
                    up += moeToFloat(w1_expert[up_row * hidden_size + h]) * x;
                }
            }
            if (b1 != nullptr) {
                const T *b1_expert = b1 + static_cast<size_t>(expert) * w1_cols;
                gate += moeToFloat(b1_expert[gate_row]);
                if (activation == INFINIOP_FUSED_MOE_ACT_SWIGLU) {
                    up += moeToFloat(b1_expert[up_row]);
                }
            }
            float act = activation == INFINIOP_FUSED_MOE_ACT_SWIGLU ? (moeSilu(gate) * up) : moeSilu(gate);
            const T *w2_expert = w2 + (static_cast<size_t>(expert) * hidden_size * inter_size);
            expert_out += moeToFloat(w2_expert[out_h * inter_size + j]) * act;
        }
        if (b2 != nullptr) {
            expert_out += moeToFloat(b2[static_cast<size_t>(expert) * hidden_size + out_h]);
        }
        accum += route_scale * expert_out;
    }
    out[idx] = moeFromFloat<T>(accum);
}

#endif // __FUSED_MOE_KERNEL_CUH__
