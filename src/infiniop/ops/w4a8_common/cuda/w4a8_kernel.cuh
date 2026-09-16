#ifndef __W4A8_COMMON_CUDA_KERNEL_CUH__
#define __W4A8_COMMON_CUDA_KERNEL_CUH__

#include "../../mxfp4_common/cuda/mxfp4_kernel.cuh"
#include "infiniop/ops/fused_moe.h"

#include <cstddef>
#include <cstdint>

namespace op::w4a8_common::cuda {

__device__ __forceinline__ int decodeInt4(uint8_t value) {
    const int code = value & 0x0f;
    return code >= 8 ? code - 16 : code;
}

template <typename T>
__global__ void quantizeRows(int8_t *output,
                             float *scales,
                             const T *input,
                             size_t rows,
                             size_t cols) {
    const size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    const T *input_row = input + row * cols;
    int8_t *output_row = output + row * cols;
    float max_abs = 0.0f;
    for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        max_abs = fmaxf(max_abs, fabsf(mxfp4Load(input_row, col)));
    }
    extern __shared__ float scratch[];
    scratch[threadIdx.x] = max_abs;
    __syncthreads();
    for (size_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float scale = fmaxf(scratch[0] / 127.0f, 1.0e-8f);
    if (threadIdx.x == 0) {
        scales[row] = scale;
    }
    const float inv_scale = 1.0f / scale;
    for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
        const float value = mxfp4Load(input_row, col) * inv_scale;
        int quantized = static_cast<int>(roundf(value));
        quantized = quantized > 127 ? 127 : quantized;
        quantized = quantized < -127 ? -127 : quantized;
        output_row[col] = static_cast<int8_t>(quantized);
    }
}

__device__ __forceinline__ float dotInt8Int4(
    const int8_t *input,
    const int8_t *packed_weight,
    size_t K) {
    float sum = 0.0f;
    const size_t packed_width = K / 2;
    for (size_t packed_k = threadIdx.x; packed_k < packed_width; packed_k += blockDim.x) {
        const uint8_t packed = static_cast<uint8_t>(packed_weight[packed_k]);
        const size_t k = packed_k * 2;
        sum += static_cast<float>(input[k]) * decodeInt4(packed >> 4)
             + static_cast<float>(input[k + 1]) * decodeInt4(packed);
    }
    return sum;
}

template <typename T>
__global__ void linearKernel(T *output,
                             const int8_t *input,
                             const float *input_scale,
                             const int8_t *packed_weight,
                             const float *weight_scale,
                             const T *bias,
                             size_t M,
                             size_t N,
                             size_t K,
                             float alpha) {
    const size_t n = blockIdx.x;
    const size_t m = blockIdx.y;
    if (m >= M || n >= N) {
        return;
    }
    const float partial = dotInt8Int4(
        input + m * K, packed_weight + n * (K / 2), K);
    float values[1] = {partial};
    extern __shared__ float scratch[];
    mxfp4BlockReduce(values, scratch);
    if (threadIdx.x == 0) {
        float value = alpha * values[0] * input_scale[m] * weight_scale[n];
        if (bias != nullptr) {
            value += mxfp4Load(bias, n);
        }
        output[m * N + n] = mxfp4Store<T>(value);
    }
}

template <typename T, typename Stream>
void launchLinear(T *output,
                  const T *input,
                  const int8_t *packed_weight,
                  const float *weight_scale,
                  const T *bias,
                  int8_t *quantized_input,
                  float *input_scale,
                  size_t M,
                  size_t N,
                  size_t K,
                  float alpha,
                  Stream stream) {
    constexpr size_t block_size = 256;
    quantizeRows<<<M, block_size, block_size * sizeof(float), stream>>>(
        quantized_input, input_scale, input, M, K);
    linearKernel<<<dim3(N, M), block_size, block_size * sizeof(float), stream>>>(
        output, quantized_input, input_scale, packed_weight, weight_scale,
        bias, M, N, K, alpha);
}

__device__ __forceinline__ float activate(float gate,
                                          float up,
                                          infiniopFusedMoeActivation_t activation) {
    if (activation == INFINIOP_FUSED_MOE_ACT_SITUGLU) {
        constexpr float beta = 4.0f;
        constexpr float linear_beta = 25.0f;
        const float situ_gate = beta * tanhf(gate / beta) / (1.0f + expf(-gate));
        const float bounded_up = linear_beta * tanhf(up / linear_beta);
        return situ_gate * bounded_up;
    }
    return gate / (1.0f + expf(-gate)) * up;
}

template <typename T>
__global__ void moeW13Kernel(
    T *activated,
    const int8_t *input,
    const float *input_scale,
    const int32_t *selected_experts,
    const int8_t *w13_packed,
    const float *w13_scale,
    size_t route_count,
    size_t topk,
    size_t num_experts,
    size_t hidden_size,
    size_t intermediate_size,
    infiniopFusedMoeActivation_t activation) {
    const size_t block = blockIdx.x;
    const size_t route = block / intermediate_size;
    const size_t i = block - route * intermediate_size;
    if (route >= route_count) {
        return;
    }
    const int32_t expert = selected_experts[route];
    if (expert < 0 || static_cast<size_t>(expert) >= num_experts) {
        if (threadIdx.x == 0) {
            activated[route * intermediate_size + i] = mxfp4Store<T>(0.0f);
        }
        return;
    }

    const size_t token = route / topk;
    const size_t packed_width = hidden_size / 2;
    const size_t gate_row = static_cast<size_t>(expert) * 2 * intermediate_size + i;
    const size_t up_row = gate_row + intermediate_size;
    const auto *token_input = input + token * hidden_size;
    float values[2] = {
        dotInt8Int4(token_input, w13_packed + gate_row * packed_width, hidden_size),
        dotInt8Int4(token_input, w13_packed + up_row * packed_width, hidden_size)};
    extern __shared__ float scratch[];
    mxfp4BlockReduce(values, scratch);
    if (threadIdx.x == 0) {
        const float activation_scale = input_scale[token];
        const float gate = values[0] * activation_scale * w13_scale[gate_row];
        const float up = values[1] * activation_scale * w13_scale[up_row];
        activated[route * intermediate_size + i]
            = mxfp4Store<T>(activate(gate, up, activation));
    }
}

template <typename T>
__global__ void moeW2Kernel(
    T *output,
    const int8_t *activated,
    const float *activated_scale,
    const int32_t *selected_experts,
    const float *routing_weights,
    const int8_t *w2_packed,
    const float *w2_scale,
    size_t num_tokens,
    size_t topk,
    size_t num_experts,
    size_t hidden_size,
    size_t intermediate_size) {
    const size_t block = blockIdx.x;
    const size_t token = block / hidden_size;
    const size_t h = block - token * hidden_size;
    if (token >= num_tokens) {
        return;
    }

    float output_value = 0.0f;
    extern __shared__ float scratch[];
    for (size_t route_index = 0; route_index < topk; ++route_index) {
        const size_t route = token * topk + route_index;
        const int32_t expert = selected_experts[route];
        if (expert < 0 || static_cast<size_t>(expert) >= num_experts) {
            continue;
        }
        const size_t weight_row = static_cast<size_t>(expert) * hidden_size + h;
        float values[1] = {dotInt8Int4(
            activated + route * intermediate_size,
            w2_packed + weight_row * (intermediate_size / 2),
            intermediate_size)};
        mxfp4BlockReduce(values, scratch);
        if (threadIdx.x == 0) {
            output_value += routing_weights[route] * values[0]
                          * activated_scale[route] * w2_scale[weight_row];
        }
    }
    if (threadIdx.x == 0) {
        output[token * hidden_size + h] = mxfp4Store<T>(output_value);
    }
}

template <typename T, typename Stream>
void launchFusedMoe(
    T *output,
    const T *input,
    const int32_t *selected_experts,
    const float *routing_weights,
    const int8_t *w13_packed,
    const float *w13_scale,
    const int8_t *w2_packed,
    const float *w2_scale,
    int8_t *quantized_input,
    float *input_scale,
    T *activated,
    int8_t *quantized_activated,
    float *activated_scale,
    size_t num_tokens,
    size_t topk,
    size_t num_experts,
    size_t hidden_size,
    size_t intermediate_size,
    infiniopFusedMoeActivation_t activation,
    Stream stream) {
    constexpr size_t block_size = 256;
    const size_t route_count = num_tokens * topk;
    quantizeRows<<<num_tokens, block_size, block_size * sizeof(float), stream>>>(
        quantized_input, input_scale, input, num_tokens, hidden_size);
    moeW13Kernel<<<route_count * intermediate_size, block_size,
                   2 * block_size * sizeof(float), stream>>>(
        activated, quantized_input, input_scale, selected_experts,
        w13_packed, w13_scale, route_count, topk, num_experts,
        hidden_size, intermediate_size, activation);
    quantizeRows<<<route_count, block_size, block_size * sizeof(float), stream>>>(
        quantized_activated, activated_scale, activated,
        route_count, intermediate_size);
    moeW2Kernel<<<num_tokens * hidden_size, block_size,
                  block_size * sizeof(float), stream>>>(
        output, quantized_activated, activated_scale, selected_experts,
        routing_weights, w2_packed, w2_scale, num_tokens, topk,
        num_experts, hidden_size, intermediate_size);
}

} // namespace op::w4a8_common::cuda

#endif
