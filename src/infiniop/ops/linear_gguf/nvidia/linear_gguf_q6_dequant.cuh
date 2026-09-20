#pragma once

#include "../ggml_blocks.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace op::linear_gguf::nvidia {

// A warp decodes one Q6_K block, so adjacent lanes write adjacent elements.
__global__ void dequant_q6_warp_kernel(
    const uint8_t *__restrict__ weight, __nv_bfloat16 *__restrict__ output,
    int64_t row_start, int rows, int k, int64_t row_bytes) {
    const int64_t warp = (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    const int blocks_per_row = k / ggml_blocks::QK_K;
    const int64_t row = warp / blocks_per_row;
    if (row >= rows) {
        return;
    }
    const int block = warp % blocks_per_row;
    const uint8_t *packed = weight + (row_start + row) * row_bytes + block * ggml_blocks::SIZE_Q6_K;
    const float d = ggml_blocks::half_to_float(ggml_blocks::read_u16(packed + ggml_blocks::Q6K_OFF_D));
    const int8_t *scales = reinterpret_cast<const int8_t *>(packed + ggml_blocks::Q6K_OFF_SCALES);
    __nv_bfloat16 *out = output + row * k + block * ggml_blocks::QK_K;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const uint8_t low0 = packed[half * 64 + lane];
        const uint8_t low1 = packed[half * 64 + lane + 32];
        const uint8_t high = packed[ggml_blocks::Q6K_OFF_QH + half * 32 + lane];
        const int scale = half * 8 + lane / 16;
        const int q0 = ((low0 & 15) | ((high & 3) << 4)) - 32;
        const int q1 = ((low1 & 15) | (((high >> 2) & 3) << 4)) - 32;
        const int q2 = ((low0 >> 4) | (((high >> 4) & 3) << 4)) - 32;
        const int q3 = ((low1 >> 4) | (((high >> 6) & 3) << 4)) - 32;
        out[half * 128 + lane] = __float2bfloat16_rn(d * scales[scale] * q0);
        out[half * 128 + lane + 32] = __float2bfloat16_rn(d * scales[scale + 2] * q1);
        out[half * 128 + lane + 64] = __float2bfloat16_rn(d * scales[scale + 4] * q2);
        out[half * 128 + lane + 96] = __float2bfloat16_rn(d * scales[scale + 6] * q3);
    }
}

inline void launch_dequant_q6_warp(const uint8_t *weight, __nv_bfloat16 *output,
                                   int64_t row_start, int rows, int k,
                                   int64_t row_bytes, cudaStream_t stream) {
    constexpr int threads = 256;
    const int64_t warps = static_cast<int64_t>(rows) * (k / ggml_blocks::QK_K);
    const unsigned grid = static_cast<unsigned>((warps + threads / 32 - 1) / (threads / 32));
    dequant_q6_warp_kernel<<<grid, threads, 0, stream>>>(weight, output, row_start, rows, k, row_bytes);
}

} // namespace op::linear_gguf::nvidia
