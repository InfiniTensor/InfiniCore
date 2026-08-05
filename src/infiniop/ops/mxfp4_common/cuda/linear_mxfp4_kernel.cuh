#ifndef __MXFP4_COMMON_CUDA_LINEAR_MXFP4_KERNEL_CUH__
#define __MXFP4_COMMON_CUDA_LINEAR_MXFP4_KERNEL_CUH__

#include "mxfp4_kernel.cuh"

#include <cstddef>
#include <cstdint>

#ifndef INFINIOP_MXFP4_KERNEL
#define INFINIOP_MXFP4_KERNEL __global__ void
#define INFINIOP_MXFP4_KERNEL_DEFINED_HERE
#endif

namespace op::mxfp4_common::cuda {

template <typename T, size_t M_TILE>
INFINIOP_MXFP4_KERNEL linearMxfp4Kernel(
    T *output,
    const T *input,
    const uint8_t *packed_weight,
    const uint8_t *weight_scale,
    const T *bias,
    size_t M,
    size_t N,
    size_t K,
    float alpha) {
    const size_t n = blockIdx.x;
    const size_t m_begin = blockIdx.y * M_TILE;
    const size_t packed_width = K / 2;
    const size_t scale_width = K / 32;
    const auto *packed_row = packed_weight + n * packed_width;
    const auto *scale_row = weight_scale + n * scale_width;

    float sums[M_TILE] = {};
    for (size_t packed_k = threadIdx.x; packed_k < packed_width; packed_k += blockDim.x) {
        float weight_low;
        float weight_high;
        mxfp4DecodePair(
            packed_row[packed_k], scale_row[packed_k / 16], weight_low, weight_high);
        const size_t k = packed_k * 2;
#pragma unroll
        for (size_t tile_m = 0; tile_m < M_TILE; ++tile_m) {
            const size_t m = m_begin + tile_m;
            if (m < M) {
                const size_t input_offset = m * K + k;
                sums[tile_m] += mxfp4Load(input, input_offset) * weight_low
                              + mxfp4Load(input, input_offset + 1) * weight_high;
            }
        }
    }

    extern __shared__ float scratch[];
    mxfp4BlockReduce(sums, scratch);
    if (threadIdx.x == 0) {
#pragma unroll
        for (size_t tile_m = 0; tile_m < M_TILE; ++tile_m) {
            const size_t m = m_begin + tile_m;
            if (m < M) {
                float value = alpha * sums[tile_m];
                if (bias != nullptr) {
                    value += mxfp4Load(bias, n);
                }
                output[m * N + n] = mxfp4Store<T>(value);
            }
        }
    }
}

template <typename T, typename Stream>
void launchLinearMxfp4(T *output,
                       const T *input,
                       const uint8_t *packed_weight,
                       const uint8_t *weight_scale,
                       const T *bias,
                       size_t M,
                       size_t N,
                       size_t K,
                       float alpha,
                       Stream stream) {
    constexpr size_t block_size = 256;
    if (M == 1) {
        linearMxfp4Kernel<T, 1><<<dim3(N, 1), block_size,
                                  block_size * sizeof(float), stream>>>(
            output, input, packed_weight, weight_scale, bias, M, N, K, alpha);
        return;
    }

    constexpr size_t m_tile = 4;
    const dim3 grid(N, (M + m_tile - 1) / m_tile);
    linearMxfp4Kernel<T, m_tile><<<grid, block_size,
                                   m_tile * block_size * sizeof(float), stream>>>(
        output, input, packed_weight, weight_scale, bias, M, N, K, alpha);
}

} // namespace op::mxfp4_common::cuda

#ifdef INFINIOP_MXFP4_KERNEL_DEFINED_HERE
#undef INFINIOP_MXFP4_KERNEL
#undef INFINIOP_MXFP4_KERNEL_DEFINED_HERE
#endif

#endif
