#pragma once
#include <cuda_fp16.h>
#include <cstdint>

/**
 * @brief Unpack 8x 4-bit unsigned integers (0..15) from a uint32_t into 8 half values.
 *
 * GPTQ dequant is applied outside this helper (aligned with nvidia impl):
 *   out = (q - (z + 1)) * s
 *
 * Output order matches interleaved half2 packing:
 *   (v0,v4), (v1,v5), (v2,v6), (v3,v7)
 */
__device__ __forceinline__ uint4 dequantize_s4_to_fp16x2_gptq(uint32_t const &source) {
    const unsigned int v0 = (source >> 0) & 0x0F;
    const unsigned int v1 = (source >> 4) & 0x0F;
    const unsigned int v2 = (source >> 8) & 0x0F;
    const unsigned int v3 = (source >> 12) & 0x0F;
    const unsigned int v4 = (source >> 16) & 0x0F;
    const unsigned int v5 = (source >> 20) & 0x0F;
    const unsigned int v6 = (source >> 24) & 0x0F;
    const unsigned int v7 = (source >> 28) & 0x0F;

    // NOTE: GPTQ uses unsigned q/z in [0,15]. No "-8" signed mapping here.
    const half hv0 = half(v0);
    const half hv1 = half(v1);
    const half hv2 = half(v2);
    const half hv3 = half(v3);
    const half hv4 = half(v4);
    const half hv5 = half(v5);
    const half hv6 = half(v6);
    const half hv7 = half(v7);

    uint4 result;
    __half2 *p = reinterpret_cast<__half2 *>(&result);
    p[0] = __halves2half2(hv0, hv4);
    p[1] = __halves2half2(hv1, hv5);
    p[2] = __halves2half2(hv2, hv6);
    p[3] = __halves2half2(hv3, hv7);
    return result;
}
