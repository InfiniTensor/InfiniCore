#pragma once
#include <musa_fp16.h> // __half / __half2

/**
 * @brief Unpack 8x 4-bit unsigned integers (0..15) from a uint32_t into 8 half values.
 *
 * GPTQ dequant is applied outside this helper (aligned with nvidia impl):
 *   out = (q - (z + 1)) * s
 *
 * Output order matches the interleaved half2 packing:
 *   (v0,v4), (v1,v5), (v2,v6), (v3,v7)
 */
__device__ __forceinline__ uint4 dequantize_s4_to_fp16x2_gptq(uint32_t const &source) {
    // unpack 8 nibbles: v0..v7 in [0, 15]
    const unsigned int v0 = (source >> 0) & 0x0F;
    const unsigned int v1 = (source >> 4) & 0x0F;
    const unsigned int v2 = (source >> 8) & 0x0F;
    const unsigned int v3 = (source >> 12) & 0x0F;
    const unsigned int v4 = (source >> 16) & 0x0F;
    const unsigned int v5 = (source >> 20) & 0x0F;
    const unsigned int v6 = (source >> 24) & 0x0F;
    const unsigned int v7 = (source >> 28) & 0x0F;

    // NOTE: no "-8" offset here (unlike signed s4). GPTQ uses unsigned q/z.
    const __half hv0 = __half(v0);
    const __half hv1 = __half(v1);
    const __half hv2 = __half(v2);
    const __half hv3 = __half(v3);
    const __half hv4 = __half(v4);
    const __half hv5 = __half(v5);
    const __half hv6 = __half(v6);
    const __half hv7 = __half(v7);

    uint4 result;
    __half2 *result_ptr = reinterpret_cast<__half2 *>(&result);
    result_ptr[0] = __halves2half2(hv0, hv4);
    result_ptr[1] = __halves2half2(hv1, hv5);
    result_ptr[2] = __halves2half2(hv2, hv6);
    result_ptr[3] = __halves2half2(hv3, hv7);
    return result;
}