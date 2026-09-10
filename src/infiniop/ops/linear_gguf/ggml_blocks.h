// Device-independent GGUF block decoders for the linear_gguf op.
//
// Scope: the four GGML block types route B stores verbatim, i.e. Q8_0 (8),
// Q4_K (12), Q5_K (13), Q6_K (14). Everything else is rejected upstream, at
// packaging time, so no other decoder lives here.
//
// Numerics: scalar port of dequantize_row_q8_0/q4_K/q5_K/q6_K in llama.cpp
// ggml/src/ggml-quants.c. The floating-point operation order is preserved
// on purpose: the K-quant path forms (d * scale) and (dmin * min) first and
// then touches the quants, so a re-associated expression is not bit-exact and
// the fp32 -> bf16 round would land on a different neighbour. Callers that
// need bit-exact agreement with llama.cpp must go through these functions
// instead of re-deriving the formula.
//
// Alignment: fields are read byte by byte. Route B packs the block rows of one
// GGUF tensor back to back into a single uint8 tensor, so a block only starts
// at 2-byte alignment (Q6_K's half sits at offset 208). Reinterpreting the
// buffer as block_q6_K et al. would be undefined behaviour on some backends.
//
// Portability: no CUDA-only type or intrinsic appears here, and bf16 values
// travel as uint16_t bit patterns, so the same header serves .cu kernels,
// host-side tests and future non-NVIDIA backends. It is the single source of
// truth for block decoding; kernels that decode inline for bandwidth must be
// validated against these functions.
#ifndef __GGML_BLOCKS_H__
#define __GGML_BLOCKS_H__

#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
// host+device so a .cu translation unit can also decode on the host side
// (test drivers, CPU fallback) without a second copy of the logic.
#define GGML_BLK_HOST_DEVICE __host__ __device__ __forceinline__
#else
#define GGML_BLK_HOST_DEVICE inline
#endif

namespace ggml_blocks {

// enum ggml_type values, pinned to gguf.h / GGML_QUANT_SIZES.
enum GgmlType : int32_t {
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
};

// Elements per super-block / per Q8_0 block, and bytes per block.
constexpr int32_t QK_K = 256;
constexpr int32_t QK8_0 = 32;
constexpr int32_t K_SCALE_SIZE = 12;
constexpr int32_t SIZE_Q8_0 = 34;    // 2 + 32
constexpr int32_t SIZE_Q4_K = 144;   // 4 + 12 + 128
constexpr int32_t SIZE_Q5_K = 176;   // 4 + 12 + 32 + 128
constexpr int32_t SIZE_Q6_K = 210;   // 128 + 64 + 16 + 2

// Number of elements one block of `type` decodes to, or -1 if `type` has no
// decoder here.
GGML_BLK_HOST_DEVICE int32_t block_elems(int32_t type) {
    return type == GGML_TYPE_Q8_0 ? QK8_0
                                  : (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K ||
                                             type == GGML_TYPE_Q6_K
                                         ? QK_K
                                         : -1);
}

// Bytes one block of `type` occupies in the blob, or -1 if unsupported.
GGML_BLK_HOST_DEVICE int32_t block_bytes(int32_t type) {
    switch (type) {
        case GGML_TYPE_Q8_0: return SIZE_Q8_0;
        case GGML_TYPE_Q4_K: return SIZE_Q4_K;
        case GGML_TYPE_Q5_K: return SIZE_Q5_K;
        case GGML_TYPE_Q6_K: return SIZE_Q6_K;
        default: return -1;
    }
}

GGML_BLK_HOST_DEVICE uint16_t read_u16(const uint8_t *p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

// IEEE binary16 -> binary32, exact, subnormal and NaN safe. Equivalent to
// ggml GGML_FP16_TO_FP32 / __half2float without depending on either.
GGML_BLK_HOST_DEVICE float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h >> 15) << 31;
    const uint32_t exp = (uint32_t)(h >> 10) & 0x1Fu;
    const uint32_t mant = (uint32_t)h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;  // +-0
        } else {
            // Normalise the subnormal: value == mant * 2^-24.
            int e = -14;
            uint32_t m = mant;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                --e;
            }
            bits = sign | ((uint32_t)(e + 127) << 23) | ((m & 0x3FFu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (mant << 13);  // inf / nan
    } else {
        bits = sign | ((exp + 112) << 23) | (mant << 13);
    }
    float f;
    static_assert(sizeof(f) == 4, "float is assumed to be IEEE binary32");
    std::memcpy(&f, &bits, 4);
    return f;
}

// binary32 -> bf16 bit pattern, round-to-nearest-even, matching
// __float2bfloat16 / torch.Tensor.to(bfloat16) behaviour on finite inputs.
GGML_BLK_HOST_DEVICE uint16_t float_to_bf16(float f) {
    uint32_t bits;
    static_assert(sizeof(f) == 4, "float is assumed to be IEEE binary32");
    std::memcpy(&bits, &f, 4);
    const uint32_t exp = (bits >> 23) & 0xFFu;
    if (exp == 0xFFu && (bits & 0x7FFFFFu) != 0) {
        // Keep the result a NaN the way the hardware conversion does.
        return (uint16_t)((bits >> 16) | 0x0040u);
    }
    const uint32_t bias = 0x7FFFu + ((bits >> 16) & 1u);
    return (uint16_t)((bits + bias) >> 16);
}

// 6-bit scale / min pair j of a Q4_K or Q5_K super-block, as in ggml-quants.c.
GGML_BLK_HOST_DEVICE void get_scale_min_k4(int j, const uint8_t *q,
                                           uint8_t &d, uint8_t &m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

// Field offsets inside one block (see the alignment note above).
constexpr int32_t Q8_0_OFF_D = 0, Q8_0_OFF_QS = 2;
constexpr int32_t QK_OFF_DMIN = 2, QK_OFF_SCALES = 4;
constexpr int32_t Q4K_OFF_QS = 16;
constexpr int32_t Q5K_OFF_QH = 16, Q5K_OFF_QS = 48;
constexpr int32_t Q6K_OFF_QH = 128, Q6K_OFF_SCALES = 192, Q6K_OFF_D = 208;

// out: QK8_0 fp32 values.
GGML_BLK_HOST_DEVICE void decode_q8_0(const uint8_t *blk, float *out) {
    const float d = half_to_float(read_u16(blk + Q8_0_OFF_D));
    const uint8_t *qs = blk + Q8_0_OFF_QS;
    for (int32_t j = 0; j < QK8_0; ++j) {
        out[j] = (float)(int8_t)qs[j] * d;
    }
}

// out: QK_K fp32 values.
GGML_BLK_HOST_DEVICE void decode_q4_K(const uint8_t *blk, float *out) {
    const float d = half_to_float(read_u16(blk));
    const float min = half_to_float(read_u16(blk + QK_OFF_DMIN));
    const uint8_t *scales = blk + QK_OFF_SCALES;
    const uint8_t *q = blk + Q4K_OFF_QS;
    for (int32_t j = 0, is = 0; j < QK_K; j += 64, is += 2) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * sc;
        const float m1 = min * m;
        get_scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * sc;
        const float m2 = min * m;
        for (int32_t l = 0; l < 32; ++l) out[j + l] = d1 * (float)(q[l] & 0xF) - m1;
        for (int32_t l = 0; l < 32; ++l) out[j + 32 + l] = d2 * (float)(q[l] >> 4) - m2;
        q += 32;
    }
}

// out: QK_K fp32 values.
GGML_BLK_HOST_DEVICE void decode_q5_K(const uint8_t *blk, float *out) {
    const float d = half_to_float(read_u16(blk));
    const float min = half_to_float(read_u16(blk + QK_OFF_DMIN));
    const uint8_t *scales = blk + QK_OFF_SCALES;
    const uint8_t *qh = blk + Q5K_OFF_QH;
    const uint8_t *ql = blk + Q5K_OFF_QS;
    uint8_t u1 = 1, u2 = 2;
    for (int32_t j = 0, is = 0; j < QK_K; j += 64, is += 2) {
        uint8_t sc, m;
        get_scale_min_k4(is + 0, scales, sc, m);
        const float d1 = d * sc;
        const float m1 = min * m;
        get_scale_min_k4(is + 1, scales, sc, m);
        const float d2 = d * sc;
        const float m2 = min * m;
        for (int32_t l = 0; l < 32; ++l)
            out[j + l] = d1 * (float)((ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0)) - m1;
        for (int32_t l = 0; l < 32; ++l)
            out[j + 32 + l] = d2 * (float)((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
        ql += 32;
        u1 <<= 2;
        u2 <<= 2;
    }
    // qh never advances: the 5th bit of the 64 elements of one iteration lives
    // in two different bit positions of the same 32 qh bytes, which is what the
    // shifting u1/u2 masks walk instead of a second pointer.
}

// out: QK_K fp32 values.
GGML_BLK_HOST_DEVICE void decode_q6_K(const uint8_t *blk, float *out) {
    const float d = half_to_float(read_u16(blk + Q6K_OFF_D));
    const uint8_t *ql = blk;
    const uint8_t *qh = blk + Q6K_OFF_QH;
    const int8_t *sc = (const int8_t *)(blk + Q6K_OFF_SCALES);
    for (int32_t n = 0, y = 0; n < QK_K; n += 128, y += 128) {
        for (int32_t l = 0; l < 32; ++l) {
            const int is = l / 16;
            const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
            const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
            const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
            const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
            out[y + l + 0] = d * sc[is + 0] * q1;
            out[y + l + 32] = d * sc[is + 2] * q2;
            out[y + l + 64] = d * sc[is + 4] * q3;
            out[y + l + 96] = d * sc[is + 6] * q4;
        }
        ql += 64;
        qh += 32;
        sc += 8;
    }
}

// Decode `n_blocks` consecutive blocks of `type` into fp32.
// Returns false without writing when `type` has no decoder.
GGML_BLK_HOST_DEVICE bool decode_blocks(int32_t type, const uint8_t *blk,
                                        int64_t n_blocks, float *out) {
    const int32_t bytes = block_bytes(type);
    const int32_t elems = block_elems(type);
    if (bytes < 0) return false;
    for (int64_t i = 0; i < n_blocks; ++i) {
        float *o = out + i * elems;
        switch (type) {
            case GGML_TYPE_Q8_0: decode_q8_0(blk + i * bytes, o); break;
            case GGML_TYPE_Q4_K: decode_q4_K(blk + i * bytes, o); break;
            case GGML_TYPE_Q5_K: decode_q5_K(blk + i * bytes, o); break;
            case GGML_TYPE_Q6_K: decode_q6_K(blk + i * bytes, o); break;
            default: return false;
        }
    }
    return true;
}

// Same as decode_blocks, storing bf16 bit patterns; `tmp` must hold
// block_elems(type) fp32 values.
template <int32_t MAX_ELEMS>
GGML_BLK_HOST_DEVICE bool decode_blocks_bf16(int32_t type, const uint8_t *blk,
                                             int64_t n_blocks, uint16_t *out) {
    const int32_t bytes = block_bytes(type);
    const int32_t elems = block_elems(type);
    if (bytes < 0 || elems > MAX_ELEMS) return false;
    for (int64_t i = 0; i < n_blocks; ++i) {
        float tmp[MAX_ELEMS];
        switch (type) {
            case GGML_TYPE_Q8_0: decode_q8_0(blk + i * bytes, tmp); break;
            case GGML_TYPE_Q4_K: decode_q4_K(blk + i * bytes, tmp); break;
            case GGML_TYPE_Q5_K: decode_q5_K(blk + i * bytes, tmp); break;
            case GGML_TYPE_Q6_K: decode_q6_K(blk + i * bytes, tmp); break;
            default: return false;
        }
        uint16_t *o = out + i * elems;
        for (int32_t j = 0; j < elems; ++j) o[j] = float_to_bf16(tmp[j]);
    }
    return true;
}

#undef GGML_BLK_HOST_DEVICE

}  // namespace ggml_blocks

#endif
