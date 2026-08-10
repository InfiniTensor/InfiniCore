#include "../../../devices/ascend/ascend_kernel_common.h"

#include <math.h>
#include <type_traits>

using namespace AscendC;

template <typename T>
__aicore__ inline float gatingToFloat(T value) {
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        uint32_t bits = static_cast<uint32_t>(
                            *reinterpret_cast<uint16_t *>(&value))
                     << 16;
        return *reinterpret_cast<float *>(&bits);
    } else {
        return static_cast<float>(value);
    }
}

template <typename T>
__aicore__ inline float gatingRoundToInput(float value) {
    // vLLM stores sigmoid(beta) in the model dtype before the recurrent
    // update. Preserve that rounding point while keeping the public output F32.
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        uint32_t bits = *reinterpret_cast<uint32_t *>(&value);
        // BF16 round-to-nearest-even. Returning the expanded FP32 value avoids
        // an unsupported scalar float -> bfloat16_t cast in the AscendC backend.
        uint32_t rounding_bias = 0x7fffu + ((bits >> 16) & 1u);
        bits += rounding_bias;
        bits &= 0xffff0000u;
        return *reinterpret_cast<float *>(&bits);
    } else {
        return gatingToFloat(static_cast<T>(value));
    }
}

// AscendC does not expose the host scalar exp/log functions to AICore code.
// These helpers use range reduction and short polynomials instead.
__aicore__ inline float gatingExp(float x) {
    if (x > 88.0f) {
        x = 88.0f;
    } else if (x < -87.0f) {
        return 0.0f;
    }

    constexpr float inv_ln2 = 1.4426950408889634f;
    constexpr float ln2_hi = 0.693145751953125f;
    constexpr float ln2_lo = 1.428606765330187e-6f;
    float scaled = x * inv_ln2;
    int32_t exponent = static_cast<int32_t>(scaled);
    if (scaled < static_cast<float>(exponent)) {
        --exponent;
    }
    float r = x - static_cast<float>(exponent) * ln2_hi
            - static_cast<float>(exponent) * ln2_lo;
    float polynomial = 1.0f
                     + r * (1.0f + r * (0.5f + r * (0.1666666716f + r * (0.0416666679f + r * (0.0083333338f + r * 0.0013888889f)))));
    uint32_t scale_bits = static_cast<uint32_t>(exponent + 127) << 23;
    float scale = *reinterpret_cast<float *>(&scale_bits);
    return polynomial * scale;
}

__aicore__ inline float gatingLog(float x) {
    uint32_t bits = *reinterpret_cast<uint32_t *>(&x);
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xff) - 127;
    bits = (bits & 0x007fffffu) | 0x3f800000u;
    float mantissa = *reinterpret_cast<float *>(&bits);

    float z = (mantissa - 1.0f) / (mantissa + 1.0f);
    float z2 = z * z;
    float series = z * (1.0f + z2 * (0.3333333333f + z2 * (0.2f + z2 * (0.1428571429f + z2 * (0.1111111111f + z2 * (0.0909090909f + z2 * 0.0769230769f))))));
    return static_cast<float>(exponent) * 0.6931471805599453f
         + 2.0f * series;
}

__aicore__ inline float gatingSigmoid(float x) {
    if (x >= 0.0f) {
        float z = gatingExp(-x);
        return 1.0f / (1.0f + z);
    }
    float z = gatingExp(x);
    return z / (1.0f + z);
}

__aicore__ inline float gatingSoftplus(float x, float beta, float threshold) {
    float bx = beta * x;
    if (bx > threshold) {
        return x;
    }
    if (bx < -20.0f) {
        return gatingExp(bx) / beta;
    }
    return gatingLog(1.0f + gatingExp(bx)) / beta;
}

template <typename T>
__aicore__ inline void fused_gated_delta_net_gating_process(
    GM_ADDR g_ptr,
    GM_ADDR beta_output_ptr,
    GM_ADDR A_log_ptr,
    GM_ADDR a_ptr,
    GM_ADDR b_ptr,
    GM_ADDR dt_bias_ptr,
    size_t total,
    size_t seq_len,
    size_t hidden,
    ptrdiff_t g_s0,
    ptrdiff_t g_s1,
    ptrdiff_t g_s2,
    ptrdiff_t beta_s0,
    ptrdiff_t beta_s1,
    ptrdiff_t beta_s2,
    ptrdiff_t A_log_s0,
    ptrdiff_t a_s0,
    ptrdiff_t a_s1,
    ptrdiff_t a_s2,
    ptrdiff_t b_s0,
    ptrdiff_t b_s1,
    ptrdiff_t b_s2,
    ptrdiff_t dt_bias_s0,
    float beta,
    float threshold) {

    GlobalTensor<float> g;
    GlobalTensor<float> beta_output;
    GlobalTensor<T> A_log;
    GlobalTensor<T> a;
    GlobalTensor<T> b;
    GlobalTensor<T> dt_bias;
    g.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(g_ptr));
    beta_output.SetGlobalBuffer(
        reinterpret_cast<__gm__ float *>(beta_output_ptr));
    A_log.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(A_log_ptr));
    a.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(a_ptr));
    b.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(b_ptr));
    dt_bias.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(dt_bias_ptr));

    for (size_t linear = 0; linear < total; ++linear) {
        size_t h = linear % hidden;
        size_t tmp = linear / hidden;
        size_t s = tmp % seq_len;
        size_t batch = tmp / seq_len;

        ptrdiff_t g_off = static_cast<ptrdiff_t>(batch) * g_s0
                        + static_cast<ptrdiff_t>(s) * g_s1
                        + static_cast<ptrdiff_t>(h) * g_s2;
        ptrdiff_t beta_off = static_cast<ptrdiff_t>(batch) * beta_s0
                           + static_cast<ptrdiff_t>(s) * beta_s1
                           + static_cast<ptrdiff_t>(h) * beta_s2;
        ptrdiff_t a_off = static_cast<ptrdiff_t>(batch) * a_s0
                        + static_cast<ptrdiff_t>(s) * a_s1
                        + static_cast<ptrdiff_t>(h) * a_s2;
        ptrdiff_t b_off = static_cast<ptrdiff_t>(batch) * b_s0
                        + static_cast<ptrdiff_t>(s) * b_s1
                        + static_cast<ptrdiff_t>(h) * b_s2;

        float x = gatingToFloat(a.GetValue(a_off))
                + gatingToFloat(
                      dt_bias.GetValue(static_cast<ptrdiff_t>(h) * dt_bias_s0));
        float decay = -gatingExp(gatingToFloat(
            A_log.GetValue(static_cast<ptrdiff_t>(h) * A_log_s0)));
        g.SetValue(g_off, decay * gatingSoftplus(x, beta, threshold));
        float beta_value = gatingSigmoid(gatingToFloat(b.GetValue(b_off)));
        // vLLM's Qwen3.5 decode fusion keeps sigmoid(b) in FP32, while its
        // prefill path materializes beta in the model dtype. InfiniLM lays
        // decode out as [active_sequences, 1, heads].
        if (seq_len != 1) {
            beta_value = gatingRoundToInput<T>(beta_value);
        }
        beta_output.SetValue(beta_off, beta_value);
    }
}

#define DEFINE_GATING_KERNEL(NAME, TYPE)                                  \
    __global__ __aicore__ void NAME(                                      \
        GM_ADDR g, GM_ADDR beta_output, GM_ADDR A_log, GM_ADDR a,         \
        GM_ADDR b, GM_ADDR dt_bias, size_t total, size_t seq_len,         \
        size_t hidden, ptrdiff_t g_s0, ptrdiff_t g_s1, ptrdiff_t g_s2,    \
        ptrdiff_t beta_s0, ptrdiff_t beta_s1, ptrdiff_t beta_s2,          \
        ptrdiff_t A_log_s0, ptrdiff_t a_s0, ptrdiff_t a_s1,               \
        ptrdiff_t a_s2, ptrdiff_t b_s0, ptrdiff_t b_s1, ptrdiff_t b_s2,   \
        ptrdiff_t dt_bias_s0, float beta, float threshold) {              \
        fused_gated_delta_net_gating_process<TYPE>(                       \
            g, beta_output, A_log, a, b, dt_bias, total, seq_len, hidden, \
            g_s0, g_s1, g_s2, beta_s0, beta_s1, beta_s2, A_log_s0,        \
            a_s0, a_s1, a_s2, b_s0, b_s1, b_s2, dt_bias_s0, beta,         \
            threshold);                                                   \
    }

DEFINE_GATING_KERNEL(fused_gating_half, half)
DEFINE_GATING_KERNEL(fused_gating_float, float)
DEFINE_GATING_KERNEL(fused_gating_bf16, bfloat16_t)
#undef DEFINE_GATING_KERNEL

extern "C" infiniStatus_t fused_gated_delta_net_gating_kernel_launch(
    void *g,
    void *beta_output,
    const void *A_log,
    const void *a,
    const void *b,
    const void *dt_bias,
    infiniDtype_t dtype,
    size_t total,
    size_t seq_len,
    size_t hidden,
    ptrdiff_t g_s0,
    ptrdiff_t g_s1,
    ptrdiff_t g_s2,
    ptrdiff_t beta_s0,
    ptrdiff_t beta_s1,
    ptrdiff_t beta_s2,
    ptrdiff_t A_log_s0,
    ptrdiff_t a_s0,
    ptrdiff_t a_s1,
    ptrdiff_t a_s2,
    ptrdiff_t b_s0,
    ptrdiff_t b_s1,
    ptrdiff_t b_s2,
    ptrdiff_t dt_bias_s0,
    float beta,
    float threshold,
    void *stream) {

    if (total == 0) {
        return INFINI_STATUS_SUCCESS;
    }

#define LAUNCH_GATING(DTYPE, NAME)                                 \
    case DTYPE:                                                    \
        NAME<<<1, nullptr, stream>>>(                              \
            g, beta_output, const_cast<void *>(A_log),             \
            const_cast<void *>(a), const_cast<void *>(b),          \
            const_cast<void *>(dt_bias), total, seq_len, hidden,   \
            g_s0, g_s1, g_s2, beta_s0, beta_s1, beta_s2, A_log_s0, \
            a_s0, a_s1, a_s2, b_s0, b_s1, b_s2, dt_bias_s0, beta,  \
            threshold);                                            \
        return INFINI_STATUS_SUCCESS;

    switch (dtype) {
        LAUNCH_GATING(INFINI_DTYPE_F16, fused_gating_half)
        LAUNCH_GATING(INFINI_DTYPE_BF16, fused_gating_bf16)
        LAUNCH_GATING(INFINI_DTYPE_F32, fused_gating_float)
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

#undef LAUNCH_GATING
}
