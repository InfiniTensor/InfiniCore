#include "../../../devices/ascend/ascend_kernel_common.h"

#include <type_traits>

using namespace AscendC;

template <typename T>
__aicore__ inline float mropeToFloat(T value) {
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        uint32_t bits = static_cast<uint32_t>(*reinterpret_cast<uint16_t *>(&value)) << 16;
        return *reinterpret_cast<float *>(&bits);
    } else {
        return static_cast<float>(value);
    }
}

template <typename T>
__aicore__ inline T mropeFromFloat(float value) {
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        uint32_t bits = *reinterpret_cast<uint32_t *>(&value);
        uint16_t upper = static_cast<uint16_t>(bits >> 16);
        return *reinterpret_cast<bfloat16_t *>(&upper);
    } else {
        return static_cast<T>(value);
    }
}

__aicore__ inline int64_t mropeLoadPosition(
    GM_ADDR positions, bool positions_i64, ptrdiff_t offset) {
    if (positions_i64) {
        return reinterpret_cast<__gm__ int64_t *>(positions)[offset];
    }
    return static_cast<int64_t>(
        reinterpret_cast<__gm__ int32_t *>(positions)[offset]);
}

template <typename T>
__aicore__ inline void mropeRotateHeads(
    GlobalTensor<T> &output,
    GlobalTensor<T> &input,
    GlobalTensor<T> &cos,
    GlobalTensor<T> &sin,
    GM_ADDR positions,
    bool positions_i64,
    size_t token,
    size_t num_heads,
    size_t head_size,
    size_t rotary_dim,
    size_t half_rotary_dim,
    ptrdiff_t output_stride_token,
    ptrdiff_t output_stride_head,
    ptrdiff_t input_stride_token,
    ptrdiff_t input_stride_head,
    ptrdiff_t cos_stride_position,
    ptrdiff_t sin_stride_position,
    ptrdiff_t positions_stride_axis,
    ptrdiff_t positions_stride_token,
    size_t max_position_embeddings,
    size_t section_t,
    size_t section_h,
    size_t section_w,
    bool positions_has_axes,
    bool interleaved) {

    for (size_t head = 0; head < num_heads; ++head) {
        ptrdiff_t out_base = static_cast<ptrdiff_t>(token) * output_stride_token
                           + static_cast<ptrdiff_t>(head) * output_stride_head;
        ptrdiff_t in_base = static_cast<ptrdiff_t>(token) * input_stride_token
                          + static_cast<ptrdiff_t>(head) * input_stride_head;
        for (size_t i = 0; i < half_rotary_dim; ++i) {
            size_t axis;
            if (interleaved) {
                bool h_mask = i % 3 == 1 && i < section_h * 3;
                bool w_mask = i % 3 == 2 && i < section_w * 3;
                axis = h_mask ? 1 : (w_mask ? 2 : 0);
            } else {
                axis = i < section_t ? 0
                                     : (i < section_t + section_h ? 1 : 2);
            }
            ptrdiff_t position_offset = positions_has_axes
                                          ? static_cast<ptrdiff_t>(axis) * positions_stride_axis
                                                + static_cast<ptrdiff_t>(token)
                                                      * positions_stride_token
                                          : static_cast<ptrdiff_t>(token)
                                                * positions_stride_token;
            int64_t raw_position = mropeLoadPosition(positions, positions_i64, position_offset);
            size_t position = raw_position >= 0
                                   && static_cast<size_t>(raw_position)
                                          < max_position_embeddings
                                ? static_cast<size_t>(raw_position)
                                : 0;
            float cos_value = mropeToFloat(cos.GetValue(
                static_cast<ptrdiff_t>(position) * cos_stride_position + i));
            float sin_value = mropeToFloat(sin.GetValue(
                static_cast<ptrdiff_t>(position) * sin_stride_position + i));
            float x0 = mropeToFloat(input.GetValue(in_base + i));
            float x1 = mropeToFloat(input.GetValue(in_base + i + half_rotary_dim));
            output.SetValue(
                out_base + i,
                mropeFromFloat<T>(x0 * cos_value - x1 * sin_value));
            output.SetValue(
                out_base + i + half_rotary_dim,
                mropeFromFloat<T>(x1 * cos_value + x0 * sin_value));
        }
        for (size_t i = rotary_dim; i < head_size; ++i) {
            output.SetValue(out_base + i, input.GetValue(in_base + i));
        }
    }
}

template <typename T>
__aicore__ inline void mropeProcess(
    GM_ADDR q_out_ptr, GM_ADDR k_out_ptr, GM_ADDR q_ptr, GM_ADDR k_ptr,
    GM_ADDR cos_ptr, GM_ADDR sin_ptr, GM_ADDR positions,
    bool positions_i64, size_t num_tokens, size_t num_q_heads,
    size_t num_kv_heads, size_t head_size, size_t rotary_dim,
    size_t half_rotary_dim, ptrdiff_t q_out_stride_token,
    ptrdiff_t q_out_stride_head, ptrdiff_t k_out_stride_token,
    ptrdiff_t k_out_stride_head, ptrdiff_t q_stride_token,
    ptrdiff_t q_stride_head, ptrdiff_t k_stride_token,
    ptrdiff_t k_stride_head, ptrdiff_t cos_stride_position,
    ptrdiff_t sin_stride_position, ptrdiff_t positions_stride_axis,
    ptrdiff_t positions_stride_token, size_t max_position_embeddings,
    size_t section_t, size_t section_h, size_t section_w,
    bool positions_has_axes, bool interleaved) {

    size_t token = GetBlockIdx();
    if (token >= num_tokens) {
        return;
    }
    GlobalTensor<T> q_out, k_out, q, k, cos, sin;
    q_out.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(q_out_ptr));
    k_out.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(k_out_ptr));
    q.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(q_ptr));
    k.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(k_ptr));
    cos.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(cos_ptr));
    sin.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(sin_ptr));
    mropeRotateHeads(
        q_out, q, cos, sin, positions, positions_i64, token, num_q_heads,
        head_size, rotary_dim, half_rotary_dim, q_out_stride_token,
        q_out_stride_head, q_stride_token, q_stride_head,
        cos_stride_position, sin_stride_position, positions_stride_axis,
        positions_stride_token, max_position_embeddings, section_t,
        section_h, section_w, positions_has_axes, interleaved);
    mropeRotateHeads(
        k_out, k, cos, sin, positions, positions_i64, token, num_kv_heads,
        head_size, rotary_dim, half_rotary_dim, k_out_stride_token,
        k_out_stride_head, k_stride_token, k_stride_head,
        cos_stride_position, sin_stride_position, positions_stride_axis,
        positions_stride_token, max_position_embeddings, section_t,
        section_h, section_w, positions_has_axes, interleaved);
}

#define DEFINE_MROPE_KERNEL(NAME, TYPE)                                     \
    __global__ __aicore__ void NAME(                                        \
        GM_ADDR q_out, GM_ADDR k_out, GM_ADDR q, GM_ADDR k, GM_ADDR cos,    \
        GM_ADDR sin, GM_ADDR positions, bool positions_i64,                 \
        size_t num_tokens, size_t num_q_heads, size_t num_kv_heads,         \
        size_t head_size, size_t rotary_dim, size_t half_rotary_dim,        \
        ptrdiff_t q_out_stride_token, ptrdiff_t q_out_stride_head,          \
        ptrdiff_t k_out_stride_token, ptrdiff_t k_out_stride_head,          \
        ptrdiff_t q_stride_token, ptrdiff_t q_stride_head,                  \
        ptrdiff_t k_stride_token, ptrdiff_t k_stride_head,                  \
        ptrdiff_t cos_stride_position, ptrdiff_t sin_stride_position,       \
        ptrdiff_t positions_stride_axis, ptrdiff_t positions_stride_token,  \
        size_t max_position_embeddings, size_t section_t, size_t section_h, \
        size_t section_w, bool positions_has_axes, bool interleaved) {      \
        mropeProcess<TYPE>(                                                 \
            q_out, k_out, q, k, cos, sin, positions, positions_i64,         \
            num_tokens, num_q_heads, num_kv_heads, head_size, rotary_dim,   \
            half_rotary_dim, q_out_stride_token, q_out_stride_head,         \
            k_out_stride_token, k_out_stride_head, q_stride_token,          \
            q_stride_head, k_stride_token, k_stride_head,                   \
            cos_stride_position, sin_stride_position,                       \
            positions_stride_axis, positions_stride_token,                  \
            max_position_embeddings, section_t, section_h, section_w,       \
            positions_has_axes, interleaved);                               \
    }

DEFINE_MROPE_KERNEL(mrope_half, half)
DEFINE_MROPE_KERNEL(mrope_bf16, bfloat16_t)
DEFINE_MROPE_KERNEL(mrope_float, float)
#undef DEFINE_MROPE_KERNEL

extern "C" infiniStatus_t mrope_ascend_kernel_launch(
    void *q_out, void *k_out, const void *q, const void *k,
    const void *cos, const void *sin, const void *positions,
    infiniDtype_t data_type, bool positions_i64,
    size_t num_tokens, size_t num_q_heads, size_t num_kv_heads,
    size_t head_size, size_t rotary_dim, size_t half_rotary_dim,
    ptrdiff_t q_out_stride_token, ptrdiff_t q_out_stride_head,
    ptrdiff_t k_out_stride_token, ptrdiff_t k_out_stride_head,
    ptrdiff_t q_stride_token, ptrdiff_t q_stride_head,
    ptrdiff_t k_stride_token, ptrdiff_t k_stride_head,
    ptrdiff_t cos_stride_position, ptrdiff_t sin_stride_position,
    ptrdiff_t positions_stride_axis, ptrdiff_t positions_stride_token,
    size_t max_position_embeddings, size_t section_t,
    size_t section_h, size_t section_w,
    bool positions_has_axes, bool interleaved, void *stream) {

    if (num_tokens == 0) {
        return INFINI_STATUS_SUCCESS;
    }
    uint32_t blocks = static_cast<uint32_t>(num_tokens);
#define LAUNCH_MROPE(DTYPE, NAME)                                       \
    case DTYPE:                                                         \
        NAME<<<blocks, nullptr, stream>>>(                              \
            q_out, k_out, const_cast<void *>(q), const_cast<void *>(k), \
            const_cast<void *>(cos), const_cast<void *>(sin),           \
            const_cast<void *>(positions), positions_i64, num_tokens,   \
            num_q_heads, num_kv_heads, head_size, rotary_dim,           \
            half_rotary_dim, q_out_stride_token, q_out_stride_head,     \
            k_out_stride_token, k_out_stride_head, q_stride_token,      \
            q_stride_head, k_stride_token, k_stride_head,               \
            cos_stride_position, sin_stride_position,                   \
            positions_stride_axis, positions_stride_token,              \
            max_position_embeddings, section_t, section_h, section_w,   \
            positions_has_axes, interleaved);                           \
        return INFINI_STATUS_SUCCESS;
    switch (data_type) {
        LAUNCH_MROPE(INFINI_DTYPE_F16, mrope_half)
        LAUNCH_MROPE(INFINI_DTYPE_BF16, mrope_bf16)
        LAUNCH_MROPE(INFINI_DTYPE_F32, mrope_float)
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
#undef LAUNCH_MROPE
}
