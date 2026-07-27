#include "../../../devices/ascend/ascend_kernel_common.h"
#include <type_traits>
using namespace AscendC;

template <typename T>
__aicore__ inline float causalDataToFloat(T value) {
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
__aicore__ inline T causalFloatToData(float value) {
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        uint32_t bits = *reinterpret_cast<uint32_t *>(&value);
        uint16_t upper = static_cast<uint16_t>(bits >> 16);
        return *reinterpret_cast<bfloat16_t *>(&upper);
    } else {
        return static_cast<T>(value);
    }
}

__aicore__ inline int64_t causalLoadIndex(
    GM_ADDR ptr, bool is_i64, int index, int fallback) {
    if (ptr == nullptr) {
        return static_cast<int64_t>(fallback);
    }
    if (is_i64) {
        return static_cast<int64_t>(
            reinterpret_cast<__gm__ int64_t *>(ptr)[index]);
    }
    return static_cast<int64_t>(
        reinterpret_cast<__gm__ int32_t *>(ptr)[index]);
}

template <typename T>
__aicore__ inline float causalLoadHistory(
    GlobalTensor<T> &conv_state, GlobalTensor<T> &qkv,
    int64_t history_pos, int64_t token_begin, int token_batch,
    int channel, ptrdiff_t state_base, ptrdiff_t state_s2,
    ptrdiff_t qkv_s0, ptrdiff_t qkv_s1, ptrdiff_t qkv_s2) {
    constexpr int64_t STATE_LEN = 3;
    if (history_pos < STATE_LEN) {
        return causalDataToFloat(
            conv_state.GetValue(state_base + history_pos * state_s2));
    }
    int64_t token_idx = token_begin + history_pos - STATE_LEN;
    ptrdiff_t qkv_offset = static_cast<ptrdiff_t>(token_batch) * qkv_s0
                         + static_cast<ptrdiff_t>(token_idx) * qkv_s1
                         + static_cast<ptrdiff_t>(channel) * qkv_s2;
    return causalDataToFloat(qkv.GetValue(qkv_offset));
}

template <typename T>
class CausalConv1dKernel {
public:
    __aicore__ inline void process(
        GM_ADDR out_ptr, GM_ADDR conv_state_ptr, GM_ADDR final_state_ptr,
        GM_ADDR qkv_ptr, GM_ADDR weight_ptr, GM_ADDR bias_ptr,
        GM_ADDR cu_seqlens, GM_ADDR initial_indices, GM_ADDR final_indices,
        bool has_bias, bool has_cu, bool cu_i64, bool initial_i64,
        bool final_i64, bool indexed_pool, bool update_state_only, bool fuse_state_update,
        size_t request_count, size_t T_tokens,
        size_t C, size_t total_tokens, size_t pool_size,
        ptrdiff_t out_s0, ptrdiff_t out_s1, ptrdiff_t out_s2,
        ptrdiff_t state_s0, ptrdiff_t state_s1, ptrdiff_t state_s2,
        ptrdiff_t final_s0, ptrdiff_t final_s1, ptrdiff_t final_s2,
        ptrdiff_t qkv_s0, ptrdiff_t qkv_s1, ptrdiff_t qkv_s2,
        ptrdiff_t weight_s0, ptrdiff_t weight_s2, ptrdiff_t bias_s0) {
        GlobalTensor<T> out, conv_state, final_state, qkv, weight, bias;
        out.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(out_ptr));
        conv_state.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(conv_state_ptr));
        final_state.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(final_state_ptr));
        qkv.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(qkv_ptr));
        weight.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(weight_ptr));
        bias.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(bias_ptr));

        const size_t block = GetBlockIdx();
        const size_t block_count = GetBlockNum();
        const bool vector_fast_path = fuse_state_update && !update_state_only
                                   && block_count != 0 && C % block_count == 0
                                   && ((C / block_count) * sizeof(T)) % BYTE_ALIGN == 0
                                   && out_s2 == 1 && qkv_s2 == 1
                                   && state_s1 == 3 && state_s2 == 1
                                   && weight_s0 == 4 && weight_s2 == 1
                                   && (!has_bias || bias_s0 == 1);
        const size_t fast_tile_len = vector_fast_path ? C / block_count : 1;
        const size_t fast_copy_len = alignTileLen<T>(fast_tile_len, BYTE_ALIGN);
        const size_t fast_channel_begin = block * fast_tile_len;

        TPipe pipe;
        const event_t mte2_to_v = static_cast<event_t>(
            pipe.FetchEventID(HardEvent::MTE2_V));
        const event_t v_to_mte3 = static_cast<event_t>(
            pipe.FetchEventID(HardEvent::V_MTE3));
        const event_t s_to_v = static_cast<event_t>(
            pipe.FetchEventID(HardEvent::S_V));
        const event_t mte2_to_s = static_cast<event_t>(
            pipe.FetchEventID(HardEvent::MTE2_S));
        const event_t s_to_mte3 = static_cast<event_t>(
            pipe.FetchEventID(HardEvent::S_MTE3));
        const event_t v_to_s = static_cast<event_t>(
            pipe.FetchEventID(HardEvent::V_S));
        const event_t mte3_to_v = static_cast<event_t>(
            pipe.FetchEventID(HardEvent::MTE3_V));
        const event_t v_to_mte2 = static_cast<event_t>(
            pipe.FetchEventID(HardEvent::V_MTE2));
        TBuf<QuePosition::VECCALC> x_data_buf, weight_data_buf;
        TBuf<QuePosition::VECCALC> state_raw_buf, weight_raw_buf, qkv_raw_buf;
        TBuf<QuePosition::VECCALC> bias_data_buf, out_data_buf;
        TBuf<QuePosition::VECCALC> gather_offsets_buf;
        TBuf<QuePosition::VECCALC> x_float_buf, weight_float_buf;
        TBuf<QuePosition::VECCALC> acc_float_buf, tmp_float_buf;
        if (vector_fast_path) {
            pipe.InitBuffer(x_data_buf, fast_copy_len * sizeof(T));
            pipe.InitBuffer(weight_data_buf, 4 * fast_copy_len * sizeof(T));
            pipe.InitBuffer(state_raw_buf, 3 * fast_copy_len * sizeof(T));
            pipe.InitBuffer(weight_raw_buf, 4 * fast_copy_len * sizeof(T));
            pipe.InitBuffer(qkv_raw_buf, fast_copy_len * sizeof(T));
            pipe.InitBuffer(bias_data_buf, fast_copy_len * sizeof(T));
            pipe.InitBuffer(out_data_buf, fast_copy_len * sizeof(T));
            pipe.InitBuffer(gather_offsets_buf, fast_copy_len * sizeof(uint32_t));
            pipe.InitBuffer(x_float_buf, fast_copy_len * sizeof(float));
            pipe.InitBuffer(weight_float_buf, 4 * fast_copy_len * sizeof(float));
            pipe.InitBuffer(acc_float_buf, fast_copy_len * sizeof(float));
            pipe.InitBuffer(tmp_float_buf, fast_copy_len * sizeof(float));

            LocalTensor<T> weight_data = weight_data_buf.Get<T>();
            LocalTensor<T> weight_raw = weight_raw_buf.Get<T>();
            LocalTensor<float> weight_float = weight_float_buf.Get<float>();
            LocalTensor<uint32_t> gather_offsets = gather_offsets_buf.Get<uint32_t>();
            if (weight_s0 != 1) {
                DataCopy(weight_raw,
                         weight[static_cast<ptrdiff_t>(fast_channel_begin) * weight_s0],
                         4 * fast_copy_len);
                SetFlag<HardEvent::MTE2_V>(mte2_to_v);
                WaitFlag<HardEvent::MTE2_V>(mte2_to_v);
                ArithProgression<int32_t>(
                    gather_offsets.ReinterpretCast<int32_t>(), 0,
                    static_cast<int32_t>(4 * sizeof(T)), fast_tile_len);
                PipeBarrier<PIPE_V>();
                for (int k = 0; k < 4; ++k) {
                    if constexpr (std::is_same<T, float>::value) {
                        Gather(weight_float[k * fast_copy_len], weight_raw,
                               gather_offsets, k * sizeof(T), fast_tile_len);
                    } else {
                        Gather(weight_data[k * fast_copy_len], weight_raw,
                               gather_offsets, k * sizeof(T), fast_tile_len);
                        PipeBarrier<PIPE_V>();
                        Cast(weight_float[k * fast_copy_len],
                             weight_data[k * fast_copy_len],
                             RoundMode::CAST_NONE, fast_copy_len);
                    }
                    PipeBarrier<PIPE_V>();
                }
            } else {
                for (int k = 0; k < 4; ++k) {
                    const ptrdiff_t weight_offset = k * weight_s2 + static_cast<ptrdiff_t>(fast_channel_begin);
                    if constexpr (std::is_same<T, float>::value) {
                        DataCopy(weight_float[k * fast_copy_len],
                                 weight[weight_offset], fast_copy_len);
                    } else {
                        DataCopy(weight_data[k * fast_copy_len],
                                 weight[weight_offset], fast_copy_len);
                    }
                    SetFlag<HardEvent::MTE2_V>(mte2_to_v);
                    WaitFlag<HardEvent::MTE2_V>(mte2_to_v);
                    if constexpr (!std::is_same<T, float>::value) {
                        Cast(weight_float[k * fast_copy_len],
                             weight_data[k * fast_copy_len],
                             RoundMode::CAST_NONE, fast_copy_len);
                    }
                }
            }
            ArithProgression<int32_t>(
                gather_offsets.ReinterpretCast<int32_t>(), 0,
                static_cast<int32_t>(3 * sizeof(T)), fast_tile_len);
            PipeBarrier<PIPE_V>();
        }
        for (size_t request = 0; request < request_count; ++request) {
            int64_t token_begin = 0;
            int64_t token_end = static_cast<int64_t>(T_tokens);
            int token_batch = static_cast<int>(request);
            if (has_cu) {
                token_begin = causalLoadIndex(cu_seqlens, cu_i64, request, 0);
                token_end = causalLoadIndex(cu_seqlens, cu_i64, request + 1, 0);
                token_batch = 0;
                if (token_begin < 0 || token_end < token_begin
                    || token_end > static_cast<int64_t>(total_tokens)) {
                    return;
                }
            }
            const int64_t request_len = token_end - token_begin;
            const int64_t read_slot = indexed_pool
                                        ? causalLoadIndex(initial_indices, initial_i64, request, request)
                                        : static_cast<int64_t>(request);
            const int64_t write_slot = indexed_pool && final_indices != nullptr
                                         ? causalLoadIndex(final_indices, final_i64, request, request)
                                         : static_cast<int64_t>(request);
            if (read_slot < 0 || write_slot < 0
                || read_slot >= static_cast<int64_t>(pool_size)
                || (final_indices != nullptr
                    && write_slot >= static_cast<int64_t>(pool_size))) {
                return;
            }

            if (!update_state_only) {
                if (vector_fast_path) {
                    LocalTensor<T> x_data = x_data_buf.Get<T>();
                    LocalTensor<T> state_raw = state_raw_buf.Get<T>();
                    LocalTensor<T> weight_raw = weight_raw_buf.Get<T>();
                    LocalTensor<T> qkv_raw = qkv_raw_buf.Get<T>();
                    LocalTensor<T> bias_data = bias_data_buf.Get<T>();
                    LocalTensor<T> out_data = out_data_buf.Get<T>();
                    LocalTensor<float> x_float = x_float_buf.Get<float>();
                    LocalTensor<uint32_t> gather_offsets = gather_offsets_buf.Get<uint32_t>();
                    LocalTensor<float> weight_float = weight_float_buf.Get<float>();
                    LocalTensor<float> acc_float = acc_float_buf.Get<float>();
                    LocalTensor<float> tmp_float = tmp_float_buf.Get<float>();
                    const ptrdiff_t state_base = read_slot * state_s0
                                               + static_cast<ptrdiff_t>(fast_channel_begin) * state_s1;
                    if (state_s1 != 1) {
                        DataCopy(state_raw, conv_state[state_base],
                                 3 * fast_copy_len);
                        SetFlag<HardEvent::MTE2_S>(mte2_to_s);
                        WaitFlag<HardEvent::MTE2_S>(mte2_to_s);
                    }
                    if (has_bias) {
                        DataCopy(bias_data,
                                 bias[static_cast<ptrdiff_t>(fast_channel_begin) * bias_s0],
                                 fast_copy_len);
                        SetFlag<HardEvent::MTE2_V>(mte2_to_v);
                        WaitFlag<HardEvent::MTE2_V>(mte2_to_v);
                    }
                    for (int64_t t = 0; t < request_len; ++t) {
                        Duplicate(acc_float, 0.0f, fast_copy_len);
                        for (int k = 0; k < 4; ++k) {
                            const int64_t history_pos = t + k;
                            if (history_pos < 3) {
                                if constexpr (std::is_same<T, float>::value) {
                                    Gather(x_float, state_raw, gather_offsets,
                                           history_pos * sizeof(T), fast_tile_len);
                                } else {
                                    Gather(x_data, state_raw, gather_offsets,
                                           history_pos * sizeof(T), fast_tile_len);
                                }
                                PipeBarrier<PIPE_V>();
                            } else {
                                const int64_t token_idx = token_begin + history_pos - 3;
                                const ptrdiff_t qkv_offset = static_cast<ptrdiff_t>(token_batch) * qkv_s0
                                                           + static_cast<ptrdiff_t>(token_idx) * qkv_s1
                                                           + static_cast<ptrdiff_t>(fast_channel_begin);
                                if constexpr (std::is_same<T, float>::value) {
                                    DataCopy(x_float, qkv[qkv_offset], fast_copy_len);
                                } else {
                                    DataCopy(x_data, qkv[qkv_offset], fast_copy_len);
                                }
                                SetFlag<HardEvent::MTE2_V>(mte2_to_v);
                                WaitFlag<HardEvent::MTE2_V>(mte2_to_v);
                            }
                            if constexpr (!std::is_same<T, float>::value) {
                                Cast(x_float, x_data, RoundMode::CAST_NONE, fast_copy_len);
                            }
                            PipeBarrier<PIPE_V>();
                            Mul(tmp_float, x_float,
                                weight_float[k * fast_copy_len], fast_copy_len);
                            Add(acc_float, acc_float, tmp_float, fast_copy_len);
                            SetFlag<HardEvent::V_S>(v_to_s);
                            WaitFlag<HardEvent::V_S>(v_to_s);
                            SetFlag<HardEvent::V_MTE2>(v_to_mte2);
                            WaitFlag<HardEvent::V_MTE2>(v_to_mte2);
                        }
                        if (has_bias) {
                            if constexpr (std::is_same<T, float>::value) {
                                Add(acc_float, acc_float, bias_data, fast_copy_len);
                            } else {
                                Cast(tmp_float, bias_data, RoundMode::CAST_NONE, fast_copy_len);
                                Add(acc_float, acc_float, tmp_float, fast_copy_len);
                            }
                        }
                        const ptrdiff_t out_offset = static_cast<ptrdiff_t>(token_batch) * out_s0
                                                   + (token_begin + t) * out_s1
                                                   + static_cast<ptrdiff_t>(fast_channel_begin);
                        if constexpr (std::is_same<T, float>::value) {
                            SetFlag<HardEvent::V_MTE3>(v_to_mte3);
                            WaitFlag<HardEvent::V_MTE3>(v_to_mte3);
                            DataCopy(out[out_offset], acc_float, fast_tile_len);
                        } else {
                            Cast(out_data, acc_float, RoundMode::CAST_RINT, fast_copy_len);
                            SetFlag<HardEvent::V_MTE3>(v_to_mte3);
                            WaitFlag<HardEvent::V_MTE3>(v_to_mte3);
                            DataCopy(out[out_offset], out_data, fast_tile_len);
                        }
                        SetFlag<HardEvent::MTE3_V>(mte3_to_v);
                        WaitFlag<HardEvent::MTE3_V>(mte3_to_v);
                    }
                } else {
                    for (size_t channel = block; channel < C; channel += block_count) {
                        const ptrdiff_t state_base = read_slot * state_s0
                                                   + static_cast<ptrdiff_t>(channel) * state_s1;
                        const ptrdiff_t weight_base = static_cast<ptrdiff_t>(channel) * weight_s0;
                        for (int64_t t = 0; t < request_len; ++t) {
                            float acc = 0.0f;
                            for (int k = 0; k < 4; ++k) {
                                const float w = causalDataToFloat(
                                    weight.GetValue(weight_base + k * weight_s2));
                                const float x = causalLoadHistory(
                                    conv_state, qkv, t + k, token_begin, token_batch, channel,
                                    state_base, state_s2, qkv_s0, qkv_s1, qkv_s2);
                                acc += w * x;
                            }
                            if (has_bias) {
                                acc += causalDataToFloat(
                                    bias.GetValue(static_cast<ptrdiff_t>(channel) * bias_s0));
                            }
                            const ptrdiff_t out_offset = static_cast<ptrdiff_t>(token_batch) * out_s0
                                                       + (token_begin + t) * out_s1
                                                       + static_cast<ptrdiff_t>(channel) * out_s2;
                            out.SetValue(out_offset, causalFloatToData<T>(acc));
                        }
                    }
                }
            }
            if (update_state_only || fuse_state_update) {
                // Keep block boundaries on 512-byte cache-line boundaries. This
                // prevents different AI cores from flushing overlapping lines of
                // the interleaved [C, 3] recurrent-state layout.
                constexpr size_t state_align_channels = std::is_same<T, float>::value ? 128 : 256;
                const size_t raw_channels = (C + block_count - 1) / block_count;
                const size_t block_channels = ((raw_channels + state_align_channels - 1) / state_align_channels)
                                            * state_align_channels;
                const size_t channel_limit = (block + 1) * block_channels;
                const size_t channel_end = channel_limit < C ? channel_limit : C;
                const bool write_to_pool = final_indices != nullptr;
                const bool vector_state_update = vector_fast_path
                                              && (write_to_pool
                                                  || (final_s1 == 3 && final_s2 == 1));
                if (vector_state_update) {
                    LocalTensor<T> state_raw = state_raw_buf.Get<T>();
                    LocalTensor<T> qkv_raw = qkv_raw_buf.Get<T>();
                    for (int k = 0; k < 3; ++k) {
                        const int64_t history_pos = request_len + k;
                        if (history_pos < 3) {
                            for (size_t i = 0; i < fast_tile_len; ++i) {
                                state_raw.SetValue(
                                    i * 3 + k,
                                    state_raw.GetValue(i * 3 + history_pos));
                            }
                        } else {
                            const int64_t token_idx = token_begin + history_pos - 3;
                            const ptrdiff_t qkv_offset = static_cast<ptrdiff_t>(token_batch) * qkv_s0
                                                       + static_cast<ptrdiff_t>(token_idx) * qkv_s1
                                                       + static_cast<ptrdiff_t>(fast_channel_begin);
                            DataCopy(qkv_raw, qkv[qkv_offset], fast_copy_len);
                            SetFlag<HardEvent::MTE2_S>(mte2_to_s);
                            WaitFlag<HardEvent::MTE2_S>(mte2_to_s);
                            for (size_t i = 0; i < fast_tile_len; ++i) {
                                state_raw.SetValue(
                                    i * 3 + k, qkv_raw.GetValue(i));
                            }
                        }
                    }
                    const ptrdiff_t target_base = write_to_pool
                                                    ? write_slot * state_s0
                                                          + static_cast<ptrdiff_t>(fast_channel_begin) * state_s1
                                                    : static_cast<ptrdiff_t>(request) * final_s0
                                                          + static_cast<ptrdiff_t>(fast_channel_begin) * final_s1;
                    SetFlag<HardEvent::S_MTE3>(s_to_mte3);
                    WaitFlag<HardEvent::S_MTE3>(s_to_mte3);
                    if (write_to_pool) {
                        DataCopy(conv_state[target_base], state_raw,
                                 3 * fast_copy_len);
                    } else {
                        DataCopy(final_state[target_base], state_raw,
                                 3 * fast_copy_len);
                    }
                } else {
                    for (size_t channel = block * block_channels; channel < channel_end; ++channel) {
                        const ptrdiff_t state_base = read_slot * state_s0
                                                   + static_cast<ptrdiff_t>(channel) * state_s1;
                        const bool write_to_pool = final_indices != nullptr;
                        const ptrdiff_t target_base = write_to_pool
                                                        ? write_slot * state_s0
                                                              + static_cast<ptrdiff_t>(channel) * state_s1
                                                        : static_cast<ptrdiff_t>(request) * final_s0
                                                              + static_cast<ptrdiff_t>(channel) * final_s1;
                        const ptrdiff_t target_s2 = write_to_pool ? state_s2 : final_s2;
                        for (int k = 0; k < 3; ++k) {
                            const T value = causalFloatToData<T>(causalLoadHistory(
                                conv_state, qkv, request_len + k, token_begin, token_batch,
                                channel, state_base, state_s2, qkv_s0, qkv_s1, qkv_s2));
                            if (write_to_pool) {
                                conv_state.SetValue(target_base + k * target_s2, value);
                            } else {
                                final_state.SetValue(target_base + k * target_s2, value);
                            }
                        }
                    }
                }
            }
        }
        if (!update_state_only) {
            DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE>(out);
        }
        DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE>(conv_state);
        if (final_state_ptr != nullptr) {
            DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE>(final_state);
        }
    }
};
#define DEFINE_CAUSAL_CONV1D_KERNEL(NAME, TYPE)                                          \
    __global__ __aicore__ void NAME(                                                     \
        GM_ADDR out, GM_ADDR conv_state, GM_ADDR final_state, GM_ADDR qkv,               \
        GM_ADDR weight, GM_ADDR bias, GM_ADDR cu, GM_ADDR initial_indices,               \
        GM_ADDR final_indices, bool has_bias, bool has_cu, bool cu_i64,                  \
        bool initial_i64, bool final_i64, bool indexed_pool,                             \
        bool update_state_only, bool fuse_state_update,                                  \
        size_t request_count, size_t T_tokens, size_t C,                                 \
        size_t total_tokens, size_t pool_size, ptrdiff_t out_s0,                         \
        ptrdiff_t out_s1, ptrdiff_t out_s2, ptrdiff_t state_s0,                          \
        ptrdiff_t state_s1, ptrdiff_t state_s2, ptrdiff_t final_s0,                      \
        ptrdiff_t final_s1, ptrdiff_t final_s2, ptrdiff_t qkv_s0,                        \
        ptrdiff_t qkv_s1, ptrdiff_t qkv_s2, ptrdiff_t weight_s0,                         \
        ptrdiff_t weight_s2, ptrdiff_t bias_s0) {                                        \
        CausalConv1dKernel<TYPE> kernel;                                                 \
        kernel.process(out, conv_state, final_state, qkv, weight, bias, cu,              \
                       initial_indices, final_indices, has_bias, has_cu,                 \
                       cu_i64, initial_i64, final_i64, indexed_pool,                     \
                       update_state_only, fuse_state_update, request_count, T_tokens, C, \
                       total_tokens, pool_size,                                          \
                       out_s0, out_s1, out_s2, state_s0, state_s1, state_s2,             \
                       final_s0, final_s1, final_s2, qkv_s0, qkv_s1, qkv_s2,             \
                       weight_s0, weight_s2, bias_s0);                                   \
    }

DEFINE_CAUSAL_CONV1D_KERNEL(causal_conv1d_half, half)
DEFINE_CAUSAL_CONV1D_KERNEL(causal_conv1d_float, float)
DEFINE_CAUSAL_CONV1D_KERNEL(causal_conv1d_bf16, bfloat16_t)
#undef DEFINE_CAUSAL_CONV1D_KERNEL

extern "C" infiniStatus_t causal_conv1d_kernel_launch(
    void *out, void *conv_state, void *final_state, const void *qkv,
    const void *weight, const void *bias, const void *cu,
    const void *initial_indices, const void *final_indices,
    infiniDtype_t dtype, bool has_bias, bool has_cu, bool cu_i64,
    bool initial_i64, bool final_i64, bool indexed_pool,
    size_t request_count, size_t T_tokens, size_t C, size_t total_tokens,
    size_t pool_size, ptrdiff_t out_s0, ptrdiff_t out_s1, ptrdiff_t out_s2,
    ptrdiff_t state_s0, ptrdiff_t state_s1, ptrdiff_t state_s2,
    ptrdiff_t final_s0, ptrdiff_t final_s1, ptrdiff_t final_s2,
    ptrdiff_t qkv_s0, ptrdiff_t qkv_s1, ptrdiff_t qkv_s2,
    ptrdiff_t weight_s0, ptrdiff_t weight_s2, ptrdiff_t bias_s0,
    void *stream) {
    if (request_count == 0 || C == 0) {
        return INFINI_STATUS_SUCCESS;
    }
    const size_t elem_size = dtype == INFINI_DTYPE_F32 ? sizeof(float) : sizeof(uint16_t);
    uint32_t blocks = static_cast<uint32_t>(BLOCK_NUM);
    while (blocks > 1
           && (C % blocks != 0
               || ((C / blocks) * elem_size) % BYTE_ALIGN != 0)) {
        --blocks;
    }
    const bool fuse_state_update = out_s2 == 1 && qkv_s2 == 1
                                && state_s1 == 3 && state_s2 == 1
                                && weight_s0 == 4 && weight_s2 == 1
                                && (!has_bias || bias_s0 == 1);
#define LAUNCH_CASE(DTYPE, NAME)                                             \
    case DTYPE:                                                              \
        NAME<<<blocks, nullptr, stream>>>(                                   \
            out, conv_state, final_state, const_cast<void *>(qkv),           \
            const_cast<void *>(weight), const_cast<void *>(bias),            \
            const_cast<void *>(cu), const_cast<void *>(initial_indices),     \
            const_cast<void *>(final_indices), has_bias, has_cu, cu_i64,     \
            initial_i64, final_i64, indexed_pool, false, fuse_state_update,  \
            request_count, T_tokens, C, total_tokens, pool_size,             \
            out_s0, out_s1, out_s2,                                          \
            state_s0, state_s1, state_s2, final_s0, final_s1, final_s2,      \
            qkv_s0, qkv_s1, qkv_s2, weight_s0, weight_s2, bias_s0);          \
        if (!fuse_state_update) {                                            \
            NAME<<<blocks, nullptr, stream>>>(                               \
                out, conv_state, final_state, const_cast<void *>(qkv),       \
                const_cast<void *>(weight), const_cast<void *>(bias),        \
                const_cast<void *>(cu), const_cast<void *>(initial_indices), \
                const_cast<void *>(final_indices), has_bias, has_cu, cu_i64, \
                initial_i64, final_i64, indexed_pool, true, false,           \
                request_count, T_tokens, C, total_tokens, pool_size,         \
                out_s0, out_s1, out_s2, state_s0, state_s1, state_s2,        \
                final_s0, final_s1, final_s2, qkv_s0, qkv_s1, qkv_s2,        \
                weight_s0, weight_s2, bias_s0);                              \
        }                                                                    \
        return INFINI_STATUS_SUCCESS;
    switch (dtype) {
        LAUNCH_CASE(INFINI_DTYPE_F16, causal_conv1d_half)
        LAUNCH_CASE(INFINI_DTYPE_F32, causal_conv1d_float)
        LAUNCH_CASE(INFINI_DTYPE_BF16, causal_conv1d_bf16)
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
#undef LAUNCH_CASE
}
