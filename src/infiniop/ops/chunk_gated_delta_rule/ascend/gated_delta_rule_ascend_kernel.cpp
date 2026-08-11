#include "gated_delta_rule_ascend_kernel.h"
#include "../../../devices/ascend/ascend_kernel_common.h"

#include <type_traits>

using namespace AscendC;

template <typename T>
__aicore__ inline float gdrToFloat(T value) {
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        uint32_t bits = static_cast<uint32_t>(*reinterpret_cast<uint16_t *>(&value)) << 16;
        return *reinterpret_cast<float *>(&bits);
    } else {
        return static_cast<float>(value);
    }
}

template <typename T>
__aicore__ inline T gdrFromFloat(float value) {
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        uint32_t bits = *reinterpret_cast<uint32_t *>(&value);
        uint16_t upper = static_cast<uint16_t>(bits >> 16);
        return *reinterpret_cast<bfloat16_t *>(&upper);
    } else {
        return static_cast<T>(value);
    }
}

__aicore__ inline float gdrExp(float x) {
    if (x > 88.0f) {
        x = 88.0f;
    } else if (x < -87.0f) {
        return 0.0f;
    }
    float scaled = x * 1.4426950408889634f;
    int32_t exponent = static_cast<int32_t>(scaled);
    if (scaled < static_cast<float>(exponent)) {
        --exponent;
    }
    float r = x - static_cast<float>(exponent) * 0.693145751953125f
            - static_cast<float>(exponent) * 1.428606765330187e-6f;
    float polynomial = 1.0f
                     + r * (1.0f + r * (0.5f + r * (0.1666666716f + r * (0.0416666679f + r * (0.0083333338f + r * 0.0013888889f)))));
    uint32_t bits = static_cast<uint32_t>(exponent + 127) << 23;
    return polynomial * *reinterpret_cast<float *>(&bits);
}

__aicore__ inline float gdrLocalSum(
    LocalTensor<float> tensor,
    LocalTensor<float> reduce_work,
    LocalTensor<float> reduce_result,
    size_t count,
    event_t v_to_s) {
    ReduceSum(reduce_result, tensor, reduce_work, static_cast<int32_t>(count));
    SetFlag<HardEvent::V_S>(v_to_s);
    WaitFlag<HardEvent::V_S>(v_to_s);
    return reduce_result.GetValue(0);
}

__aicore__ inline float gdrRsqrt(float x) {
    if (x <= 0.0f) {
        return 0.0f;
    }
    float scaled = x;
    float rescale = 1.0f;
    while (scaled > 4.0f) {
        scaled *= 0.25f;
        rescale *= 0.5f;
    }
    while (scaled < 1.0f) {
        scaled *= 4.0f;
        rescale *= 2.0f;
    }
    float y = 0.75f;
    for (int i = 0; i < 6; ++i) {
        y *= 1.5f - 0.5f * scaled * y * y;
    }
    return y * rescale;
}

__aicore__ inline int64_t gdrLoadIndex(
    GM_ADDR ptr, bool is_i64, size_t index, int64_t fallback) {
    if (ptr == nullptr) {
        return fallback;
    }
    if (is_i64) {
        return reinterpret_cast<__gm__ int64_t *>(ptr)[index];
    }
    return static_cast<int64_t>(
        reinterpret_cast<__gm__ int32_t *>(ptr)[index]);
}

template <typename TData, typename TState>
__aicore__ inline void gatedDeltaRuleProcess(
    GM_ADDR workspace_ptr,
    GM_ADDR out_ptr,
    GM_ADDR initial_state_ptr,
    GM_ADDR final_state_ptr,
    GM_ADDR q_ptr,
    GM_ADDR k_ptr,
    GM_ADDR v_ptr,
    GM_ADDR g_ptr,
    GM_ADDR beta_ptr,
    GM_ADDR cu_seqlens_ptr,
    GM_ADDR initial_indices_ptr,
    GM_ADDR final_indices_ptr,
    bool use_qk_l2norm,
    bool has_cu_seqlens,
    bool cu_seqlens_i64,
    bool has_initial_indices,
    bool initial_indices_i64,
    bool has_final_indices,
    bool final_indices_i64,
    size_t B,
    size_t T_tokens,
    size_t total_tokens,
    size_t Hk,
    size_t Hv,
    size_t Dk,
    size_t Dv,
    size_t pool_size,
    size_t value_heads_per_key_head,
    float q_scale,
    ptrdiff_t out_s0,
    ptrdiff_t out_s1,
    ptrdiff_t out_s2,
    ptrdiff_t out_s3,
    ptrdiff_t q_s0,
    ptrdiff_t q_s1,
    ptrdiff_t q_s2,
    ptrdiff_t q_s3,
    ptrdiff_t k_s0,
    ptrdiff_t k_s1,
    ptrdiff_t k_s2,
    ptrdiff_t k_s3,
    ptrdiff_t v_s0,
    ptrdiff_t v_s1,
    ptrdiff_t v_s2,
    ptrdiff_t v_s3) {

    const bool ub_fast_path = Dk == 128 && Dv == 128
                           && q_s3 == 1 && k_s3 == 1 && v_s3 == 1 && out_s3 == 1;
    constexpr size_t FAST_VALUE_TILE_LEN = 32;
    const size_t value_tiles = ub_fast_path ? Dv / FAST_VALUE_TILE_LEN : 1;
    size_t block = GetBlockIdx();
    size_t request = block / (Hv * value_tiles);
    size_t head_tile = block % (Hv * value_tiles);
    size_t vh = head_tile / value_tiles;
    size_t value_tile = head_tile % value_tiles;
    if (request >= B) {
        return;
    }
    size_t kh = vh / value_heads_per_key_head;

    GlobalTensor<float> work;
    GlobalTensor<float> gate;
    GlobalTensor<float> beta;
    GlobalTensor<TData> out;
    GlobalTensor<TState> initial_state;
    GlobalTensor<TState> final_state;
    GlobalTensor<TData> q;
    GlobalTensor<TData> k;
    GlobalTensor<TData> v;
    work.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(workspace_ptr));
    gate.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(g_ptr));
    beta.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(beta_ptr));
    out.SetGlobalBuffer(reinterpret_cast<__gm__ TData *>(out_ptr));
    initial_state.SetGlobalBuffer(
        reinterpret_cast<__gm__ TState *>(initial_state_ptr));
    final_state.SetGlobalBuffer(reinterpret_cast<__gm__ TState *>(final_state_ptr));
    q.SetGlobalBuffer(reinterpret_cast<__gm__ TData *>(q_ptr));
    k.SetGlobalBuffer(reinterpret_cast<__gm__ TData *>(k_ptr));
    v.SetGlobalBuffer(reinterpret_cast<__gm__ TData *>(v_ptr));

    int64_t begin = 0;
    int64_t end = static_cast<int64_t>(T_tokens);
    if (has_cu_seqlens) {
        begin = gdrLoadIndex(
            cu_seqlens_ptr, cu_seqlens_i64, request, 0);
        end = gdrLoadIndex(
            cu_seqlens_ptr, cu_seqlens_i64, request + 1, 0);
        if (begin < 0 || end < begin
            || end > static_cast<int64_t>(total_tokens)) {
            return;
        }
    }

    int64_t read_slot = has_initial_indices
                          ? gdrLoadIndex(
                              initial_indices_ptr, initial_indices_i64,
                              request, static_cast<int64_t>(request))
                          : static_cast<int64_t>(request);
    int64_t write_slot = has_final_indices
                           ? gdrLoadIndex(
                               final_indices_ptr, final_indices_i64,
                               request, static_cast<int64_t>(request))
                           : static_cast<int64_t>(request);
    if (read_slot < 0 || read_slot >= static_cast<int64_t>(pool_size)
        || (has_final_indices
            && (write_slot < 0
                || write_slot >= static_cast<int64_t>(pool_size)))) {
        return;
    }

    size_t matrix_size = Dv * Dk;
    size_t initial_base = (static_cast<size_t>(read_slot) * Hv + vh) * matrix_size;

    if (ub_fast_path) {
        constexpr size_t VECTOR_LEN = 128;
        constexpr size_t VALUE_TILE_LEN = FAST_VALUE_TILE_LEN;
        constexpr size_t MATRIX_LEN = VALUE_TILE_LEN * VECTOR_LEN;
        constexpr size_t STATE_DATA_BYTES = std::is_same<TState, float>::value
                                              ? 32
                                              : MATRIX_LEN * sizeof(TState);
        constexpr size_t IO_DATA_BYTES = std::is_same<TData, float>::value
                                           ? 32
                                           : VECTOR_LEN * sizeof(TData);
        TPipe pipe;
        const event_t v_to_s = static_cast<event_t>(
            pipe.FetchEventID(HardEvent::V_S));
        const event_t s_to_v = static_cast<event_t>(
            pipe.FetchEventID(HardEvent::S_V));
        const event_t mte2_to_v = static_cast<event_t>(
            pipe.FetchEventID(HardEvent::MTE2_V));
        const event_t v_to_mte2 = static_cast<event_t>(
            pipe.FetchEventID(HardEvent::V_MTE2));
        const event_t v_to_mte3 = static_cast<event_t>(
            pipe.FetchEventID(HardEvent::V_MTE3));
        const event_t s_to_mte3 = static_cast<event_t>(
            pipe.FetchEventID(HardEvent::S_MTE3));
        TBuf<QuePosition::VECCALC> state_data_buf, io_data_buf, out_data_buf;
        TBuf<QuePosition::VECCALC> state_float_buf;
        TBuf<QuePosition::VECCALC> q_float_buf, k_float_buf;
        TBuf<QuePosition::VECCALC> v_float_buf, out_float_buf;
        TBuf<QuePosition::VECCALC> tmp_float_buf;
        TBuf<QuePosition::VECCALC> reduce_work_buf, reduce_result_buf;
        pipe.InitBuffer(state_data_buf, STATE_DATA_BYTES);
        pipe.InitBuffer(io_data_buf, IO_DATA_BYTES);
        pipe.InitBuffer(out_data_buf, IO_DATA_BYTES);
        pipe.InitBuffer(state_float_buf, MATRIX_LEN * sizeof(float));
        pipe.InitBuffer(q_float_buf, VECTOR_LEN * sizeof(float));
        pipe.InitBuffer(k_float_buf, VECTOR_LEN * sizeof(float));
        pipe.InitBuffer(v_float_buf, VALUE_TILE_LEN * sizeof(float));
        pipe.InitBuffer(out_float_buf, VALUE_TILE_LEN * sizeof(float));
        pipe.InitBuffer(tmp_float_buf, VECTOR_LEN * sizeof(float));
        pipe.InitBuffer(reduce_work_buf, VECTOR_LEN * sizeof(float));
        pipe.InitBuffer(reduce_result_buf, 8 * sizeof(float));

        LocalTensor<TState> state_data = state_data_buf.Get<TState>();
        LocalTensor<TData> io_data = io_data_buf.Get<TData>();
        LocalTensor<TData> out_data = out_data_buf.Get<TData>();
        LocalTensor<float> state_float = state_float_buf.Get<float>();
        LocalTensor<float> q_float = q_float_buf.Get<float>();
        LocalTensor<float> k_float = k_float_buf.Get<float>();
        LocalTensor<float> v_float = v_float_buf.Get<float>();
        LocalTensor<float> out_float = out_float_buf.Get<float>();
        LocalTensor<float> tmp_float = tmp_float_buf.Get<float>();
        LocalTensor<float> reduce_work = reduce_work_buf.Get<float>();
        LocalTensor<float> reduce_result = reduce_result_buf.Get<float>();

        size_t tiled_initial_base = initial_base
                                  + value_tile * VALUE_TILE_LEN * VECTOR_LEN;
        if constexpr (std::is_same<TState, float>::value) {
            constexpr size_t COPY_LEN = 4096;
            for (size_t offset = 0; offset < MATRIX_LEN; offset += COPY_LEN) {
                DataCopy(state_float[offset], initial_state[tiled_initial_base + offset],
                         COPY_LEN);
            }
            SetFlag<HardEvent::MTE2_V>(mte2_to_v);
            WaitFlag<HardEvent::MTE2_V>(mte2_to_v);
        } else {
            DataCopy(state_data, initial_state[tiled_initial_base], MATRIX_LEN);
            SetFlag<HardEvent::MTE2_V>(mte2_to_v);
            WaitFlag<HardEvent::MTE2_V>(mte2_to_v);
            Cast(state_float, state_data, RoundMode::CAST_NONE, MATRIX_LEN);
        }
        PipeBarrier<PIPE_V>();

        for (int64_t token = begin; token < end; ++token) {
            size_t token_batch = has_cu_seqlens ? 0 : request;
            size_t token_index = static_cast<size_t>(token);
            ptrdiff_t q_base = static_cast<ptrdiff_t>(token_batch) * q_s0
                             + static_cast<ptrdiff_t>(token_index) * q_s1
                             + static_cast<ptrdiff_t>(kh) * q_s2;
            ptrdiff_t k_base = static_cast<ptrdiff_t>(token_batch) * k_s0
                             + static_cast<ptrdiff_t>(token_index) * k_s1
                             + static_cast<ptrdiff_t>(kh) * k_s2;
            ptrdiff_t v_base = static_cast<ptrdiff_t>(token_batch) * v_s0
                             + static_cast<ptrdiff_t>(token_index) * v_s1
                             + static_cast<ptrdiff_t>(vh) * v_s2
                             + static_cast<ptrdiff_t>(value_tile * VALUE_TILE_LEN) * v_s3;
            ptrdiff_t out_base = static_cast<ptrdiff_t>(token_batch) * out_s0
                               + static_cast<ptrdiff_t>(token_index) * out_s1
                               + static_cast<ptrdiff_t>(vh) * out_s2
                               + static_cast<ptrdiff_t>(value_tile * VALUE_TILE_LEN) * out_s3;

            if constexpr (std::is_same<TData, float>::value) {
                DataCopy(q_float, q[q_base], VECTOR_LEN);
                DataCopy(k_float, k[k_base], VECTOR_LEN);
                DataCopy(v_float, v[v_base], VALUE_TILE_LEN);
                SetFlag<HardEvent::MTE2_V>(mte2_to_v);
                WaitFlag<HardEvent::MTE2_V>(mte2_to_v);
            } else {
                DataCopy(io_data, q[q_base], VECTOR_LEN);
                SetFlag<HardEvent::MTE2_V>(mte2_to_v);
                WaitFlag<HardEvent::MTE2_V>(mte2_to_v);
                Cast(q_float, io_data, RoundMode::CAST_NONE, VECTOR_LEN);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(v_to_mte2);
                WaitFlag<HardEvent::V_MTE2>(v_to_mte2);
                DataCopy(io_data, k[k_base], VECTOR_LEN);
                SetFlag<HardEvent::MTE2_V>(mte2_to_v);
                WaitFlag<HardEvent::MTE2_V>(mte2_to_v);
                Cast(k_float, io_data, RoundMode::CAST_NONE, VECTOR_LEN);
                PipeBarrier<PIPE_V>();
                SetFlag<HardEvent::V_MTE2>(v_to_mte2);
                WaitFlag<HardEvent::V_MTE2>(v_to_mte2);
                DataCopy(io_data, v[v_base], VALUE_TILE_LEN);
                SetFlag<HardEvent::MTE2_V>(mte2_to_v);
                WaitFlag<HardEvent::MTE2_V>(mte2_to_v);
                Cast(v_float, io_data, RoundMode::CAST_NONE, VALUE_TILE_LEN);
            }
            SetFlag<HardEvent::V_S>(v_to_s);
            WaitFlag<HardEvent::V_S>(v_to_s);

            if (use_qk_l2norm) {
                Mul(tmp_float, q_float, q_float, VECTOR_LEN);
                float q_norm_inv = gdrRsqrt(
                    gdrLocalSum(tmp_float, reduce_work, reduce_result,
                                VECTOR_LEN, v_to_s)
                    + 1e-6f);
                Muls(q_float, q_float, q_norm_inv * q_scale, VECTOR_LEN);

                Mul(tmp_float, k_float, k_float, VECTOR_LEN);
                float k_norm_inv = gdrRsqrt(
                    gdrLocalSum(tmp_float, reduce_work, reduce_result,
                                VECTOR_LEN, v_to_s)
                    + 1e-6f);
                Muls(k_float, k_float, k_norm_inv, VECTOR_LEN);
            } else if (q_scale != 1.0f) {
                Muls(q_float, q_float, q_scale, VECTOR_LEN);
            }

            size_t gate_offset = (token_batch * T_tokens + token_index) * Hv + vh;
            float gate_value = gdrExp(gate.GetValue(gate_offset));
            float beta_value = beta.GetValue(gate_offset);
            Muls(state_float, state_float, gate_value, MATRIX_LEN);

            for (size_t dv = 0; dv < VALUE_TILE_LEN; ++dv) {
                LocalTensor<float> state_row = state_float[dv * VECTOR_LEN];
                Mul(tmp_float, state_row, k_float, VECTOR_LEN);
                float kv_memory = gdrLocalSum(
                    tmp_float, reduce_work, reduce_result, VECTOR_LEN, v_to_s);
                float delta = (v_float.GetValue(dv) - kv_memory) * beta_value;
                SetFlag<HardEvent::S_V>(s_to_v);
                WaitFlag<HardEvent::S_V>(s_to_v);
                Muls(tmp_float, k_float, delta, VECTOR_LEN);
                PipeBarrier<PIPE_V>();
                Add(state_row, state_row, tmp_float, VECTOR_LEN);
                PipeBarrier<PIPE_V>();
                Mul(tmp_float, state_row, q_float, VECTOR_LEN);
                out_float.SetValue(
                    dv, gdrLocalSum(tmp_float, reduce_work, reduce_result,
                                    VECTOR_LEN, v_to_s));
            }

            if constexpr (std::is_same<TData, float>::value) {
                SetFlag<HardEvent::S_MTE3>(s_to_mte3);
                WaitFlag<HardEvent::S_MTE3>(s_to_mte3);
                DataCopy(out[out_base], out_float, VALUE_TILE_LEN);
            } else {
                SetFlag<HardEvent::S_V>(s_to_v);
                WaitFlag<HardEvent::S_V>(s_to_v);
                Cast(out_data, out_float, RoundMode::CAST_RINT, VALUE_TILE_LEN);
                SetFlag<HardEvent::V_MTE3>(v_to_mte3);
                WaitFlag<HardEvent::V_MTE3>(v_to_mte3);
                DataCopy(out[out_base], out_data, VALUE_TILE_LEN);
            }
        }

        size_t destination_base = has_final_indices
                                    ? (static_cast<size_t>(write_slot) * Hv + vh) * matrix_size
                                    : (request * Hv + vh) * matrix_size;
        destination_base += value_tile * VALUE_TILE_LEN * VECTOR_LEN;
        if constexpr (std::is_same<TState, float>::value) {
            SetFlag<HardEvent::V_MTE3>(v_to_mte3);
            WaitFlag<HardEvent::V_MTE3>(v_to_mte3);
            if (has_final_indices) {
                constexpr size_t COPY_LEN = 4096;
                for (size_t offset = 0; offset < MATRIX_LEN; offset += COPY_LEN) {
                    DataCopy(initial_state[destination_base + offset],
                             state_float[offset], COPY_LEN);
                }
            } else {
                constexpr size_t COPY_LEN = 4096;
                for (size_t offset = 0; offset < MATRIX_LEN; offset += COPY_LEN) {
                    DataCopy(final_state[destination_base + offset],
                             state_float[offset], COPY_LEN);
                }
            }
        } else {
            Cast(state_data, state_float, RoundMode::CAST_RINT, MATRIX_LEN);
            SetFlag<HardEvent::V_MTE3>(v_to_mte3);
            WaitFlag<HardEvent::V_MTE3>(v_to_mte3);
            if (has_final_indices) {
                DataCopy(initial_state[destination_base], state_data,
                         MATRIX_LEN);
            } else {
                DataCopy(final_state[destination_base], state_data,
                         MATRIX_LEN);
            }
        }
        return;
    }

    size_t work_base = (request * Hv + vh) * matrix_size;
    for (size_t dv = 0; dv < Dv; ++dv) {
        for (size_t dk = 0; dk < Dk; ++dk) {
            size_t element = dv * Dk + dk;
            work.SetValue(
                work_base + element,
                gdrToFloat(initial_state.GetValue(initial_base + element)));
        }
    }

    for (int64_t token = begin; token < end; ++token) {
        size_t token_batch = has_cu_seqlens ? 0 : request;
        size_t token_index = static_cast<size_t>(token);
        ptrdiff_t q_base = static_cast<ptrdiff_t>(token_batch) * q_s0
                         + static_cast<ptrdiff_t>(token_index) * q_s1
                         + static_cast<ptrdiff_t>(kh) * q_s2;
        ptrdiff_t k_base = static_cast<ptrdiff_t>(token_batch) * k_s0
                         + static_cast<ptrdiff_t>(token_index) * k_s1
                         + static_cast<ptrdiff_t>(kh) * k_s2;
        float q_norm_inv = 1.0f;
        float k_norm_inv = 1.0f;
        if (use_qk_l2norm) {
            float q_sum = 0.0f;
            float k_sum = 0.0f;
            for (size_t dk = 0; dk < Dk; ++dk) {
                float q_value = gdrToFloat(q.GetValue(q_base + dk * q_s3));
                float k_value = gdrToFloat(k.GetValue(k_base + dk * k_s3));
                q_sum += q_value * q_value;
                k_sum += k_value * k_value;
            }
            q_norm_inv = gdrRsqrt(q_sum + 1e-6f);
            k_norm_inv = gdrRsqrt(k_sum + 1e-6f);
        }

        size_t gate_offset = (token_batch * T_tokens + token_index) * Hv + vh;
        float gate_value = gdrExp(gate.GetValue(gate_offset));
        float beta_value = beta.GetValue(gate_offset);
        ptrdiff_t v_base = static_cast<ptrdiff_t>(token_batch) * v_s0
                         + static_cast<ptrdiff_t>(token_index) * v_s1
                         + static_cast<ptrdiff_t>(vh) * v_s2;
        ptrdiff_t out_base = static_cast<ptrdiff_t>(token_batch) * out_s0
                           + static_cast<ptrdiff_t>(token_index) * out_s1
                           + static_cast<ptrdiff_t>(vh) * out_s2;

        for (size_t dv = 0; dv < Dv; ++dv) {
            float kv_memory = 0.0f;
            size_t row_base = work_base + dv * Dk;
            for (size_t dk = 0; dk < Dk; ++dk) {
                float state = work.GetValue(row_base + dk) * gate_value;
                work.SetValue(row_base + dk, state);
                float key = gdrToFloat(k.GetValue(k_base + dk * k_s3)) * k_norm_inv;
                kv_memory += state * key;
            }
            float value = gdrToFloat(v.GetValue(v_base + dv * v_s3));
            float delta = (value - kv_memory) * beta_value;
            float output = 0.0f;
            for (size_t dk = 0; dk < Dk; ++dk) {
                float key = gdrToFloat(k.GetValue(k_base + dk * k_s3)) * k_norm_inv;
                float state = work.GetValue(row_base + dk) + key * delta;
                work.SetValue(row_base + dk, state);
                float query = gdrToFloat(q.GetValue(q_base + dk * q_s3))
                            * q_norm_inv * q_scale;
                output += state * query;
            }
            out.SetValue(
                out_base + dv * out_s3, gdrFromFloat<TData>(output));
        }
    }

    size_t destination_base;
    if (has_final_indices) {
        destination_base = (static_cast<size_t>(write_slot) * Hv + vh) * matrix_size;
    } else {
        destination_base = (request * Hv + vh) * matrix_size;
    }
    for (size_t element = 0; element < matrix_size; ++element) {
        TState value = gdrFromFloat<TState>(work.GetValue(work_base + element));
        if (has_final_indices) {
            initial_state.SetValue(destination_base + element, value);
        } else {
            final_state.SetValue(destination_base + element, value);
        }
    }
}

#define DEFINE_GDR_KERNEL(NAME, DATA_TYPE, STATE_TYPE)                    \
    __global__ __aicore__ void NAME(                                      \
        GM_ADDR workspace, GM_ADDR out, GM_ADDR initial_state,            \
        GM_ADDR final_state, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR g,  \
        GM_ADDR beta, GM_ADDR cu_seqlens, GM_ADDR initial_indices,        \
        GM_ADDR final_indices, bool use_qk_l2norm, bool has_cu_seqlens,   \
        bool cu_seqlens_i64, bool has_initial_indices,                    \
        bool initial_indices_i64, bool has_final_indices,                 \
        bool final_indices_i64, size_t B, size_t T_tokens,                \
        size_t total_tokens, size_t Hk, size_t Hv, size_t Dk, size_t Dv,  \
        size_t pool_size, size_t value_heads_per_key_head, float q_scale, \
        ptrdiff_t out_s0, ptrdiff_t out_s1, ptrdiff_t out_s2,             \
        ptrdiff_t out_s3, ptrdiff_t q_s0, ptrdiff_t q_s1, ptrdiff_t q_s2, \
        ptrdiff_t q_s3, ptrdiff_t k_s0, ptrdiff_t k_s1, ptrdiff_t k_s2,   \
        ptrdiff_t k_s3, ptrdiff_t v_s0, ptrdiff_t v_s1, ptrdiff_t v_s2,   \
        ptrdiff_t v_s3) {                                                 \
        gatedDeltaRuleProcess<DATA_TYPE, STATE_TYPE>(                     \
            workspace, out, initial_state, final_state, q, k, v, g, beta, \
            cu_seqlens, initial_indices, final_indices, use_qk_l2norm,    \
            has_cu_seqlens, cu_seqlens_i64, has_initial_indices,          \
            initial_indices_i64, has_final_indices, final_indices_i64, B, \
            T_tokens, total_tokens, Hk, Hv, Dk, Dv, pool_size,            \
            value_heads_per_key_head, q_scale, out_s0, out_s1, out_s2,    \
            out_s3, q_s0, q_s1, q_s2, q_s3, k_s0, k_s1, k_s2, k_s3,       \
            v_s0, v_s1, v_s2, v_s3);                                      \
    }

DEFINE_GDR_KERNEL(gated_delta_rule_half, half, half)
DEFINE_GDR_KERNEL(gated_delta_rule_half_float_state, half, float)
DEFINE_GDR_KERNEL(gated_delta_rule_bf16, bfloat16_t, bfloat16_t)
DEFINE_GDR_KERNEL(gated_delta_rule_bf16_float_state, bfloat16_t, float)
DEFINE_GDR_KERNEL(gated_delta_rule_float, float, float)
#undef DEFINE_GDR_KERNEL

extern "C" infiniStatus_t gated_delta_rule_ascend_kernel_launch(
    void *workspace,
    void *out,
    void *initial_state,
    void *final_state,
    const void *q,
    const void *k,
    const void *v,
    const void *g,
    const void *beta,
    const void *cu_seqlens,
    const void *initial_state_indices,
    const void *final_state_indices,
    const GatedDeltaRuleAscendParams *p,
    void *stream) {

    if (p->B == 0 || p->Hv == 0) {
        return INFINI_STATUS_SUCCESS;
    }
    const bool ub_fast_path = p->Dk == 128 && p->Dv == 128
                           && p->q_strides[3] == 1 && p->k_strides[3] == 1
                           && p->v_strides[3] == 1 && p->out_strides[3] == 1;
    const size_t value_tiles = ub_fast_path ? p->Dv / 32 : 1;
    uint32_t blocks = static_cast<uint32_t>(p->B * p->Hv * value_tiles);

#define LAUNCH_GDR(NAME)                                              \
    NAME<<<blocks, nullptr, stream>>>(                                \
        workspace, out, initial_state, final_state,                   \
        const_cast<void *>(q), const_cast<void *>(k),                 \
        const_cast<void *>(v), const_cast<void *>(g),                 \
        const_cast<void *>(beta), const_cast<void *>(cu_seqlens),     \
        const_cast<void *>(initial_state_indices),                    \
        const_cast<void *>(final_state_indices), p->use_qk_l2norm,    \
        p->has_cu_seqlens, p->cu_seqlens_i64, p->has_initial_indices, \
        p->initial_indices_i64, p->has_final_indices,                 \
        p->final_indices_i64, p->B, p->T, p->total_tokens, p->Hk,     \
        p->Hv, p->Dk, p->Dv, p->pool_size,                            \
        p->value_heads_per_key_head, p->q_scale, p->out_strides[0],   \
        p->out_strides[1], p->out_strides[2], p->out_strides[3],      \
        p->q_strides[0], p->q_strides[1], p->q_strides[2],            \
        p->q_strides[3], p->k_strides[0], p->k_strides[1],            \
        p->k_strides[2], p->k_strides[3], p->v_strides[0],            \
        p->v_strides[1], p->v_strides[2], p->v_strides[3]);           \
    return INFINI_STATUS_SUCCESS;

    switch (static_cast<infiniDtype_t>(p->data_dtype)) {
    case INFINI_DTYPE_F16:
        switch (static_cast<infiniDtype_t>(p->state_dtype)) {
        case INFINI_DTYPE_F16:
            LAUNCH_GDR(gated_delta_rule_half)
        case INFINI_DTYPE_F32:
            LAUNCH_GDR(gated_delta_rule_half_float_state)
        default:
            break;
        }
        break;
    case INFINI_DTYPE_BF16:
        switch (static_cast<infiniDtype_t>(p->state_dtype)) {
        case INFINI_DTYPE_BF16:
            LAUNCH_GDR(gated_delta_rule_bf16)
        case INFINI_DTYPE_F32:
            LAUNCH_GDR(gated_delta_rule_bf16_float_state)
        default:
            break;
        }
        break;
    case INFINI_DTYPE_F32:
        if (p->state_dtype == INFINI_DTYPE_F32) {
            LAUNCH_GDR(gated_delta_rule_float)
        }
        break;
    default:
        break;
    }
#undef LAUNCH_GDR
    return INFINI_STATUS_BAD_TENSOR_DTYPE;
}
