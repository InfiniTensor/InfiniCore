#include "recurrent_gated_delta_rule_native_kernel.h"
#include "../../../devices/ascend/ascend_kernel_common.h"

using namespace AscendC;

namespace {

constexpr size_t D = 128;
constexpr size_t MATRIX = D * D;
constexpr size_t STATE_TILE = 4096;

__aicore__ inline float nativeRsqrt(float x) {
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

__aicore__ inline int64_t loadIndex(
    GM_ADDR ptr, bool is_i64, size_t index) {
    if (is_i64) {
        return reinterpret_cast<__gm__ int64_t *>(ptr)[index];
    }
    return static_cast<int64_t>(
        reinterpret_cast<__gm__ int32_t *>(ptr)[index]);
}

__global__ __aicore__ void recurrent_gdr_native_preprocess(
    GM_ADDR q_normalized_ptr,
    GM_ADDR k_normalized_ptr,
    GM_ADDR v_contiguous_ptr,
    GM_ADDR beta_bf16_ptr,
    GM_ADDR state_staging_ptr,
    GM_ADDR actual_seq_lengths_ptr,
    GM_ADDR state_indices_ptr,
    GM_ADDR q_ptr,
    GM_ADDR k_ptr,
    GM_ADDR v_ptr,
    GM_ADDR beta_ptr,
    GM_ADDR state_ptr,
    GM_ADDR initial_state_indices_ptr,
    GM_ADDR final_state_indices_ptr,
    size_t B,
    size_t Hk,
    size_t Hv,
    size_t pool_size,
    bool initial_indices_i64,
    bool final_indices_i64,
    ptrdiff_t q_s0,
    ptrdiff_t q_s2,
    ptrdiff_t k_s0,
    ptrdiff_t k_s2,
    ptrdiff_t v_s0,
    ptrdiff_t v_s2,
    ptrdiff_t beta_s0,
    ptrdiff_t beta_s2) {

    size_t block = GetBlockIdx();

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> q_input_queue, k_input_queue, v_input_queue;
    TQue<QuePosition::VECOUT, 1> q_output_queue, k_output_queue;
    TBuf<QuePosition::VECCALC> q_float_buf, k_float_buf, state_copy_buf;
    pipe.InitBuffer(q_input_queue, 1, D * sizeof(bfloat16_t));
    pipe.InitBuffer(q_output_queue, 1, D * sizeof(bfloat16_t));
    pipe.InitBuffer(k_input_queue, 1, D * sizeof(bfloat16_t));
    pipe.InitBuffer(k_output_queue, 1, D * sizeof(bfloat16_t));
    pipe.InitBuffer(v_input_queue, 1, D * sizeof(bfloat16_t));
    pipe.InitBuffer(q_float_buf, D * sizeof(float));
    pipe.InitBuffer(k_float_buf, D * sizeof(float));
    pipe.InitBuffer(state_copy_buf, STATE_TILE * sizeof(bfloat16_t));

    GlobalTensor<bfloat16_t> q_normalized, k_normalized;
    GlobalTensor<bfloat16_t> v_contiguous, beta_bf16, state_staging;
    GlobalTensor<bfloat16_t> q, k, v, state;
    GlobalTensor<float> beta;
    GlobalTensor<int32_t> actual_seq_lengths, state_indices;
    q_normalized.SetGlobalBuffer(
        reinterpret_cast<__gm__ bfloat16_t *>(q_normalized_ptr));
    k_normalized.SetGlobalBuffer(
        reinterpret_cast<__gm__ bfloat16_t *>(k_normalized_ptr));
    v_contiguous.SetGlobalBuffer(
        reinterpret_cast<__gm__ bfloat16_t *>(v_contiguous_ptr));
    beta_bf16.SetGlobalBuffer(
        reinterpret_cast<__gm__ bfloat16_t *>(beta_bf16_ptr));
    state_staging.SetGlobalBuffer(
        reinterpret_cast<__gm__ bfloat16_t *>(state_staging_ptr));
    q.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(q_ptr));
    k.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(k_ptr));
    v.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(v_ptr));
    beta.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(beta_ptr));
    state.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(state_ptr));
    actual_seq_lengths.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(actual_seq_lengths_ptr));
    state_indices.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(state_indices_ptr));

    if (block < B * Hk) {
        size_t request = block / Hk;
        size_t head = block % Hk;
        ptrdiff_t q_base = static_cast<ptrdiff_t>(request) * q_s0
                         + static_cast<ptrdiff_t>(head) * q_s2;
        ptrdiff_t k_base = static_cast<ptrdiff_t>(request) * k_s0
                         + static_cast<ptrdiff_t>(head) * k_s2;
        size_t output_base = block * D;

        LocalTensor<bfloat16_t> q_input = q_input_queue.AllocTensor<bfloat16_t>();
        DataCopy(q_input, q[q_base], D);
        q_input_queue.EnQue(q_input);
        q_input = q_input_queue.DeQue<bfloat16_t>();
        LocalTensor<float> q_values = q_float_buf.Get<float>();
        Cast(q_values, q_input, AscendC::RoundMode::CAST_NONE, D);
        float q_sum = 0.0f;
        for (size_t i = 0; i < D; ++i) {
            float value = q_values.GetValue(i);
            q_sum += value * value;
        }
        Muls(q_values, q_values, nativeRsqrt(q_sum), D);
        LocalTensor<bfloat16_t> q_output = q_output_queue.AllocTensor<bfloat16_t>();
        Cast(q_output, q_values, AscendC::RoundMode::CAST_RINT, D);
        q_output_queue.EnQue(q_output);
        q_input_queue.FreeTensor(q_input);
        q_output = q_output_queue.DeQue<bfloat16_t>();
        DataCopy(q_normalized[output_base], q_output, D);
        q_output_queue.FreeTensor(q_output);

        LocalTensor<bfloat16_t> k_input = k_input_queue.AllocTensor<bfloat16_t>();
        DataCopy(k_input, k[k_base], D);
        k_input_queue.EnQue(k_input);
        k_input = k_input_queue.DeQue<bfloat16_t>();
        LocalTensor<float> k_values = k_float_buf.Get<float>();
        Cast(k_values, k_input, AscendC::RoundMode::CAST_NONE, D);
        float k_sum = 0.0f;
        for (size_t i = 0; i < D; ++i) {
            float value = k_values.GetValue(i);
            k_sum += value * value;
        }
        Muls(k_values, k_values, nativeRsqrt(k_sum), D);
        LocalTensor<bfloat16_t> k_output = k_output_queue.AllocTensor<bfloat16_t>();
        Cast(k_output, k_values, AscendC::RoundMode::CAST_RINT, D);
        k_output_queue.EnQue(k_output);
        k_input_queue.FreeTensor(k_input);
        k_output = k_output_queue.DeQue<bfloat16_t>();
        DataCopy(k_normalized[output_base], k_output, D);
        k_output_queue.FreeTensor(k_output);
    }

    if (block < B * Hv) {
        size_t request = block / Hv;
        size_t head = block % Hv;
        int64_t read_slot = loadIndex(
            initial_state_indices_ptr, initial_indices_i64, request);
        int64_t write_slot = loadIndex(
            final_state_indices_ptr, final_indices_i64, request);
        if (read_slot >= 0 && read_slot < static_cast<int64_t>(pool_size)
            && write_slot >= 0
            && write_slot < static_cast<int64_t>(pool_size)
            && read_slot != write_slot) {
            LocalTensor<bfloat16_t> state_local = state_copy_buf.Get<bfloat16_t>();
            size_t source_base = (static_cast<size_t>(read_slot) * Hv + head) * MATRIX;
            size_t destination_base = (static_cast<size_t>(write_slot) * Hv + head) * MATRIX;
            TEventID mte2_event = GetTPipePtr()->FetchEventID(HardEvent::MTE2_S);
            TEventID mte3_event = GetTPipePtr()->FetchEventID(HardEvent::MTE3_S);
            for (size_t tile = 0; tile < MATRIX; tile += STATE_TILE) {
                DataCopy(state_local, state[source_base + tile], STATE_TILE);
                SetFlag<HardEvent::MTE2_S>(mte2_event);
                WaitFlag<HardEvent::MTE2_S>(mte2_event);
                DataCopy(
                    state[destination_base + tile],
                    state_local, STATE_TILE);
                SetFlag<HardEvent::MTE3_S>(mte3_event);
                WaitFlag<HardEvent::MTE3_S>(mte3_event);
            }
        }
    }
}

__global__ __aicore__ void recurrent_gdr_native_metadata(
    GM_ADDR actual_seq_lengths_ptr,
    GM_ADDR state_indices_ptr,
    GM_ADDR initial_state_indices_ptr,
    size_t B,
    bool initial_indices_i64) {

    GlobalTensor<int32_t> actual_seq_lengths, state_indices;
    actual_seq_lengths.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(actual_seq_lengths_ptr));
    state_indices.SetGlobalBuffer(
        reinterpret_cast<__gm__ int32_t *>(state_indices_ptr));
    for (size_t request = 0; request < B; ++request) {
        int64_t state_slot = loadIndex(
            initial_state_indices_ptr, initial_indices_i64, request);
        actual_seq_lengths.SetValue(request, 1);
        state_indices.SetValue(request, static_cast<int32_t>(state_slot));
    }
}

__global__ __aicore__ void recurrent_gdr_native_cast_beta(
    GM_ADDR beta_bf16_ptr,
    GM_ADDR beta_ptr,
    size_t count) {

    size_t copy_len = alignTileLen<float>(count, BYTE_ALIGN);
    GlobalTensor<float> beta;
    GlobalTensor<bfloat16_t> beta_bf16;
    beta.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(beta_ptr));
    beta_bf16.SetGlobalBuffer(
        reinterpret_cast<__gm__ bfloat16_t *>(beta_bf16_ptr));

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> input_queue;
    TQue<QuePosition::VECOUT, 1> output_queue;
    pipe.InitBuffer(input_queue, 1, copy_len * sizeof(float));
    pipe.InitBuffer(output_queue, 1, copy_len * sizeof(bfloat16_t));

    LocalTensor<float> input = input_queue.AllocTensor<float>();
    DataCopy(input, beta, copy_len);
    input_queue.EnQue(input);
    input = input_queue.DeQue<float>();
    LocalTensor<bfloat16_t> output = output_queue.AllocTensor<bfloat16_t>();
    Cast(output, input, AscendC::RoundMode::CAST_RINT, copy_len);
    output_queue.EnQue(output);
    input_queue.FreeTensor(input);
    output = output_queue.DeQue<bfloat16_t>();
    if (count * sizeof(bfloat16_t) % BYTE_ALIGN != 0) {
        DataCopyExtParams params = {
            1, static_cast<uint32_t>(count * sizeof(bfloat16_t)), 0, 0, 0};
        DataCopyPad(beta_bf16, output, params);
    } else {
        DataCopy(beta_bf16, output, count);
    }
    output_queue.FreeTensor(output);
}

__global__ __aicore__ void recurrent_gdr_native_commit_state(
    GM_ADDR state_ptr,
    GM_ADDR state_staging_ptr,
    GM_ADDR initial_state_indices_ptr,
    GM_ADDR final_state_indices_ptr,
    size_t B,
    size_t Hv,
    size_t pool_size,
    bool initial_indices_i64,
    bool final_indices_i64) {

    size_t block = GetBlockIdx();
    if (block >= B * Hv) {
        return;
    }
    size_t request = block / Hv;
    size_t head = block % Hv;
    int64_t read_slot = loadIndex(
        initial_state_indices_ptr, initial_indices_i64, request);
    int64_t write_slot = loadIndex(
        final_state_indices_ptr, final_indices_i64, request);
    if (read_slot < 0 || read_slot >= static_cast<int64_t>(pool_size)
        || write_slot < 0 || write_slot >= static_cast<int64_t>(pool_size)
        || read_slot == write_slot) {
        return;
    }

    GlobalTensor<bfloat16_t> state, state_staging;
    state.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t *>(state_ptr));
    state_staging.SetGlobalBuffer(
        reinterpret_cast<__gm__ bfloat16_t *>(state_staging_ptr));

    TPipe pipe;
    TBuf<QuePosition::VECCALC> state_buf;
    pipe.InitBuffer(state_buf, STATE_TILE * sizeof(bfloat16_t));
    LocalTensor<bfloat16_t> state_local = state_buf.Get<bfloat16_t>();
    size_t staging_base = block * MATRIX;
    size_t source_base = (static_cast<size_t>(read_slot) * Hv + head) * MATRIX;
    size_t destination_base = (static_cast<size_t>(write_slot) * Hv + head) * MATRIX;
    TEventID mte2_event = GetTPipePtr()->FetchEventID(HardEvent::MTE2_S);
    TEventID mte3_event = GetTPipePtr()->FetchEventID(HardEvent::MTE3_S);
    for (size_t tile = 0; tile < MATRIX; tile += STATE_TILE) {
        DataCopy(state_local, state[source_base + tile], STATE_TILE);
        SetFlag<HardEvent::MTE2_S>(mte2_event);
        WaitFlag<HardEvent::MTE2_S>(mte2_event);
        DataCopy(state[destination_base + tile], state_local, STATE_TILE);
        SetFlag<HardEvent::MTE3_S>(mte3_event);
        WaitFlag<HardEvent::MTE3_S>(mte3_event);
        DataCopy(state_local, state_staging[staging_base + tile], STATE_TILE);
        SetFlag<HardEvent::MTE2_S>(mte2_event);
        WaitFlag<HardEvent::MTE2_S>(mte2_event);
        DataCopy(state[source_base + tile], state_local, STATE_TILE);
        SetFlag<HardEvent::MTE3_S>(mte3_event);
        WaitFlag<HardEvent::MTE3_S>(mte3_event);
    }
}

} // namespace

extern "C" infiniStatus_t recurrent_gdr_native_preprocess_launch(
    void *q_normalized,
    void *k_normalized,
    void *v_contiguous,
    void *beta_bf16,
    void *state_staging,
    void *actual_seq_lengths,
    void *state_indices,
    const void *q,
    const void *k,
    const void *v,
    const void *beta,
    const void *state,
    const void *initial_state_indices,
    const void *final_state_indices,
    const RecurrentGdrNativeParams *p,
    void *stream) {

    if (p->B == 0) {
        return INFINI_STATUS_SUCCESS;
    }
    uint32_t blocks = static_cast<uint32_t>(p->B * (p->Hv > p->Hk ? p->Hv : p->Hk));
    recurrent_gdr_native_preprocess<<<blocks, nullptr, stream>>>(
        q_normalized, k_normalized, v_contiguous, beta_bf16, state_staging,
        actual_seq_lengths, state_indices, const_cast<void *>(q),
        const_cast<void *>(k), const_cast<void *>(v), const_cast<void *>(beta),
        const_cast<void *>(state), const_cast<void *>(initial_state_indices),
        const_cast<void *>(final_state_indices), p->B, p->Hk, p->Hv,
        p->pool_size, p->initial_indices_i64, p->final_indices_i64,
        p->q_s0, p->q_s2, p->k_s0, p->k_s2, p->v_s0, p->v_s2,
        p->beta_s0, p->beta_s2);
    recurrent_gdr_native_metadata<<<1, nullptr, stream>>>(
        actual_seq_lengths, state_indices,
        const_cast<void *>(final_state_indices),
        p->B, p->final_indices_i64);
    recurrent_gdr_native_cast_beta<<<1, nullptr, stream>>>(
        beta_bf16, const_cast<void *>(beta), p->B * p->Hv);
    return INFINI_STATUS_SUCCESS;
}

extern "C" infiniStatus_t recurrent_gdr_native_commit_state_launch(
    void *state,
    const void *state_staging,
    const void *initial_state_indices,
    const void *final_state_indices,
    const RecurrentGdrNativeParams *p,
    void *stream) {

    if (p->B == 0) {
        return INFINI_STATUS_SUCCESS;
    }
    uint32_t blocks = static_cast<uint32_t>(p->B * p->Hv);
    recurrent_gdr_native_commit_state<<<blocks, nullptr, stream>>>(
        state, const_cast<void *>(state_staging),
        const_cast<void *>(initial_state_indices),
        const_cast<void *>(final_state_indices), p->B, p->Hv, p->pool_size,
        p->initial_indices_i64, p->final_indices_i64);
    return INFINI_STATUS_SUCCESS;
}
