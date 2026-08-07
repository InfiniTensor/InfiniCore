#include "chunk_gated_delta_rule_ascend.h"
#include "../../../devices/ascend/ascend_handle.h"
#include "../../../devices/ascend/common_ascend.h"
#include "../../recurrent_gated_delta_rule/ascend/recurrent_gated_delta_rule_native_kernel.h"
#include "gated_delta_rule_ascend_kernel.h"

#include <aclnnop/aclnn_recurrent_gated_delta_rule.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <vector>

namespace op::chunk_gated_delta_rule::ascend {

namespace {

constexpr size_t ACLNN_DIM = 128;
constexpr size_t ACLNN_MAX_SEQ_LEN = 8;
constexpr size_t ACLNN_ALIGNMENT = 512;

size_t alignAclnn(size_t value) {
    return (value + ACLNN_ALIGNMENT - 1) & ~(ACLNN_ALIGNMENT - 1);
}

size_t reserveAclnn(size_t &cursor, size_t bytes) {
    cursor = alignAclnn(cursor);
    const size_t result = cursor;
    cursor += bytes;
    return result;
}

template <typename T>
infiniStatus_t copyDeviceVector(
    std::vector<int64_t> &dst, const void *src, size_t count) {
    std::vector<T> temporary(count);
    CHECK_ACL(aclrtMemcpy(
        temporary.data(), temporary.size() * sizeof(T), src,
        temporary.size() * sizeof(T), ACL_MEMCPY_DEVICE_TO_HOST));
    for (size_t i = 0; i < count; ++i) {
        dst[i] = static_cast<int64_t>(temporary[i]);
    }
    return INFINI_STATUS_SUCCESS;
}

struct Segment {
    size_t offset;
    size_t length;
    int32_t state_slot;
};

} // namespace

struct Descriptor::Opaque {
    struct Call {
        aclnnTensorDescriptor_t q = nullptr;
        aclnnTensorDescriptor_t k = nullptr;
        aclnnTensorDescriptor_t v = nullptr;
        aclnnTensorDescriptor_t beta = nullptr;
        aclnnTensorDescriptor_t state = nullptr;
        aclnnTensorDescriptor_t actual_seq_lengths = nullptr;
        aclnnTensorDescriptor_t state_indices = nullptr;
        aclnnTensorDescriptor_t g = nullptr;
        aclnnTensorDescriptor_t out = nullptr;
        aclOpExecutor *executor = nullptr;
        uint64_t workspace_size = 0;

        ~Call() {
            delete q;
            delete k;
            delete v;
            delete beta;
            delete state;
            delete actual_seq_lengths;
            delete state_indices;
            delete g;
            delete out;
        }
    };

    bool aclnn = false;
    std::array<std::vector<std::unique_ptr<Call>>, ACLNN_MAX_SEQ_LEN + 1> calls;
    size_t q_offset = 0;
    size_t k_offset = 0;
    size_t beta_offset = 0;
    size_t input_indices_offset = 0;
    size_t state_indices_offset = 0;
    size_t actual_seq_lengths_offset = 0;
    size_t state_staging_offset = 0;
    size_t aclnn_workspace_offset = 0;
    uint64_t aclnn_workspace_size = 0;
    void *buffer = nullptr;
    int32_t *host_indices = nullptr;
    int32_t *host_lengths = nullptr;
    size_t metadata_capacity = 0;

    ~Opaque() {
        if (buffer != nullptr) {
            aclrtFree(buffer);
        }
        if (host_indices != nullptr) {
            aclrtFreeHost(host_indices);
        }
        if (host_lengths != nullptr) {
            aclrtFreeHost(host_lengths);
        }
    }
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t initial_state_desc,
    infiniopTensorDescriptor_t final_state_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t g_desc,
    infiniopTensorDescriptor_t beta_desc,
    infiniopTensorDescriptor_t cu_seqlens_desc,
    infiniopTensorDescriptor_t initial_state_indices_desc,
    infiniopTensorDescriptor_t final_state_indices_desc,
    bool use_qk_l2norm,
    size_t chunk_size) {

    auto result = ChunkGatedDeltaRuleInfo::create(
        out_desc, initial_state_desc, final_state_desc, q_desc, k_desc, v_desc,
        g_desc, beta_desc, cu_seqlens_desc, initial_state_indices_desc,
        final_state_indices_desc, use_qk_l2norm, chunk_size);
    CHECK_RESULT(result);
    auto info = result.take();
    auto opaque = new Opaque{};

    const bool state_contiguous = info.initial_state_strides[3] == 1
                               && info.initial_state_strides[2] == static_cast<ptrdiff_t>(info.Dk)
                               && info.initial_state_strides[1]
                                      == static_cast<ptrdiff_t>(info.Dv * info.Dk)
                               && info.initial_state_strides[0]
                                      == static_cast<ptrdiff_t>(info.Hv * info.Dv * info.Dk);
    const bool tensors_contiguous = info.q_strides[1] == static_cast<ptrdiff_t>(info.Hk * info.Dk)
                                 && info.q_strides[2] == static_cast<ptrdiff_t>(info.Dk)
                                 && info.q_strides[3] == 1
                                 && info.k_strides[1] == static_cast<ptrdiff_t>(info.Hk * info.Dk)
                                 && info.k_strides[2] == static_cast<ptrdiff_t>(info.Dk)
                                 && info.k_strides[3] == 1
                                 && info.v_strides[1] == static_cast<ptrdiff_t>(info.Hv * info.Dv)
                                 && info.v_strides[2] == static_cast<ptrdiff_t>(info.Dv)
                                 && info.v_strides[3] == 1
                                 && info.out_strides[1] == static_cast<ptrdiff_t>(info.Hv * info.Dv)
                                 && info.out_strides[2] == static_cast<ptrdiff_t>(info.Dv)
                                 && info.out_strides[3] == 1
                                 && info.g_strides[1] == static_cast<ptrdiff_t>(info.Hv)
                                 && info.g_strides[2] == 1
                                 && info.beta_strides[1] == static_cast<ptrdiff_t>(info.Hv)
                                 && info.beta_strides[2] == 1;
    opaque->aclnn = info.data_dtype == INFINI_DTYPE_BF16
                 && info.gate_dtype == INFINI_DTYPE_F32
                 && info.use_qk_l2norm
                 && info.has_cu_seqlens
                 && info.has_initial_state_indices
                 && info.has_final_state_indices
                 && info.Dk == ACLNN_DIM
                 && info.Dv == ACLNN_DIM
                 && state_contiguous
                 && tensors_contiguous;

    size_t workspace_size = info.B * info.Hv * info.Dv * info.Dk
                          * sizeof(float);
    if (opaque->aclnn) {
        const int64_t Hk = static_cast<int64_t>(info.Hk);
        const int64_t Hv = static_cast<int64_t>(info.Hv);
        const int64_t D = static_cast<int64_t>(ACLNN_DIM);
        const int64_t pool = static_cast<int64_t>(info.pool_size);
        for (size_t length = 1; length <= ACLNN_MAX_SEQ_LEN; ++length) {
            const size_t call_count = length == ACLNN_MAX_SEQ_LEN
                                        ? std::max<size_t>(1, info.total_tokens / ACLNN_MAX_SEQ_LEN + info.B)
                                        : std::max<size_t>(1, info.B);
            auto &pool_calls = opaque->calls[length];
            pool_calls.reserve(call_count);
            for (size_t call_index = 0; call_index < call_count; ++call_index) {
                auto call = std::make_unique<Opaque::Call>();
                const int64_t L = static_cast<int64_t>(length);
                call->q = new aclnnTensorDescriptor(
                    ACL_BF16, {L, Hk, D}, {Hk * D, D, 1}, nullptr);
                call->k = new aclnnTensorDescriptor(
                    ACL_BF16, {L, Hk, D}, {Hk * D, D, 1}, nullptr);
                call->v = new aclnnTensorDescriptor(
                    ACL_BF16, {L, Hv, D}, {Hv * D, D, 1}, nullptr);
                call->beta = new aclnnTensorDescriptor(
                    ACL_BF16, {L, Hv}, {Hv, 1}, nullptr);
                call->state = new aclnnTensorDescriptor(
                    ACL_BF16, {pool, Hv, D, D},
                    {Hv * D * D, D * D, D, 1}, nullptr);
                call->actual_seq_lengths = new aclnnTensorDescriptor(
                    ACL_INT32, {1}, {1}, nullptr);
                call->state_indices = new aclnnTensorDescriptor(
                    ACL_INT32, {L}, {1}, nullptr);
                call->g = new aclnnTensorDescriptor(
                    ACL_FLOAT, {L, Hv}, {Hv, 1}, nullptr);
                call->out = new aclnnTensorDescriptor(
                    ACL_BF16, {L, Hv, D}, {Hv * D, D, 1}, nullptr);
                CHECK_ACL(aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(
                    call->q->tensor, call->k->tensor, call->v->tensor,
                    call->beta->tensor, call->state->tensor,
                    call->actual_seq_lengths->tensor, call->state_indices->tensor,
                    call->g->tensor, nullptr, nullptr,
                    1.0f / std::sqrt(static_cast<float>(ACLNN_DIM)),
                    call->out->tensor, &call->workspace_size, &call->executor));
                CHECK_ACL(aclSetAclOpExecutorRepeatable(call->executor));
                opaque->aclnn_workspace_size = std::max(
                    opaque->aclnn_workspace_size, call->workspace_size);
                pool_calls.push_back(std::move(call));
            }
        }

        size_t cursor = 0;
        opaque->q_offset = reserveAclnn(
            cursor, ACLNN_MAX_SEQ_LEN * info.Hk * ACLNN_DIM
                        * sizeof(uint16_t));
        opaque->k_offset = reserveAclnn(
            cursor, ACLNN_MAX_SEQ_LEN * info.Hk * ACLNN_DIM
                        * sizeof(uint16_t));
        opaque->beta_offset = reserveAclnn(
            cursor, ACLNN_MAX_SEQ_LEN * info.Hv * sizeof(uint16_t));
        opaque->input_indices_offset = reserveAclnn(
            cursor, ACLNN_MAX_SEQ_LEN * sizeof(int32_t));
        opaque->state_indices_offset = reserveAclnn(
            cursor, ACLNN_MAX_SEQ_LEN * sizeof(int32_t));
        opaque->actual_seq_lengths_offset = reserveAclnn(
            cursor, ACLNN_MAX_SEQ_LEN * sizeof(int32_t));
        opaque->state_staging_offset = reserveAclnn(cursor, ACLNN_ALIGNMENT);
        opaque->aclnn_workspace_offset = alignAclnn(cursor);
        cursor = opaque->aclnn_workspace_offset
               + opaque->aclnn_workspace_size;
        CHECK_ACL(aclrtMalloc(
            &opaque->buffer, cursor, ACL_MEM_MALLOC_HUGE_FIRST));
        opaque->metadata_capacity = info.total_tokens / ACLNN_MAX_SEQ_LEN + info.B;
        CHECK_ACL(aclrtMallocHost(
            reinterpret_cast<void **>(&opaque->host_indices),
            opaque->metadata_capacity * ACLNN_MAX_SEQ_LEN * sizeof(int32_t)));
        CHECK_ACL(aclrtMallocHost(
            reinterpret_cast<void **>(&opaque->host_lengths),
            opaque->metadata_capacity * sizeof(int32_t)));
        workspace_size = std::max(workspace_size, cursor);
    }

    *desc_ptr = new Descriptor(
        opaque, std::move(info), workspace_size,
        handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
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
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    if (_info.gate_dtype != INFINI_DTYPE_F32) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    if (_opaque->aclnn) {
        std::vector<int64_t> cu(_info.B + 1);
        std::vector<int64_t> initial_indices(_info.B);
        std::vector<int64_t> final_indices(_info.B);
        if (_info.cu_seqlens_dtype == INFINI_DTYPE_I64) {
            CHECK_STATUS(copyDeviceVector<int64_t>(
                cu, cu_seqlens, _info.B + 1));
        } else {
            CHECK_STATUS(copyDeviceVector<int32_t>(
                cu, cu_seqlens, _info.B + 1));
        }
        if (_info.initial_state_indices_dtype == INFINI_DTYPE_I64) {
            CHECK_STATUS(copyDeviceVector<int64_t>(
                initial_indices, initial_state_indices, _info.B));
        } else {
            CHECK_STATUS(copyDeviceVector<int32_t>(
                initial_indices, initial_state_indices, _info.B));
        }
        if (_info.final_state_indices_dtype == INFINI_DTYPE_I64) {
            CHECK_STATUS(copyDeviceVector<int64_t>(
                final_indices, final_state_indices, _info.B));
        } else {
            CHECK_STATUS(copyDeviceVector<int32_t>(
                final_indices, final_state_indices, _info.B));
        }

        std::vector<Segment> segments;
        const size_t state_bytes = _info.Hv * ACLNN_DIM * ACLNN_DIM * sizeof(uint16_t);
        auto *state_bytes_ptr = static_cast<uint8_t *>(initial_state);
        for (size_t request = 0; request < _info.B; ++request) {
            if (cu[request] < 0 || cu[request + 1] < cu[request]
                || cu[request + 1]
                       > static_cast<int64_t>(_info.total_tokens)
                || initial_indices[request] < 0
                || final_indices[request] < 0
                || initial_indices[request]
                       >= static_cast<int64_t>(_info.pool_size)
                || final_indices[request]
                       >= static_cast<int64_t>(_info.pool_size)) {
                return INFINI_STATUS_BAD_PARAM;
            }
            const size_t source = static_cast<size_t>(initial_indices[request]);
            const size_t destination = static_cast<size_t>(final_indices[request]);
            if (source != destination) {
                CHECK_ACL(aclrtMemcpyAsync(
                    state_bytes_ptr + destination * state_bytes, state_bytes,
                    state_bytes_ptr + source * state_bytes, state_bytes,
                    ACL_MEMCPY_DEVICE_TO_DEVICE, stream));
            }
            for (size_t offset = static_cast<size_t>(cu[request]);
                 offset < static_cast<size_t>(cu[request + 1]);
                 offset += ACLNN_MAX_SEQ_LEN) {
                const size_t length = std::min(
                    ACLNN_MAX_SEQ_LEN,
                    static_cast<size_t>(cu[request + 1]) - offset);
                segments.push_back(Segment{
                    offset, length, static_cast<int32_t>(destination)});
            }
        }

        if (segments.size() > _opaque->metadata_capacity) {
            return INFINI_STATUS_BAD_PARAM;
        }
        int32_t *host_indices = _opaque->host_indices;
        int32_t *host_lengths = _opaque->host_lengths;
        for (size_t i = 0; i < segments.size(); ++i) {
            std::fill_n(
                host_indices + i * ACLNN_MAX_SEQ_LEN,
                ACLNN_MAX_SEQ_LEN, segments[i].state_slot);
            host_lengths[i] = static_cast<int32_t>(segments[i].length);
        }

        auto *base = static_cast<uint8_t *>(_opaque->buffer);
        void *q_normalized = base + _opaque->q_offset;
        void *k_normalized = base + _opaque->k_offset;
        void *beta_bf16 = base + _opaque->beta_offset;
        void *input_indices = base + _opaque->input_indices_offset;
        void *state_indices = base + _opaque->state_indices_offset;
        void *actual_seq_lengths = base + _opaque->actual_seq_lengths_offset;
        void *state_staging = base + _opaque->state_staging_offset;
        void *aclnn_workspace = base + _opaque->aclnn_workspace_offset;

        auto add_bytes = [](const void *ptr, size_t bytes) -> void * {
            return const_cast<uint8_t *>(
                       static_cast<const uint8_t *>(ptr))
                 + bytes;
        };
        std::array<size_t, ACLNN_MAX_SEQ_LEN + 1> call_cursors{};
        for (size_t i = 0; i < segments.size(); ++i) {
            const auto &segment = segments[i];
            const size_t q_bytes = segment.offset * _info.Hk * ACLNN_DIM * sizeof(uint16_t);
            const size_t v_bytes = segment.offset * _info.Hv * ACLNN_DIM * sizeof(uint16_t);
            const size_t gate_bytes = segment.offset * _info.Hv * sizeof(float);
            CHECK_ACL(aclrtMemcpyAsync(
                input_indices,
                ACLNN_MAX_SEQ_LEN * sizeof(int32_t),
                host_indices + i * ACLNN_MAX_SEQ_LEN,
                ACLNN_MAX_SEQ_LEN * sizeof(int32_t),
                ACL_MEMCPY_HOST_TO_DEVICE, stream));

            RecurrentGdrNativeParams params{};
            params.B = segment.length;
            params.Hk = _info.Hk;
            params.Hv = _info.Hv;
            params.pool_size = _info.pool_size;
            params.initial_indices_i64 = false;
            params.final_indices_i64 = false;
            params.q_s0 = _info.q_strides[1];
            params.q_s2 = _info.q_strides[2];
            params.k_s0 = _info.k_strides[1];
            params.k_s2 = _info.k_strides[2];
            params.v_s0 = _info.v_strides[1];
            params.v_s2 = _info.v_strides[2];
            params.beta_s0 = _info.beta_strides[1];
            params.beta_s2 = _info.beta_strides[2];
            CHECK_STATUS(recurrent_gdr_native_preprocess_launch(
                q_normalized, k_normalized,
                add_bytes(v, v_bytes), beta_bf16, state_staging,
                actual_seq_lengths, state_indices,
                add_bytes(q, q_bytes), add_bytes(k, q_bytes),
                add_bytes(v, v_bytes), add_bytes(beta, gate_bytes),
                initial_state, input_indices, input_indices,
                &params, stream));
            CHECK_ACL(aclrtMemcpyAsync(
                state_indices,
                segment.length * sizeof(int32_t),
                host_indices + i * ACLNN_MAX_SEQ_LEN,
                segment.length * sizeof(int32_t),
                ACL_MEMCPY_HOST_TO_DEVICE, stream));
            CHECK_ACL(aclrtMemcpyAsync(
                actual_seq_lengths, sizeof(int32_t),
                host_lengths + i, sizeof(int32_t),
                ACL_MEMCPY_HOST_TO_DEVICE, stream));

            const size_t length = segment.length;
            const size_t call_index = call_cursors[length]++;
            if (call_index >= _opaque->calls[length].size()) {
                return INFINI_STATUS_BAD_PARAM;
            }
            const auto &call = *_opaque->calls[length][call_index];
            CHECK_ACL(AclSetTensorAddr(call.executor, 0, call.q->tensor, q_normalized));
            CHECK_ACL(AclSetTensorAddr(call.executor, 1, call.k->tensor, k_normalized));
            CHECK_ACL(AclSetTensorAddr(call.executor, 2, call.v->tensor, add_bytes(v, v_bytes)));
            CHECK_ACL(AclSetTensorAddr(call.executor, 3, call.beta->tensor, beta_bf16));
            CHECK_ACL(AclSetTensorAddr(call.executor, 4, call.state->tensor, initial_state));
            CHECK_ACL(AclSetTensorAddr(call.executor, 5, call.actual_seq_lengths->tensor, actual_seq_lengths));
            CHECK_ACL(AclSetTensorAddr(call.executor, 6, call.state_indices->tensor, state_indices));
            CHECK_ACL(AclSetTensorAddr(call.executor, 7, call.g->tensor, add_bytes(g, gate_bytes)));
            CHECK_ACL(AclSetTensorAddr(call.executor, 8, call.out->tensor, add_bytes(out, v_bytes)));
            CHECK_ACL(aclnnRecurrentGatedDeltaRule(
                aclnn_workspace, call.workspace_size, call.executor, stream));
        }
        CHECK_ACL(aclrtSynchronizeStream(stream));
        return INFINI_STATUS_SUCCESS;
    }

    GatedDeltaRuleAscendParams p{};
    p.data_dtype = static_cast<int32_t>(_info.data_dtype);
    p.gate_dtype = static_cast<int32_t>(_info.gate_dtype);
    p.use_qk_l2norm = _info.use_qk_l2norm;
    p.has_cu_seqlens = _info.has_cu_seqlens;
    p.cu_seqlens_i64 = _info.cu_seqlens_dtype == INFINI_DTYPE_I64;
    p.has_initial_indices = _info.has_initial_state_indices;
    p.initial_indices_i64 = _info.initial_state_indices_dtype == INFINI_DTYPE_I64;
    p.has_final_indices = _info.has_final_state_indices;
    p.final_indices_i64 = _info.final_state_indices_dtype == INFINI_DTYPE_I64;
    p.B = _info.B;
    p.T = _info.T;
    p.total_tokens = _info.total_tokens;
    p.Hk = _info.Hk;
    p.Hv = _info.Hv;
    p.Dk = _info.Dk;
    p.Dv = _info.Dv;
    p.pool_size = _info.pool_size;
    p.value_heads_per_key_head = _info.value_heads_per_key_head;
    p.q_scale = 1.0f / std::sqrt(static_cast<float>(_info.Dk));
    for (int i = 0; i < 4; ++i) {
        p.out_strides[i] = _info.out_strides[i];
        p.q_strides[i] = _info.q_strides[i];
        p.k_strides[i] = _info.k_strides[i];
        p.v_strides[i] = _info.v_strides[i];
    }
    return gated_delta_rule_ascend_kernel_launch(
        workspace, out, initial_state, final_state, q, k, v, g, beta,
        cu_seqlens, initial_state_indices, final_state_indices, &p, stream);
}

} // namespace op::chunk_gated_delta_rule::ascend
