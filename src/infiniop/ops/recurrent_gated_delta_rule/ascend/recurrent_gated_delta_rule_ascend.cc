#include "recurrent_gated_delta_rule_ascend.h"
#include "../../../devices/ascend/ascend_handle.h"
#include "../../../devices/ascend/common_ascend.h"
#include "../../chunk_gated_delta_rule/ascend/gated_delta_rule_ascend_kernel.h"
#include "recurrent_gated_delta_rule_native_kernel.h"

#include <aclnnop/aclnn_recurrent_gated_delta_rule.h>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace op::recurrent_gated_delta_rule::ascend {

namespace {

constexpr size_t NATIVE_DIM = 128;
constexpr size_t NATIVE_ALIGNMENT = 512;

size_t alignNative(size_t value) {
    return (value + NATIVE_ALIGNMENT - 1) & ~(NATIVE_ALIGNMENT - 1);
}

size_t reserveNative(size_t &cursor, size_t bytes) {
    cursor = alignNative(cursor);
    size_t result = cursor;
    cursor += bytes;
    return result;
}

} // namespace

struct Descriptor::Opaque {
    bool native = false;
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
    void *q_buffer = nullptr;
    void *k_buffer = nullptr;
    void *v_buffer = nullptr;
    void *beta_buffer = nullptr;
    void *lengths_buffer = nullptr;
    void *indices_buffer = nullptr;
    void *native_workspace_buffer = nullptr;
    uint64_t native_workspace_size = 0;
    std::vector<std::vector<void *>> cached_addresses;
    std::vector<std::vector<aclnnTensorDescriptor *>> cached_descriptors;
    std::vector<aclOpExecutor *> cached_executors;
    std::vector<uint64_t> cached_workspace_sizes;
    size_t q_offset = 0;
    size_t k_offset = 0;
    size_t v_offset = 0;
    size_t beta_offset = 0;
    size_t state_offset = 0;
    size_t actual_seq_lengths_offset = 0;
    size_t state_indices_offset = 0;
    size_t native_workspace_offset = 0;

    ~Opaque() {
        for (auto &descriptors : cached_descriptors) {
            for (auto *descriptor : descriptors) {
                delete descriptor;
            }
        }
        delete q;
        delete k;
        delete v;
        delete beta;
        delete state;
        delete actual_seq_lengths;
        delete state_indices;
        delete g;
        delete out;
        if (q_buffer != nullptr) {
            aclrtFree(q_buffer);
        }
        if (k_buffer != nullptr) {
            aclrtFree(k_buffer);
        }
        if (v_buffer != nullptr) {
            aclrtFree(v_buffer);
        }
        if (beta_buffer != nullptr) {
            aclrtFree(beta_buffer);
        }
        if (lengths_buffer != nullptr) {
            aclrtFree(lengths_buffer);
        }
        if (indices_buffer != nullptr) {
            aclrtFree(indices_buffer);
        }
        if (native_workspace_buffer != nullptr) {
            aclrtFree(native_workspace_buffer);
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
    infiniopTensorDescriptor_t initial_state_indices_desc,
    infiniopTensorDescriptor_t final_state_indices_desc,
    bool use_qk_l2norm) {

    auto result = RecurrentGatedDeltaRuleInfo::create(
        out_desc, initial_state_desc, final_state_desc, q_desc, k_desc, v_desc,
        g_desc, beta_desc, initial_state_indices_desc, final_state_indices_desc,
        use_qk_l2norm);
    CHECK_RESULT(result);
    auto info = result.take();
    auto opaque = new Opaque{};

    const bool state_contiguous = info.initial_state_strides[3] == 1
                               && info.initial_state_strides[2] == static_cast<ptrdiff_t>(info.Dk)
                               && info.initial_state_strides[1]
                                      == static_cast<ptrdiff_t>(info.Dv * info.Dk)
                               && info.initial_state_strides[0]
                                      == static_cast<ptrdiff_t>(info.Hv * info.Dv * info.Dk);
    // The native preprocess kernel reads Q/K through their explicit strides
    // while normalizing them, so their outer stride need not be contiguous.
    const bool tensors_contiguous = info.q_strides[2] == static_cast<ptrdiff_t>(info.Dk)
                                 && info.k_strides[2] == static_cast<ptrdiff_t>(info.Dk)
                                 && info.v_strides[0] == static_cast<ptrdiff_t>(info.Hv * info.Dv)
                                 && info.v_strides[2] == static_cast<ptrdiff_t>(info.Dv)
                                 && info.out_strides[0] == static_cast<ptrdiff_t>(info.Hv * info.Dv)
                                 && info.out_strides[2] == static_cast<ptrdiff_t>(info.Dv)
                                 && info.g_strides[0] == static_cast<ptrdiff_t>(info.Hv)
                                 && info.g_strides[2] == 1
                                 && info.beta_strides[0] == static_cast<ptrdiff_t>(info.Hv)
                                 && info.beta_strides[2] == 1;
    opaque->native = info.data_dtype == INFINI_DTYPE_BF16
                  && info.gate_dtype == INFINI_DTYPE_F32
                  && info.use_qk_l2norm
                  && info.has_initial_state_indices
                  && info.has_final_state_indices
                  && info.Dk == NATIVE_DIM
                  && info.Dv == NATIVE_DIM
                  && state_contiguous
                  && tensors_contiguous;

    size_t workspace_size = info.B * info.Hv * info.Dv * info.Dk * sizeof(float);
    if (opaque->native) {
        const int64_t B = static_cast<int64_t>(info.B);
        const int64_t Hk = static_cast<int64_t>(info.Hk);
        const int64_t Hv = static_cast<int64_t>(info.Hv);
        const int64_t D = static_cast<int64_t>(NATIVE_DIM);
        opaque->q = new aclnnTensorDescriptor(
            ACL_BF16, {B, Hk, D}, {Hk * D, D, 1});
        opaque->k = new aclnnTensorDescriptor(
            ACL_BF16, {B, Hk, D}, {Hk * D, D, 1});
        opaque->v = new aclnnTensorDescriptor(
            ACL_BF16, {B, Hv, D}, {Hv * D, D, 1});
        opaque->beta = new aclnnTensorDescriptor(
            ACL_BF16, {B, Hv}, {Hv, 1});
        const int64_t pool = static_cast<int64_t>(info.pool_size);
        opaque->state = new aclnnTensorDescriptor(
            ACL_BF16, {pool, Hv, D, D}, {Hv * D * D, D * D, D, 1});
        opaque->actual_seq_lengths = new aclnnTensorDescriptor(
            ACL_INT32, {B}, {1});
        opaque->state_indices = new aclnnTensorDescriptor(
            ACL_INT32, {B}, {1});
        opaque->g = new aclnnTensorDescriptor(
            ACL_FLOAT, {B, Hv}, {Hv, 1});
        opaque->out = new aclnnTensorDescriptor(
            ACL_BF16, {B, Hv, D}, {Hv * D, D, 1});

        size_t cursor = 0;
        opaque->q_offset = reserveNative(
            cursor, info.B * info.Hk * NATIVE_DIM * sizeof(uint16_t));
        opaque->k_offset = reserveNative(
            cursor, info.B * info.Hk * NATIVE_DIM * sizeof(uint16_t));
        opaque->v_offset = reserveNative(
            cursor, info.B * info.Hv * NATIVE_DIM * sizeof(uint16_t));
        opaque->beta_offset = reserveNative(
            cursor, info.B * info.Hv * sizeof(uint16_t));
        opaque->state_offset = reserveNative(
            cursor, info.B * info.Hv * NATIVE_DIM * NATIVE_DIM
                        * sizeof(uint16_t));
        opaque->actual_seq_lengths_offset = reserveNative(
            cursor, info.B * sizeof(int32_t));
        opaque->state_indices_offset = reserveNative(
            cursor, info.B * sizeof(int32_t));
        opaque->native_workspace_offset = alignNative(cursor);

        CHECK_ACL(aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(
            opaque->q->tensor,
            opaque->k->tensor,
            opaque->v->tensor,
            opaque->beta->tensor,
            opaque->state->tensor,
            opaque->actual_seq_lengths->tensor,
            opaque->state_indices->tensor,
            opaque->g->tensor,
            nullptr,
            nullptr,
            1.0f / std::sqrt(static_cast<float>(NATIVE_DIM)),
            opaque->out->tensor,
            &opaque->native_workspace_size,
            &opaque->executor));
        CHECK_ACL(aclSetAclOpExecutorRepeatable(opaque->executor));
        CHECK_ACL(aclrtMalloc(
            &opaque->q_buffer, info.B * info.Hk * NATIVE_DIM * sizeof(uint16_t),
            ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMalloc(
            &opaque->k_buffer, info.B * info.Hk * NATIVE_DIM * sizeof(uint16_t),
            ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMalloc(
            &opaque->v_buffer, info.B * info.Hv * NATIVE_DIM * sizeof(uint16_t),
            ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMalloc(
            &opaque->beta_buffer, info.B * info.Hv * sizeof(uint16_t),
            ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMalloc(
            &opaque->lengths_buffer, info.B * sizeof(int32_t),
            ACL_MEM_MALLOC_HUGE_FIRST));
        CHECK_ACL(aclrtMalloc(
            &opaque->indices_buffer, info.B * sizeof(int32_t),
            ACL_MEM_MALLOC_HUGE_FIRST));
        if (opaque->native_workspace_size > 0) {
            CHECK_ACL(aclrtMalloc(
                &opaque->native_workspace_buffer,
                opaque->native_workspace_size, ACL_MEM_MALLOC_HUGE_FIRST));
        }
        workspace_size = opaque->native_workspace_offset
                       + opaque->native_workspace_size;
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
    const void *initial_state_indices,
    const void *final_state_indices,
    void *stream) const {

    if (workspace_size < _workspace_size) {
        return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
    }
    if (_info.gate_dtype != INFINI_DTYPE_F32) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    if (_opaque->native) {
        auto base = static_cast<uint8_t *>(workspace);
        void *q_normalized = _opaque->q_buffer;
        void *k_normalized = _opaque->k_buffer;
        void *v_contiguous = const_cast<void *>(v);
        void *beta_bf16 = _opaque->beta_buffer;
        void *state_staging = base + _opaque->state_offset;
        void *actual_seq_lengths = _opaque->lengths_buffer;
        void *state_indices = _opaque->indices_buffer;
        void *native_workspace = _opaque->native_workspace_buffer;

        RecurrentGdrNativeParams p{};
        p.B = _info.B;
        p.Hk = _info.Hk;
        p.Hv = _info.Hv;
        p.pool_size = _info.pool_size;
        p.initial_indices_i64 = _info.initial_state_indices_dtype == INFINI_DTYPE_I64;
        p.final_indices_i64 = _info.final_state_indices_dtype == INFINI_DTYPE_I64;
        p.q_s0 = _info.q_strides[0];
        p.q_s2 = _info.q_strides[2];
        p.k_s0 = _info.k_strides[0];
        p.k_s2 = _info.k_strides[2];
        p.v_s0 = _info.v_strides[0];
        p.v_s2 = _info.v_strides[2];
        p.beta_s0 = _info.beta_strides[0];
        p.beta_s2 = _info.beta_strides[2];

        CHECK_STATUS(recurrent_gdr_native_preprocess_launch(
            q_normalized, k_normalized, v_contiguous, beta_bf16,
            state_staging, actual_seq_lengths, state_indices,
            q, k, v, beta, initial_state, initial_state_indices,
            final_state_indices, &p, stream));
        // A single batched ACLNN call avoids one launch per active request.
        // Keep an explicit opt-out for CANN regressions or targeted debugging.
        static const bool use_batched = []() {
            const char *value = std::getenv("INFINICORE_ASCEND_GDR_BATCHED");
            return value == nullptr || std::strcmp(value, "1") == 0;
        }();
        if (use_batched) {
            CHECK_ACL(AclSetTensorAddr(
                _opaque->executor, 0, _opaque->q->tensor, q_normalized));
            CHECK_ACL(AclSetTensorAddr(
                _opaque->executor, 1, _opaque->k->tensor, k_normalized));
            CHECK_ACL(AclSetTensorAddr(
                _opaque->executor, 2, _opaque->v->tensor, v_contiguous));
            CHECK_ACL(AclSetTensorAddr(
                _opaque->executor, 3, _opaque->beta->tensor, beta_bf16));
            CHECK_ACL(AclSetTensorAddr(
                _opaque->executor, 4, _opaque->state->tensor, initial_state));
            CHECK_ACL(AclSetTensorAddr(
                _opaque->executor, 5, _opaque->actual_seq_lengths->tensor,
                actual_seq_lengths));
            CHECK_ACL(AclSetTensorAddr(
                _opaque->executor, 6, _opaque->state_indices->tensor,
                state_indices));
            CHECK_ACL(AclSetTensorAddr(
                _opaque->executor, 7, _opaque->g->tensor,
                const_cast<void *>(g)));
            CHECK_ACL(AclSetTensorAddr(
                _opaque->executor, 8, _opaque->out->tensor, out));
            CHECK_ACL(aclnnRecurrentGatedDeltaRule(
                native_workspace, _opaque->native_workspace_size,
                _opaque->executor, stream));
            return INFINI_STATUS_SUCCESS;
        }

        const int64_t Hk = static_cast<int64_t>(_info.Hk);
        const int64_t Hv = static_cast<int64_t>(_info.Hv);
        const int64_t D = static_cast<int64_t>(NATIVE_DIM);
        const int64_t pool = static_cast<int64_t>(_info.pool_size);
        auto add_bytes = [](const void *ptr, size_t bytes) -> void * {
            return const_cast<uint8_t *>(static_cast<const uint8_t *>(ptr)) + bytes;
        };
        for (size_t request = 0; request < _info.B; ++request) {
            void *q_data = add_bytes(
                q_normalized, request * _info.Hk * NATIVE_DIM * sizeof(uint16_t));
            void *k_data = add_bytes(
                k_normalized, request * _info.Hk * NATIVE_DIM * sizeof(uint16_t));
            void *v_data = add_bytes(
                v_contiguous, request * _info.Hv * NATIVE_DIM * sizeof(uint16_t));
            void *beta_data = add_bytes(
                beta_bf16, request * _info.Hv * sizeof(uint16_t));
            void *length_data = add_bytes(
                actual_seq_lengths, request * sizeof(int32_t));
            void *index_data = add_bytes(state_indices, request * sizeof(int32_t));
            void *gate_data = add_bytes(
                g, request * _info.Hv * sizeof(float));
            void *out_data = add_bytes(
                out, request * _info.Hv * NATIVE_DIM * sizeof(uint16_t));

            std::vector<void *> addresses = {
                q_data, k_data, v_data, beta_data, initial_state,
                length_data, index_data, gate_data, out_data};
            size_t cache_index = 0;
            while (cache_index < _opaque->cached_addresses.size()
                   && _opaque->cached_addresses[cache_index] != addresses) {
                ++cache_index;
            }
            if (cache_index < _opaque->cached_addresses.size()) {
                uint64_t cached_workspace_size = _opaque->cached_workspace_sizes[cache_index];
                if (cached_workspace_size > _opaque->native_workspace_size) {
                    return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
                }
                CHECK_ACL(aclnnRecurrentGatedDeltaRule(
                    native_workspace, cached_workspace_size,
                    _opaque->cached_executors[cache_index], stream));
                continue;
            }

            auto *q_desc = new aclnnTensorDescriptor(
                ACL_BF16, {1, Hk, D}, {Hk * D, D, 1}, q_data);
            auto *k_desc = new aclnnTensorDescriptor(
                ACL_BF16, {1, Hk, D}, {Hk * D, D, 1}, k_data);
            auto *v_desc = new aclnnTensorDescriptor(
                ACL_BF16, {1, Hv, D}, {Hv * D, D, 1}, v_data);
            auto *beta_desc = new aclnnTensorDescriptor(
                ACL_BF16, {1, Hv}, {Hv, 1}, beta_data);
            auto *state_desc = new aclnnTensorDescriptor(
                ACL_BF16, {pool, Hv, D, D},
                {Hv * D * D, D * D, D, 1}, initial_state);
            auto *lengths_desc = new aclnnTensorDescriptor(
                ACL_INT32, {1}, {1}, length_data);
            auto *indices_desc = new aclnnTensorDescriptor(
                ACL_INT32, {1}, {1}, index_data);
            auto *gate_desc = new aclnnTensorDescriptor(
                ACL_FLOAT, {1, Hv}, {Hv, 1}, gate_data);
            auto *out_desc = new aclnnTensorDescriptor(
                ACL_BF16, {1, Hv, D}, {Hv * D, D, 1}, out_data);
            aclnnTensorDescriptor *request_descs[] = {
                q_desc, k_desc, v_desc, beta_desc, state_desc,
                lengths_desc, indices_desc, gate_desc, out_desc};
            uint64_t call_workspace_size = 0;
            aclOpExecutor *call_executor = nullptr;
            CHECK_ACL(aclnnRecurrentGatedDeltaRuleGetWorkspaceSize(
                q_desc->tensor, k_desc->tensor, v_desc->tensor,
                beta_desc->tensor, state_desc->tensor, lengths_desc->tensor,
                indices_desc->tensor, gate_desc->tensor, nullptr, nullptr,
                1.0f / std::sqrt(static_cast<float>(NATIVE_DIM)),
                out_desc->tensor, &call_workspace_size, &call_executor));
            if (call_workspace_size > _opaque->native_workspace_size) {
                for (auto *descriptor : request_descs) {
                    delete descriptor;
                }
                return INFINI_STATUS_INSUFFICIENT_WORKSPACE;
            }
            CHECK_ACL(aclSetAclOpExecutorRepeatable(call_executor));
            _opaque->cached_addresses.push_back(std::move(addresses));
            _opaque->cached_descriptors.push_back({q_desc, k_desc, v_desc, beta_desc, state_desc,
                                                   lengths_desc, indices_desc, gate_desc, out_desc});
            _opaque->cached_executors.push_back(call_executor);
            _opaque->cached_workspace_sizes.push_back(call_workspace_size);
            CHECK_ACL(aclnnRecurrentGatedDeltaRule(
                native_workspace, call_workspace_size, call_executor, stream));
        }
        return INFINI_STATUS_SUCCESS;
    }

    GatedDeltaRuleAscendParams p{};
    p.data_dtype = static_cast<int32_t>(_info.data_dtype);
    p.gate_dtype = static_cast<int32_t>(_info.gate_dtype);
    p.use_qk_l2norm = _info.use_qk_l2norm;
    p.has_initial_indices = _info.has_initial_state_indices;
    p.initial_indices_i64 = _info.initial_state_indices_dtype == INFINI_DTYPE_I64;
    p.has_final_indices = _info.has_final_state_indices;
    p.final_indices_i64 = _info.final_state_indices_dtype == INFINI_DTYPE_I64;
    p.B = _info.B;
    p.T = _info.T;
    p.total_tokens = _info.T;
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
        nullptr, initial_state_indices, final_state_indices, &p, stream);
}

} // namespace op::recurrent_gated_delta_rule::ascend
