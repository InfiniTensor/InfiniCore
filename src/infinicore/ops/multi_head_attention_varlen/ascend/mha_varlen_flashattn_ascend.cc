#if defined(ENABLE_ASCEND_FLASH_ATTN)

#include "infinicore/context/context.hpp"
#include "infinicore/ops/mha_varlen.hpp"
#include "native/ascend/workspace_pool_.h"

#include <acl/acl.h>
#include <aclnnop/aclnn_fused_infer_attention_score.h>
#include <aclnnop/aclnn_fused_infer_attention_score_v4.h>

#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace infinicore::op::mha_varlen_impl::flashattn_ascend {

static aclDataType to_acl_dtype(DataType dtype) {
    switch (dtype) {
    case DataType::F16:
        return ACL_FLOAT16;
    case DataType::BF16:
        return ACL_BF16;
    case DataType::F32:
        return ACL_FLOAT;
    case DataType::I32:
        return ACL_INT32;
    case DataType::I64:
        return ACL_INT64;
    default:
        throw std::runtime_error(
            "[mha_varlen/ascend] Unsupported dtype for aclTensor");
    }
}

static aclIntArray *
host_vector_to_acl_int_array(const std::vector<int64_t> &vec) {
    return aclCreateIntArray(vec.data(), vec.size());
}

struct PlannedMeta {
    graph::GraphTensor out, q, k, v, cum_seqlens_q, cum_seqlens_k;
    std::optional<graph::GraphTensor> block_table;
    int max_seqlen_q, max_seqlen_k;
    std::optional<graph::GraphTensor> alibi_slopes;
    float scale;
    // Each layer owns only its aclTensor descriptor. The immutable device
    // buffer is shared by every prefill layer on the same Ascend device.
    aclTensor *mask_acl = nullptr;
};

static constexpr int64_t kCausalMaskSide = 2048;
static constexpr size_t kCausalMaskBytes = kCausalMaskSide * kCausalMaskSide * sizeof(uint8_t);

static const std::vector<uint8_t> &causal_mask_host_template() {
    static const std::vector<uint8_t> mask = [] {
        std::vector<uint8_t> value(kCausalMaskBytes, 0);
        for (int64_t i = 0; i < kCausalMaskSide; ++i) {
            for (int64_t j = i + 1; j < kCausalMaskSide; ++j) {
                value[i * kCausalMaskSide + j] = 1;
            }
        }
        return value;
    }();
    return mask;
}

namespace {

class CausalMaskCache {
public:
    void *get_or_create(int32_t device_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = masks_.find(device_id);
        if (found != masks_.end()) {
            return found->second;
        }

        int32_t current_device = -1;
        auto ret = aclrtGetDevice(&current_device);
        if (ret != ACL_SUCCESS || current_device != device_id) {
            throw std::runtime_error(
                "[mha_varlen/ascend] current device does not match causal "
                "mask device");
        }

        void *device_mask = nullptr;
        ret = aclrtMalloc(&device_mask, kCausalMaskBytes,
                          ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret == ACL_SUCCESS) {
            const auto &host_mask = causal_mask_host_template();
            ret = aclrtMemcpy(device_mask, kCausalMaskBytes, host_mask.data(),
                              kCausalMaskBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        }
        if (ret != ACL_SUCCESS) {
            if (device_mask != nullptr) {
                aclrtFree(device_mask);
            }
            throw std::runtime_error(
                "[mha_varlen/ascend] initialize shared causal mask failed: "
                + std::to_string(ret));
        }

        masks_.emplace(device_id, device_mask);
        return device_mask;
    }

    ~CausalMaskCache() {
        int32_t previous_device = -1;
        if (aclrtGetDevice(&previous_device) != ACL_SUCCESS) {
            // ACL Runtime may already be finalized during static teardown.
            return;
        }
        for (const auto &[device_id, device_mask] : masks_) {
            if (aclrtSetDevice(device_id) == ACL_SUCCESS) {
                aclrtFree(device_mask);
            }
        }
        aclrtSetDevice(previous_device);
    }

private:
    std::mutex mutex_;
    std::unordered_map<int32_t, void *> masks_;
};

CausalMaskCache &causal_mask_cache() {
    static CausalMaskCache cache;
    return cache;
}

} // namespace

static aclTensor *create_causal_mask(PlannedMeta *p) {
    if (p->mask_acl != nullptr) {
        return p->mask_acl;
    }

    const int32_t device_id = static_cast<int32_t>(p->q->device().getIndex());
    void *mask_dev = causal_mask_cache().get_or_create(device_id);
    std::vector<int64_t> mask_dims = {kCausalMaskSide, kCausalMaskSide};
    std::vector<int64_t> mask_strides = {kCausalMaskSide, 1};
    p->mask_acl = aclCreateTensor(
        mask_dims.data(), mask_dims.size(), ACL_UINT8, mask_strides.data(), 0,
        ACL_FORMAT_ND, mask_dims.data(), mask_dims.size(), mask_dev);
    return p->mask_acl;
}

void *plan(Tensor out, const Tensor &q, const Tensor &k, const Tensor &v,
           const Tensor &cum_seqlens_q, const Tensor &cum_seqlens_k,
           std::optional<Tensor> block_table, int max_seqlen_q,
           int max_seqlen_k, std::optional<Tensor> alibi_slopes, float scale) {
    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(v),
        graph::GraphTensor(cum_seqlens_q),
        graph::GraphTensor(cum_seqlens_k),
        block_table
            ? std::optional<graph::GraphTensor>(graph::GraphTensor(*block_table))
            : std::nullopt,
        max_seqlen_q,
        max_seqlen_k,
        alibi_slopes
            ? std::optional<graph::GraphTensor>(graph::GraphTensor(*alibi_slopes))
            : std::nullopt,
        scale};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    infinicore::context::setDevice(p->q->device());

    if (p->alibi_slopes.has_value()) {
        throw std::runtime_error("[mha_varlen/ascend] ALiBi not supported by "
                                 "aclnnFusedInferAttentionScore");
    }

    auto cu_q_tensor = p->cum_seqlens_q;
    auto cu_k_tensor = p->cum_seqlens_k;
    auto cu_q_shape = cu_q_tensor->shape();
    auto cu_k_shape = cu_k_tensor->shape();
    int64_t cu_q_len = cu_q_shape[0];
    int64_t cu_k_len = cu_k_shape[0];
    int64_t batch_size = cu_q_len - 1;

    std::vector<int32_t> cu_q_host(cu_q_len);
    std::vector<int32_t> cu_k_host(cu_k_len);
    aclrtMemcpy(cu_q_host.data(), cu_q_len * sizeof(int32_t),
                reinterpret_cast<const void *>(cu_q_tensor->data()),
                cu_q_len * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtMemcpy(cu_k_host.data(), cu_k_len * sizeof(int32_t),
                reinterpret_cast<const void *>(cu_k_tensor->data()),
                cu_k_len * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);

    std::vector<int64_t> actual_seq_q_vec;
    std::vector<int64_t> actual_seq_k_vec;
    for (int64_t i = 0; i < batch_size; ++i) {
        int64_t k_len = cu_k_host[i + 1] - cu_k_host[i];
        // TND query uses cumulative sequence lengths. Paged KV cache keeps
        // actualSeqLengthsKv in per-batch mode.
        actual_seq_q_vec.push_back(cu_q_host[i + 1]);
        actual_seq_k_vec.push_back(k_len);
    }

    auto q_shape = p->q->shape();
    auto k_shape = p->k->shape();
    const int64_t num_heads = q_shape[1];
    const int64_t head_size = q_shape[2];
    const int64_t block_size_val = k_shape[1];
    const int64_t num_kv_heads = k_shape[2];
    const int64_t num_blocks = k_shape[0];

    Tensor q_work = p->q->is_contiguous() ? Tensor(p->q) : p->q->contiguous();
    Tensor k_work = p->k->is_contiguous() ? Tensor(p->k) : p->k->contiguous();
    Tensor v_work = p->v->is_contiguous() ? Tensor(p->v) : p->v->contiguous();
    Tensor out_work = p->out->is_contiguous() ? Tensor(p->out) : p->out->contiguous();

    aclDataType q_dtype = to_acl_dtype(q_work->dtype());

    // Query is already contiguous TND [total_q, num_heads, head_size].
    std::vector<int64_t> q_dims = {static_cast<int64_t>(q_shape[0]), num_heads,
                                   head_size};
    std::vector<int64_t> q_strides = {num_heads * head_size, head_size, 1};
    aclTensor *query_acl = aclCreateTensor(
        q_dims.data(), q_dims.size(), q_dtype, q_strides.data(), 0, ACL_FORMAT_ND,
        q_dims.data(), q_dims.size(),
        const_cast<void *>(reinterpret_cast<const void *>(q_work->data())));

    // The physical BnBsND cache is contiguous in N and D, so expose it to FIA
    // as BnBsH without copying.
    std::vector<int64_t> k_dims = {num_blocks, block_size_val,
                                   num_kv_heads * head_size};
    std::vector<int64_t> k_strides = {num_kv_heads * block_size_val * head_size,
                                      num_kv_heads * head_size, 1};
    aclTensor *k_acl_tensor = aclCreateTensor(
        k_dims.data(), k_dims.size(), q_dtype, k_strides.data(), 0, ACL_FORMAT_ND,
        k_dims.data(), k_dims.size(),
        const_cast<void *>(reinterpret_cast<const void *>(k_work->data())));
    aclTensorList *key_acl = aclCreateTensorList(&k_acl_tensor, 1);

    // Value uses the same contiguous BnBsH representation.
    std::vector<int64_t> v_dims = {num_blocks, block_size_val,
                                   num_kv_heads * head_size};
    std::vector<int64_t> v_strides = {num_kv_heads * block_size_val * head_size,
                                      num_kv_heads * head_size, 1};
    aclTensor *v_acl_tensor = aclCreateTensor(
        v_dims.data(), v_dims.size(), q_dtype, v_strides.data(), 0, ACL_FORMAT_ND,
        v_dims.data(), v_dims.size(),
        const_cast<void *>(reinterpret_cast<const void *>(v_work->data())));
    aclTensorList *value_acl = aclCreateTensorList(&v_acl_tensor, 1);

    // Block table: [batch, max_blocks_per_seq] INT32 on device
    aclTensor *block_table_acl = nullptr;
    if (p->block_table.has_value()) {
        auto &bt = p->block_table.value();
        Tensor bt_work = bt->is_contiguous() ? Tensor(bt) : bt->contiguous();
        auto bt_shape = bt_work->shape();
        std::vector<int64_t> bt_dims = {bt_shape[0], bt_shape[1]};
        std::vector<int64_t> bt_strides = {bt_shape[1], 1};
        block_table_acl = aclCreateTensor(
            bt_dims.data(), bt_dims.size(), ACL_INT32, bt_strides.data(), 0,
            ACL_FORMAT_ND, bt_dims.data(), bt_dims.size(),
            const_cast<void *>(reinterpret_cast<const void *>(bt_work->data())));
    }

    // FIA writes directly to contiguous TND output.
    auto out_shape = out_work->shape();
    std::vector<int64_t> out_dims = {static_cast<int64_t>(out_shape[0]),
                                     num_heads, head_size};
    std::vector<int64_t> out_strides = {num_heads * head_size, head_size, 1};
    aclDataType out_dtype = to_acl_dtype(out_work->dtype());
    aclTensor *out_acl = aclCreateTensor(
        out_dims.data(), out_dims.size(), out_dtype, out_strides.data(), 0,
        ACL_FORMAT_ND, out_dims.data(), out_dims.size(),
        const_cast<void *>(reinterpret_cast<const void *>(out_work->data())));

    aclIntArray *actual_seq_q_acl = host_vector_to_acl_int_array(actual_seq_q_vec);
    aclIntArray *actual_seq_k_acl = host_vector_to_acl_int_array(actual_seq_k_vec);

    int64_t sparse_mode = 3; // rightDownCausal
    aclTensor *atten_mask_acl = create_causal_mask(p);

    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus ret = aclnnFusedInferAttentionScoreV4GetWorkspaceSize(
        query_acl, key_acl, value_acl,
        nullptr, // pseShift
        atten_mask_acl, actual_seq_q_acl, actual_seq_k_acl, nullptr, nullptr,
        nullptr, nullptr, nullptr, nullptr, nullptr,
        block_table_acl, // blockTable - Paged Attention
        nullptr,         // queryPaddingSize
        nullptr,         // kvPaddingSize
        nullptr,         // keyAntiquantScale
        nullptr,         // keyAntiquantOffset
        nullptr,         // valueAntiquantScale
        nullptr,         // valueAntiquantOffset
        nullptr,         // keySharedPrefix
        nullptr,         // valueSharedPrefix
        nullptr,         // actualSharedPrefixLen
        nullptr,         // queryRope
        nullptr,         // keyRope
        nullptr,         // keyRopeAntiquantScale
        nullptr,         // dequantScaleQuery
        nullptr,         // learnableSink
        num_heads, static_cast<double>(p->scale), 2147483647, 2147483647,
        const_cast<char *>("TND"), num_kv_heads, sparse_mode,
        0,              // innerPrecise: high precision TND path
        block_size_val, // blockSize - Paged Attention block size
        0,              // antiquantMode
        false,
        0, // keyAntiquantMode
        0, // valueAntiquantMode
        0, // queryQuantMode
        out_acl, nullptr, &workspace_size, &executor);

    if (ret != 0) {
        aclDestroyTensor(query_acl);
        aclDestroyTensorList(key_acl);
        aclDestroyTensorList(value_acl);
        if (block_table_acl) {
            aclDestroyTensor(block_table_acl);
        }
        aclDestroyTensor(out_acl);
        aclDestroyIntArray(actual_seq_q_acl);
        aclDestroyIntArray(actual_seq_k_acl);
        const char *err_msg = aclGetRecentErrMsg();
        throw std::runtime_error(
            std::string(
                "[mha_varlen/ascend] "
                "aclnnFusedInferAttentionScoreV4GetWorkspaceSize failed: ")
            + std::to_string(ret) + ", msg: " + (err_msg ? err_msg : "(null)"));
    }

    aclrtStream stream = static_cast<aclrtStream>(infinicore::context::getStream());
    void *workspace = nullptr;
    if (workspace_size > 0) {
        workspace = infini::ops::ascend::GetWorkspacePool()
                        .Ensure(stream, workspace_size, "fia")
                        .buf;
    }

    ret = aclnnFusedInferAttentionScoreV4(workspace, workspace_size, executor,
                                          stream);

    // Release aclTensor/aclTensorList/aclIntArray resources
    aclDestroyTensor(query_acl);
    aclDestroyTensorList(key_acl);
    aclDestroyTensorList(value_acl);
    if (block_table_acl) {
        aclDestroyTensor(block_table_acl);
    }
    aclDestroyTensor(out_acl);
    aclDestroyIntArray(actual_seq_q_acl);
    aclDestroyIntArray(actual_seq_k_acl);

    if (ret != 0) {
        const char *err_msg = aclGetRecentErrMsg();
        throw std::runtime_error(
            std::string(
                "[mha_varlen/ascend] aclnnFusedInferAttentionScoreV4 failed: ")
            + std::to_string(ret) + ", msg: " + (err_msg ? err_msg : "(null)"));
    }

    // Copy back if out was not contiguous
    if (!p->out->is_contiguous()) {
        p->out->copy_from(out_work);
    }
}

void cleanup(void **planned_meta_ptr) {
    auto *p = *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    if (p->mask_acl) {
        aclDestroyTensor(p->mask_acl);
    }
    delete p;
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    MultiheadAttentionVarlen::plan_dispatcher().registerDevice(
        Device::Type::ASCEND, &plan);
    MultiheadAttentionVarlen::run_dispatcher().registerDevice(
        Device::Type::ASCEND, &run);
    MultiheadAttentionVarlen::cleanup_dispatcher().registerDevice(
        Device::Type::ASCEND, &cleanup);
    return true;
}();

} // namespace infinicore::op::mha_varlen_impl::flashattn_ascend
#endif
