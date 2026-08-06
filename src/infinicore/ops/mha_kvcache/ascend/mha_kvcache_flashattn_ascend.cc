#if defined(ENABLE_ASCEND_FLASH_ATTN)

#include "infinicore/context/context.hpp"
#include "infinicore/ops/mha_kvcache.hpp"
#include "native/ascend/workspace_pool_.h"

#include <acl/acl.h>
#include <aclnnop/aclnn_fused_infer_attention_score.h>
#include <aclnnop/aclnn_fused_infer_attention_score_v4.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace infinicore::op::mha_kvcache_impl::flashattn_ascend {

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
            "[mha_kvcache/ascend] Unsupported dtype for aclTensor");
    }
}

static aclIntArray *
host_vector_to_acl_int_array(const std::vector<int64_t> &vec) {
    return aclCreateIntArray(vec.data(), vec.size());
}

struct PlannedMeta {
    graph::GraphTensor out, q, k_cache, v_cache, seqlens_k, block_table;
    std::optional<graph::GraphTensor> alibi_slopes;
    float scale;
    Tensor out_work, q_work, k_work, v_work, block_table_work;
};

static Tensor persistent_contiguous_work_tensor(const Tensor &tensor) {
    if (tensor->is_contiguous()) {
        return Tensor(tensor);
    }
    return Tensor::empty(
        tensor->shape(), tensor->dtype(), tensor->device());
}

static std::vector<int64_t>
get_actual_seq_lengths_k(const PlannedMeta *p, int64_t batch_size) {
    if (const auto *bound = graph::lookup_task_update_host_int_array(p->seqlens_k)) {
        if (bound->size() != static_cast<size_t>(batch_size)) {
            throw std::runtime_error(
                "[mha_kvcache/ascend] bound actualSeqLengthsKv size does not "
                "match the captured batch size");
        }
        return *bound;
    }

    if (graph::is_task_updating()) {
        throw std::runtime_error(
            "[mha_kvcache/ascend] missing host actualSeqLengthsKv binding "
            "during graph task update");
    }

    auto seqlens_k_shape = p->seqlens_k->shape();
    if (seqlens_k_shape.size() != 1
        || seqlens_k_shape[0] != static_cast<size_t>(batch_size)) {
        throw std::runtime_error(
            "[mha_kvcache/ascend] seqlens_k must be a 1D tensor whose length "
            "matches the batch size");
    }

    std::vector<int32_t> seqlens_k_host(batch_size);
    auto copy_ret = aclrtMemcpy(
        seqlens_k_host.data(),
        batch_size * sizeof(int32_t),
        reinterpret_cast<const void *>(p->seqlens_k->data()),
        batch_size * sizeof(int32_t),
        ACL_MEMCPY_DEVICE_TO_HOST);
    if (copy_ret != ACL_SUCCESS) {
        throw std::runtime_error(
            std::string(
                "[mha_kvcache/ascend] copy seqlens_k to host failed: ")
            + std::to_string(copy_ret));
    }

    std::vector<int64_t> actual_seq_k_vec;
    actual_seq_k_vec.reserve(batch_size);
    for (int64_t i = 0; i < batch_size; ++i) {
        actual_seq_k_vec.push_back(seqlens_k_host[i]);
    }
    return actual_seq_k_vec;
}

void *plan(Tensor out, const Tensor &q, const Tensor &k_cache,
           const Tensor &v_cache, const Tensor &seqlens_k,
           const Tensor &block_table, std::optional<Tensor> alibi_slopes,
           float scale) {
    return new PlannedMeta{graph::GraphTensor(out),
                           graph::GraphTensor(q),
                           graph::GraphTensor(k_cache),
                           graph::GraphTensor(v_cache),
                           graph::GraphTensor(seqlens_k),
                           graph::GraphTensor(block_table),
                           alibi_slopes ? std::optional<graph::GraphTensor>(
                               graph::GraphTensor(*alibi_slopes))
                                        : std::nullopt,
                           scale,
                           persistent_contiguous_work_tensor(out),
                           persistent_contiguous_work_tensor(q),
                           persistent_contiguous_work_tensor(k_cache),
                           persistent_contiguous_work_tensor(v_cache),
                           persistent_contiguous_work_tensor(block_table)};
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    infinicore::context::setDevice(p->q->device());

    if (p->alibi_slopes.has_value()) {
        throw std::runtime_error("[mha_kvcache/ascend] ALiBi not supported by "
                                 "aclnnFusedInferAttentionScore");
    }

    // q/out are BSND [batch, 1, num_heads, head_size] in InfiniCore. For
    // decode S=1, the same memory can be described to FIA as BNSD
    // [batch, num_heads, 1, head_size].
    auto q_shape = p->q->shape();
    auto k_shape = p->k_cache->shape();
    auto v_shape = p->v_cache->shape();

    if (q_shape.size() != 4 || k_shape.size() != 4 || v_shape.size() != 4) {
        throw std::runtime_error("[mha_kvcache/ascend] flash attention expects q "
                                 "and KV cache to be 4D tensors");
    }

    const int64_t batch_size = q_shape[0];
    const int64_t num_heads = q_shape[2];
    const int64_t head_size = q_shape[3];
    const int64_t num_blocks = k_shape[0];
    const int64_t block_size_val = k_shape[1];
    const int64_t num_kv_heads = k_shape[2];
    const int64_t v_head_size = v_shape[3];

    if (k_shape[3] != static_cast<size_t>(head_size)) {
        throw std::runtime_error(
            "[mha_kvcache/ascend] k_cache head_size does not match q head_size");
    }
    if (v_shape[0] != k_shape[0] || v_shape[1] != k_shape[1] || v_shape[2] != k_shape[2]) {
        throw std::runtime_error(
            "[mha_kvcache/ascend] k_cache and v_cache shapes are incompatible");
    }

    const bool task_updating = graph::is_task_updating();
    if (!task_updating) {
        if (!p->q->is_contiguous()) {
            p->q_work->copy_from(p->q);
        }
        if (!p->k_cache->is_contiguous()) {
            p->k_work->copy_from(p->k_cache);
        }
        if (!p->v_cache->is_contiguous()) {
            p->v_work->copy_from(p->v_cache);
        }
        if (!p->block_table->is_contiguous()) {
            p->block_table_work->copy_from(p->block_table);
        }
    }
    Tensor q_work = p->q_work;
    Tensor k_work = p->k_work;
    Tensor v_work = p->v_work;
    Tensor bt_work = p->block_table_work;
    Tensor out_work = p->out_work;

    aclDataType q_dtype = to_acl_dtype(q_work->dtype());

    std::vector<int64_t> actual_seq_q_vec(batch_size,
                                          1); // decode: always 1 query token
    auto actual_seq_k_vec = get_actual_seq_lengths_k(p, batch_size);

    // BNSD [batch, num_heads, 1, head_size], viewed from BSND memory.
    std::vector<int64_t> q_dims = {batch_size, num_heads, 1, head_size};
    std::vector<int64_t> q_strides = {num_heads * head_size, head_size, head_size,
                                      1};
    aclTensor *query_acl = aclCreateTensor(
        q_dims.data(), q_dims.size(), q_dtype, q_strides.data(), 0, ACL_FORMAT_ND,
        q_dims.data(), q_dims.size(),
        const_cast<void *>(reinterpret_cast<const void *>(q_work->data())));

    // The physical BnBsND cache is contiguous in N and D, so expose it to FIA
    // as BnBsH without copying.
    std::vector<int64_t> k_dims = {num_blocks, block_size_val,
                                   num_kv_heads * head_size};
    std::vector<int64_t> k_strides = {block_size_val * num_kv_heads * head_size,
                                      num_kv_heads * head_size, 1};
    aclTensor *k_acl_tensor = aclCreateTensor(
        k_dims.data(), k_dims.size(), q_dtype, k_strides.data(), 0, ACL_FORMAT_ND,
        k_dims.data(), k_dims.size(),
        const_cast<void *>(reinterpret_cast<const void *>(k_work->data())));
    aclTensorList *key_acl = aclCreateTensorList(&k_acl_tensor, 1);

    std::vector<int64_t> v_dims = {num_blocks, block_size_val,
                                   num_kv_heads * v_head_size};
    std::vector<int64_t> v_strides = {block_size_val * num_kv_heads * v_head_size,
                                      num_kv_heads * v_head_size, 1};
    aclTensor *v_acl_tensor = aclCreateTensor(
        v_dims.data(), v_dims.size(), q_dtype, v_strides.data(), 0, ACL_FORMAT_ND,
        v_dims.data(), v_dims.size(),
        const_cast<void *>(reinterpret_cast<const void *>(v_work->data())));
    aclTensorList *value_acl = aclCreateTensorList(&v_acl_tensor, 1);

    // Block table: [batch, max_blocks_per_seq] INT32 on device
    auto bt_shape = bt_work->shape();
    std::vector<int64_t> bt_dims = {bt_shape[0], bt_shape[1]};
    std::vector<int64_t> bt_strides = {bt_shape[1], 1};
    aclTensor *block_table_acl = aclCreateTensor(
        bt_dims.data(), bt_dims.size(), ACL_INT32, bt_strides.data(), 0,
        ACL_FORMAT_ND, bt_dims.data(), bt_dims.size(),
        const_cast<void *>(reinterpret_cast<const void *>(bt_work->data())));

    // BNSD [batch, num_heads, 1, head_size], viewed from BSND memory.
    std::vector<int64_t> out_dims = {batch_size, num_heads, 1, head_size};
    std::vector<int64_t> out_strides = {num_heads * head_size, head_size,
                                        head_size, 1};
    aclDataType out_dtype = to_acl_dtype(out_work->dtype());
    aclTensor *out_acl = aclCreateTensor(
        out_dims.data(), out_dims.size(), out_dtype, out_strides.data(), 0,
        ACL_FORMAT_ND, out_dims.data(), out_dims.size(),
        const_cast<void *>(reinterpret_cast<const void *>(out_work->data())));

    aclIntArray *seqlens_q_acl = host_vector_to_acl_int_array(actual_seq_q_vec);
    aclIntArray *seqlens_k_acl = host_vector_to_acl_int_array(actual_seq_k_vec);

    // Call CANN API with Paged Attention
    // inputLayout="BNSD": query/out=[B,N,S,D], KV cache is BnBsH for paged
    // attention. sparse_mode=0: no mask needed for decode (Q_S=1,
    // IncreFlashAttention branch)
    uint64_t workspace_size = 0;
    aclOpExecutor *executor = nullptr;

    aclnnStatus ret = aclnnFusedInferAttentionScoreV4GetWorkspaceSize(
        query_acl, key_acl, value_acl,
        nullptr, // pseShift
        nullptr, // atten_mask (not needed for decode)
        seqlens_q_acl, seqlens_k_acl, nullptr, nullptr, nullptr, nullptr, nullptr,
        nullptr, nullptr,
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
        const_cast<char *>("BNSD"), num_kv_heads,
        0,              // sparse_mode=0 (no mask for decode)
        0,              // innerPrecise
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
        aclDestroyTensor(block_table_acl);
        aclDestroyTensor(out_acl);
        aclDestroyIntArray(seqlens_q_acl);
        aclDestroyIntArray(seqlens_k_acl);
        const char *err_msg = aclGetRecentErrMsg();
        throw std::runtime_error(
            std::string(
                "[mha_kvcache/ascend] "
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

    const bool capture_task_group = graph::is_task_group_capturing();
    if (capture_task_group) {
        graph::begin_task_group_capture();
    }
    const bool update_task_group = graph::is_task_updating();
    if (update_task_group) {
        graph::begin_task_update();
    }
    ret = aclnnFusedInferAttentionScoreV4(workspace, workspace_size, executor,
                                          stream);
    if (update_task_group) {
        graph::end_task_update();
    }
    if (capture_task_group) {
        graph::end_task_group_capture();
    }

    // Release aclTensor/aclTensorList/aclIntArray resources
    aclDestroyTensor(query_acl);
    aclDestroyTensorList(key_acl);
    aclDestroyTensorList(value_acl);
    aclDestroyTensor(block_table_acl);
    aclDestroyTensor(out_acl);
    aclDestroyIntArray(seqlens_q_acl);
    aclDestroyIntArray(seqlens_k_acl);

    if (ret != 0) {
        const char *err_msg = aclGetRecentErrMsg();
        throw std::runtime_error(
            std::string(
                "[mha_kvcache/ascend] aclnnFusedInferAttentionScoreV4 failed: ")
            + std::to_string(ret) + ", msg: " + (err_msg ? err_msg : "(null)"));
    }

    // Copy back if out was not contiguous
    if (!task_updating && !p->out->is_contiguous()) {
        p->out->copy_from(out_work);
    }
}

void cleanup(void **planned_meta_ptr) {
    auto *p = *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    delete p;
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    MhaKVCache::plan_dispatcher().registerDevice(Device::Type::ASCEND, &plan);
    MhaKVCache::run_dispatcher().registerDevice(Device::Type::ASCEND, &run);
    MhaKVCache::cleanup_dispatcher().registerDevice(Device::Type::ASCEND,
                                                    &cleanup);
    return true;
}();

} // namespace infinicore::op::mha_kvcache_impl::flashattn_ascend
#endif
