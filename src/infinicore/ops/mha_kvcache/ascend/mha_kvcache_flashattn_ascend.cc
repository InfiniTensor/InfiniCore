#if defined(ENABLE_ASCEND_FLASH_ATTN)

#include <mutex>
#include <cstdlib>
#include <deque>
#include <map>
#include <memory>
#include <unordered_map>
#include <algorithm>
#include "infinicore/context/context.hpp"
#include "infinicore/ops/mha_kvcache.hpp"
#include "native/ascend/workspace_pool_.h"

#include <acl/acl.h>
#include <aclnnop/aclnn_fused_infer_attention_score.h>
#include <aclnnop/aclnn_fused_infer_attention_score_v5.h>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace infinicore::op {
common::OpDispatcher<graph::graph_replay_schema> &
mha_kvcache_graph_replay_dispatcher();
}

namespace infinicore::op::mha_kvcache_impl::flashattn_ascend {

static void check_task_api(aclError status, const char *api) {
    if (status != ACL_SUCCESS) {
        const char *message = aclGetRecentErrMsg();
        throw std::runtime_error(
            std::string("[mha_kvcache/ascend] ") + api + " failed: "
            + std::to_string(status) + ", msg: "
            + (message ? message : "(null)"));
    }
}

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


// A replay shares shape/length-dependent ACLNN preparation across decoder
// layers on one stream. Each launch explicitly rebinds all layer addresses.
// Keep retired executors alive until the previous graph completes on that
// stream, because their tiling/launch state can still be referenced by it.
struct ReplayExecutor {
    aclOpExecutor *executor = nullptr;
    aclTensor *query = nullptr, *block_table = nullptr, *output = nullptr;
    aclTensorList *key = nullptr, *value = nullptr;
    aclIntArray *seq_q = nullptr, *seq_k = nullptr;
    uint64_t workspace_size = 0;
    std::vector<int64_t> signature;
    ~ReplayExecutor() {
        if (executor) aclDestroyAclOpExecutor(executor);
        if (query) aclDestroyTensor(query);
        if (key) aclDestroyTensorList(key);
        if (value) aclDestroyTensorList(value);
        if (block_table) aclDestroyTensor(block_table);
        if (output) aclDestroyTensor(output);
        if (seq_q) aclDestroyIntArray(seq_q);
        if (seq_k) aclDestroyIntArray(seq_k);
    }
};
struct ReplayExecutorCache {
    int32_t device;
    aclrtStream stream;
    std::unique_ptr<ReplayExecutor> current;
    struct Retired { aclrtEvent event; std::unique_ptr<ReplayExecutor> executor; };
    std::deque<Retired> retired;
    ReplayExecutorCache(int32_t d, aclrtStream s) : device(d), stream(s) {}
    void retire_current() {
        if (current) {
            aclrtEvent event = nullptr;
            check_task_api(aclrtCreateEventWithFlag(&event, ACL_EVENT_CAPTURE_STREAM_PROGRESS), "create executor retirement event");
            auto status = aclrtRecordEvent(event, stream);
            if (status != ACL_SUCCESS) { aclrtDestroyEvent(event); check_task_api(status, "record executor retirement event"); }
            retired.push_back({event, std::move(current)});
        }
        while (!retired.empty()) {
            aclrtEventRecordedStatus status;
            check_task_api(aclrtQueryEventStatus(retired.front().event, &status), "query executor retirement event");
            if (status != ACL_EVENT_RECORDED_STATUS_COMPLETE) break;
            aclrtDestroyEvent(retired.front().event);
            retired.pop_front();
        }
        if (retired.size() > 16) {
            check_task_api(aclrtSynchronizeStream(stream), "drain retired executors");
            for (auto &x : retired) aclrtDestroyEvent(x.event);
            retired.clear();
        }
    }
    ~ReplayExecutorCache() {
        int32_t previous = -1;
        if (aclrtGetDevice(&previous) != ACL_SUCCESS) return;
        if (aclrtSetDevice(device) != ACL_SUCCESS) return;
        aclrtSynchronizeStream(stream);
        current.reset();
        for (auto &x : retired) aclrtDestroyEvent(x.event);
        retired.clear();
        if (previous != device) aclrtSetDevice(previous);
    }
};
static std::shared_ptr<ReplayExecutorCache> replay_executor_cache() {
    const char *flag = std::getenv("INFINICORE_FIA_REUSE_EXECUTOR");
    if (!flag || std::strcmp(flag, "1") != 0) return nullptr;
    int32_t device = -1;
    check_task_api(aclrtGetDevice(&device), "get replay executor device");
    auto stream = static_cast<aclrtStream>(context::getStream());
    using Key = std::pair<int32_t, uintptr_t>;
    static thread_local std::map<Key, std::weak_ptr<ReplayExecutorCache>> caches;
    auto &weak = caches[{device, reinterpret_cast<uintptr_t>(stream)}];
    auto cache = weak.lock();
    if (!cache) { cache = std::make_shared<ReplayExecutorCache>(device, stream); weak = cache; }
    return cache;
}

// One update stream per device/main stream; only ACL graph task updates and
// their completion records are submitted here. Model kernels stay on the
// original stream and consume each layer only after its captured event wait.
struct AsyncUpdateStream {
    aclrtStream stream = nullptr;
    int32_t device = -1;
    AsyncUpdateStream() {
        check_task_api(aclrtGetDevice(&device), "get async update device");
        check_task_api(aclrtCreateStream(&stream), "create async update stream");
    }
    ~AsyncUpdateStream() {
        int32_t previous = -1;
        if (aclrtGetDevice(&previous) != ACL_SUCCESS) return;
        if (aclrtSetDevice(device) != ACL_SUCCESS) return;
        aclrtSynchronizeStream(stream);
        aclrtDestroyStream(stream);
        if (previous != device) aclrtSetDevice(previous);
    }
};
static std::shared_ptr<AsyncUpdateStream> async_update_stream() {
    const char *flag = std::getenv("INFINICORE_ASCEND_ASYNC_GRAPH_UPDATE");
    if (!flag || std::strcmp(flag, "1") != 0 || !context::isGraphRecording()) return nullptr;
    int32_t device = -1;
    check_task_api(aclrtGetDevice(&device), "get async update device");
    using Key = std::pair<int32_t, uintptr_t>;
    static thread_local std::map<Key, std::weak_ptr<AsyncUpdateStream>> streams;
    auto &weak = streams[{device, reinterpret_cast<uintptr_t>(context::getStream())}];
    auto stream = weak.lock();
    if (!stream) { stream = std::make_shared<AsyncUpdateStream>(); weak = stream; }
    return stream;
}

struct PlannedMeta {
    graph::GraphTensor out, q, k_cache, v_cache, seqlens_k, block_table;
    std::optional<graph::GraphTensor> alibi_slopes;
    float scale;
    Tensor out_work, q_work, k_work, v_work, block_table_work;
    void *workspace;
    bool graph_planned;
    std::vector<int64_t> initial_seq_lengths_k;
    aclrtTaskGrp task_group = nullptr;
    bool capturing_task_group = false;
    bool updating_task_group = false;
    std::shared_ptr<ReplayExecutorCache> replay_cache;
    std::shared_ptr<AsyncUpdateStream> update_stream;
    aclrtEvent update_ready = nullptr;
    ~PlannedMeta() {
        if (update_ready) {
            // Destructors cannot throw while unwinding a failed device call.
            aclrtSynchronizeStream(static_cast<aclrtStream>(context::getStream()));
            aclrtSynchronizeStream(update_stream->stream);
            aclrtDestroyEvent(update_ready);
        }
    }
};

static Tensor persistent_contiguous_work_tensor(const Tensor &tensor) {
    if (tensor->is_contiguous()) {
        return Tensor(tensor);
    }
    return Tensor::empty(tensor->shape(), tensor->dtype(), tensor->device());
}

static std::vector<int64_t>
get_actual_seq_lengths_k(const PlannedMeta *p, int64_t batch_size) {
    const auto shape = p->seqlens_k->shape();
    if (p->seqlens_k->dtype() != DataType::I32 || shape.size() != 1
        || shape[0] != static_cast<size_t>(batch_size)) {
        throw std::runtime_error(
            "[mha_kvcache/ascend] seqlens_k must be a 1D int32 tensor whose "
            "length matches the batch size");
    }

    if (const auto *bound = graph::lookup_bound_host_int_array(p->seqlens_k)) {
        if (bound->size() != static_cast<size_t>(batch_size)) {
            throw std::runtime_error(
                "[mha_kvcache/ascend] bound actualSeqLengthsKv size does not "
                "match the batch size");
        }
        return *bound;
    }

    if (p->updating_task_group) {
        throw std::runtime_error(
            "[mha_kvcache/ascend] missing host actualSeqLengthsKv binding "
            "during graph replay update");
    }
    if (!p->initial_seq_lengths_k.empty()) {
        return p->initial_seq_lengths_k;
    }

    // Graph planning may perform this one host copy. Device graph capture and
    // replay only use the cached or explicitly bound value.
    std::vector<int32_t> host(batch_size);
    auto copy_ret = aclrtMemcpy(
        host.data(), batch_size * sizeof(int32_t),
        reinterpret_cast<const void *>(p->seqlens_k->data()),
        batch_size * sizeof(int32_t), ACL_MEMCPY_DEVICE_TO_HOST);
    if (copy_ret != ACL_SUCCESS) {
        throw std::runtime_error(
            std::string("[mha_kvcache/ascend] copy seqlens_k to host failed: ")
            + std::to_string(copy_ret));
    }

    std::vector<int64_t> result;
    result.reserve(batch_size);
    for (int64_t i = 0; i < batch_size; ++i) {
        result.push_back(host[i]);
    }
    return result;
}

void *plan(Tensor out, const Tensor &q, const Tensor &k_cache,
           const Tensor &v_cache, const Tensor &seqlens_k,
           const Tensor &block_table, std::optional<Tensor> alibi_slopes,
           float scale) {
    auto *p = new PlannedMeta{graph::GraphTensor(out),
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
                              persistent_contiguous_work_tensor(block_table),
                              {},
                              context::isGraphRecording()};
    p->replay_cache = replay_executor_cache();
    p->update_stream = async_update_stream();
    try {
        p->initial_seq_lengths_k = get_actual_seq_lengths_k(
            p, static_cast<int64_t>(q->shape().at(0)));
    } catch (...) {
        delete p;
        throw;
    }
    return p;
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

    const bool task_updating = p->updating_task_group;
    if (!task_updating && !p->q->is_contiguous()) {
        p->q_work->copy_from(p->q);
    }
    if (!task_updating && !p->k_cache->is_contiguous()) {
        p->k_work->copy_from(p->k_cache);
    }
    if (!task_updating && !p->v_cache->is_contiguous()) {
        p->v_work->copy_from(p->v_cache);
    }
    if (!task_updating && !p->block_table->is_contiguous()) {
        p->block_table_work->copy_from(p->block_table);
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

    // Reuse the retained ACLNN metadata too: no temporary tensor/list/array
    // allocation is needed when only layer buffer addresses change.
    const char *fast_hit = std::getenv("INFINICORE_FIA_REUSE_FAST_PATH");
    if (task_updating && p->replay_cache && fast_hit
        && std::strcmp(fast_hit, "1") == 0) {
        const auto bt_shape_fast = bt_work->shape();
        std::vector<int64_t> signature_fast{
            batch_size, num_heads, head_size, num_blocks, block_size_val,
            num_kv_heads, v_head_size, static_cast<int64_t>(q_dtype),
            static_cast<int64_t>(bt_shape_fast[0]), static_cast<int64_t>(bt_shape_fast[1]),
            static_cast<int64_t>(reinterpret_cast<uintptr_t>(bt_work->data()))};
        uint32_t scale_bits_fast;
        std::memcpy(&scale_bits_fast, &p->scale, sizeof(scale_bits_fast));
        signature_fast.push_back(scale_bits_fast);
        signature_fast.insert(signature_fast.end(), actual_seq_k_vec.begin(), actual_seq_k_vec.end());
        const auto &entry = p->replay_cache->current;
        if (entry && entry->signature == signature_fast) {
            auto *executor = entry->executor;
            check_task_api(aclSetInputTensorAddr(executor, 0, entry->query, q_work->data()), "refresh FIA query");
            check_task_api(aclSetDynamicInputTensorAddr(executor, 1, 0, entry->key, k_work->data()), "refresh FIA key");
            check_task_api(aclSetDynamicInputTensorAddr(executor, 2, 0, entry->value, v_work->data()), "refresh FIA value");
            check_task_api(aclSetOutputTensorAddr(executor, 0, entry->output, out_work->data()), "refresh FIA output");
            constexpr uint64_t capacity = 128u * 1024u * 1024u;
            if (entry->workspace_size > capacity || (entry->workspace_size != 0 && !p->workspace)) {
                throw std::runtime_error("[mha_kvcache/ascend] retained FIA workspace was not prepared");
            }
            auto stream = p->update_stream ? p->update_stream->stream
                : static_cast<aclrtStream>(context::getStream());
            check_task_api(aclmdlRICaptureTaskUpdateBegin(stream, p->task_group), "aclmdlRICaptureTaskUpdateBegin");
            const auto ret_fast = aclnnFusedInferAttentionScoreV5(
                p->workspace, entry->workspace_size, executor, stream);
            check_task_api(aclmdlRICaptureTaskUpdateEnd(stream), "aclmdlRICaptureTaskUpdateEnd");
            check_task_api(ret_fast, "retained FIA task update");
            return;
        }
    }

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


    aclnnStatus ret = 0;
    bool retained_descriptors = false;
    ReplayExecutor *reusable = nullptr;
    std::vector<int64_t> signature;
    if (task_updating && p->replay_cache) {
        signature = {batch_size, num_heads, head_size, num_blocks,
                     block_size_val, num_kv_heads, v_head_size,
                     static_cast<int64_t>(q_dtype), bt_dims[0], bt_dims[1],
                     static_cast<int64_t>(reinterpret_cast<uintptr_t>(bt_work->data()))};
        uint32_t scale_bits;
        std::memcpy(&scale_bits, &p->scale, sizeof(scale_bits));
        signature.push_back(scale_bits);
        signature.insert(signature.end(), actual_seq_k_vec.begin(), actual_seq_k_vec.end());
        if (p->replay_cache->current && p->replay_cache->current->signature == signature) {
            reusable = p->replay_cache->current.get();
            executor = reusable->executor;
            workspace_size = reusable->workspace_size;
            check_task_api(aclSetInputTensorAddr(executor, 0, reusable->query, q_work->data()), "refresh FIA query");
            check_task_api(aclSetDynamicInputTensorAddr(executor, 1, 0, reusable->key, k_work->data()), "refresh FIA key");
            check_task_api(aclSetDynamicInputTensorAddr(executor, 2, 0, reusable->value, v_work->data()), "refresh FIA value");
            check_task_api(aclSetOutputTensorAddr(executor, 0, reusable->output, out_work->data()), "refresh FIA output");
        } else {
            p->replay_cache->retire_current();
        }
    }
    if (!reusable) {
        ret = aclnnFusedInferAttentionScoreV5GetWorkspaceSize(
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
        nullptr,         // qStartIdx
        nullptr,         // kvStartIdx
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
        0, // pseType
        out_acl, nullptr, &workspace_size, &executor);

        if (ret == 0 && task_updating && p->replay_cache) {
            auto entry = std::make_unique<ReplayExecutor>();
            entry->executor = executor;
            entry->query = query_acl; entry->key = key_acl; entry->value = value_acl;
            entry->block_table = block_table_acl; entry->output = out_acl;
            entry->seq_q = seqlens_q_acl; entry->seq_k = seqlens_k_acl;
            entry->workspace_size = workspace_size;
            entry->signature = std::move(signature);
            retained_descriptors = true;
            check_task_api(aclSetAclOpExecutorRepeatable(executor), "enable repeatable FIA executor");
            p->replay_cache->current = std::move(entry);
        }
    }

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
                "aclnnFusedInferAttentionScoreV5GetWorkspaceSize failed: ")
            + std::to_string(ret) + ", msg: " + (err_msg ? err_msg : "(null)"));
    }

    aclrtStream stream = task_updating && p->update_stream
        ? p->update_stream->stream
        : static_cast<aclrtStream>(infinicore::context::getStream());
    void *workspace = nullptr;
    if (workspace_size > 0) {
        if (p->graph_planned) {
            // Layers and batch graphs execute serially on this stream. Keep
            // one stable arena instead of retaining 128 MiB for every plan.
            constexpr size_t capacity = 128u * 1024u * 1024u;
            if (workspace_size > capacity) {
                throw std::runtime_error(
                    "[mha_kvcache/ascend] FIA graph workspace exceeds 128 MiB");
            }
            if (!p->workspace) {
                if (p->capturing_task_group || task_updating) {
                    throw std::runtime_error(
                        "[mha_kvcache/ascend] FIA workspace was not prepared "
                        "before graph capture");
                }
                p->workspace = infini::ops::ascend::GetWorkspacePool()
                                   .Ensure(stream, capacity, "fia_graph").buf;
            }
            workspace = p->workspace;
        } else {
            workspace = infini::ops::ascend::GetWorkspacePool()
                            .Ensure(stream, workspace_size, "fia")
                            .buf;
        }
    }

    if (p->capturing_task_group) {
        check_task_api(aclmdlRICaptureTaskGrpBegin(stream),
                       "aclmdlRICaptureTaskGrpBegin");
    }
    if (task_updating) {
        check_task_api(aclmdlRICaptureTaskUpdateBegin(stream, p->task_group),
                       "aclmdlRICaptureTaskUpdateBegin");
    }
    ret = aclnnFusedInferAttentionScoreV5(workspace, workspace_size, executor,
                                          stream);
    if (task_updating) {
        check_task_api(aclmdlRICaptureTaskUpdateEnd(stream),
                       "aclmdlRICaptureTaskUpdateEnd");
    }
    if (p->capturing_task_group) {
        check_task_api(aclmdlRICaptureTaskGrpEnd(stream, &p->task_group),
                       "aclmdlRICaptureTaskGrpEnd");
    }

    if (!retained_descriptors) {
    // Release aclTensor/aclTensorList/aclIntArray resources
    aclDestroyTensor(query_acl);
    aclDestroyTensorList(key_acl);
    aclDestroyTensorList(value_acl);
    aclDestroyTensor(block_table_acl);
    aclDestroyTensor(out_acl);
    aclDestroyIntArray(seqlens_q_acl);
    aclDestroyIntArray(seqlens_k_acl);

    }

    if (ret != 0) {
        const char *err_msg = aclGetRecentErrMsg();
        throw std::runtime_error(
            std::string(
                "[mha_kvcache/ascend] aclnnFusedInferAttentionScoreV5 failed: ")
            + std::to_string(ret) + ", msg: " + (err_msg ? err_msg : "(null)"));
    }

    // Copy back if out was not contiguous
    if (!task_updating && !p->out->is_contiguous()) {
        p->out->copy_from(out_work);
    }
}

void graph_replay(void *planned_meta, graph::GraphReplayStage stage) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    switch (stage) {
    case graph::GraphReplayStage::CAPTURE_BEGIN:
        if (p->update_stream) {
            check_task_api(aclrtCreateEventWithFlag(&p->update_ready, ACL_EVENT_EXTERNAL), "create graph update event");
            auto stream = static_cast<aclrtStream>(context::getStream());
            check_task_api(aclrtStreamWaitEvent(stream, p->update_ready), "capture graph update wait");
            check_task_api(aclrtResetEvent(p->update_ready, stream), "capture graph update reset");
        }
        p->capturing_task_group = true;
        return;
    case graph::GraphReplayStage::CAPTURE_END:
        p->capturing_task_group = false;
        if (p->task_group == nullptr) {
            throw std::runtime_error(
                "[mha_kvcache/ascend] FIA task group was not captured");
        }
        return;
    case graph::GraphReplayStage::UPDATE:
        if (p->task_group == nullptr) {
            throw std::runtime_error(
                "[mha_kvcache/ascend] FIA task group was not captured");
        }
        p->updating_task_group = true;
        try {
            run(p);
        } catch (...) {
            p->updating_task_group = false;
            throw;
        }
        p->updating_task_group = false;
        if (p->update_stream) {
            check_task_api(aclrtRecordEvent(p->update_ready, p->update_stream->stream), "signal graph update ready");
        }
        return;
    case graph::GraphReplayStage::PREPARE_UPDATE:
        // Retire before launch: recording/draining a retirement event on a
        // graph stream already waiting for this update would create a cycle.
        if (p->replay_cache) p->replay_cache->retire_current();
        return;
    case graph::GraphReplayStage::ABORT_UPDATE:
        if (p->update_stream) {
            check_task_api(aclrtRecordEvent(p->update_ready, p->update_stream->stream), "release failed graph update wait");
        }
        return;
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
    mha_kvcache_graph_replay_dispatcher().registerDevice(
        Device::Type::ASCEND, &graph_replay);
    return true;
}();

} // namespace infinicore::op::mha_kvcache_impl::flashattn_ascend
#endif
