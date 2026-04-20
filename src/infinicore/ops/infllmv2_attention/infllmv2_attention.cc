/**
 * InfLLM-V2 attention ops (varlen + kvcache).
 * - With ENABLE_FLASH_ATTN: uses mha_varlen / mha_kvcache (Flash-style) fallback.
 * - With ENABLE_INFLLMV2 + ENABLE_ATEN: calls InfLLM-V2 C++ API (mha_varlen_fwd, mha_fwd_kvcache).
 *   Build InfiniCore with:
 *     xmake f --aten=y --infllmv2=/abs/path/to/libinfllm_v2.so   (recommended)
 *   or:
 *     xmake f --aten=y --infllmv2=/abs/path/to/infllmv2_cuda_impl
 *   or:
 *     xmake f --aten=y --infllmv2=y   (auto-detect under third_party/infllmv2_cuda_impl if you checked it out)
 *   Linking is handled in xmake (adds DT_NEEDED + rpath to the resolved .so).
 */
#include "infinicore/ops/infllmv2_attention.hpp"

#include "../../utils.hpp"

#if defined(ENABLE_INFLLMV2) && defined(ENABLE_ATEN)
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/infllmv2_api.hpp"
#ifdef ENABLE_NVIDIA_API
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#endif
#elif defined(ENABLE_FLASH_ATTN)
#include "infinicore/adaptor/flash_attention_adaptor.hpp"
#include "infinicore/ops/mha_kvcache.hpp"
#include "infinicore/ops/mha_varlen.hpp"
#endif

namespace infinicore::op {

INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(InfllmV2AttentionVarlen);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(InfllmV2AttentionKVCache);
INFINICORE_GRAPH_OP_DISPATCHERS_IMPL(InfllmV2AttentionKVCacheUpdate);

namespace {
void infllmv2_attention_varlen_impl(Tensor out,
                                    const Tensor &q,
                                    const Tensor &k,
                                    const Tensor &v,
                                    const Tensor &cu_seqlens_q,
                                    const Tensor &cu_seqlens_k,
                                    int max_seqlen_q,
                                    int max_seqlen_k,
                                    float scale,
                                    bool causal,
                                    int window_size_left,
                                    int window_size_right) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k, v, cu_seqlens_q, cu_seqlens_k);

#if defined(ENABLE_INFLLMV2) && defined(ENABLE_ATEN)
    // Direct InfLLM-V2 kernels (link against infllmv2_cuda_impl).
#ifdef ENABLE_NVIDIA_API
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
    auto q_at = infinicore::adaptor::to_aten_tensor(q);
    auto k_at = infinicore::adaptor::to_aten_tensor(k);
    auto v_at = infinicore::adaptor::to_aten_tensor(v);
    auto cu_q_at = infinicore::adaptor::to_aten_tensor(cu_seqlens_q);
    auto cu_k_at = infinicore::adaptor::to_aten_tensor(cu_seqlens_k);
    auto out_at = std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(out));

    c10::optional<at::Tensor> seqused_k = c10::nullopt;
    c10::optional<const at::Tensor> leftpad_k = c10::nullopt;
    c10::optional<at::Tensor> block_table = c10::nullopt;
    c10::optional<at::Tensor> alibi_slopes = c10::nullopt;
    c10::optional<at::Generator> gen_ = c10::nullopt;
    c10::optional<at::Tensor> blockmask_ = c10::nullopt;

    mha_varlen_fwd(
        q_at,
        k_at,
        v_at,
        out_at,
        cu_q_at,
        cu_k_at,
        seqused_k,
        leftpad_k,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        0.0f,
        scale,
        false,
        causal,
        window_size_left,
        window_size_right,
        0.0f,
        false,
        gen_,
        blockmask_);
    return;

#elif defined(ENABLE_FLASH_ATTN)
    // Fallback: FlashAttention-based varlen op (same kernel family as InfLLM-V2).
    auto dummy_block_table = infinicore::Tensor::zeros(
        {cu_seqlens_q->shape()[0] - 1, 1},
        cu_seqlens_q->dtype(),
        cu_seqlens_q->device());
    (void)causal;
    (void)window_size_left;
    (void)window_size_right;
    auto tmp = infinicore::op::mha_varlen(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        dummy_block_table,
        max_seqlen_q,
        max_seqlen_k,
        std::nullopt,
        scale);
    out->copy_(tmp);
    return;
#else
    (void)k;
    (void)v;
    (void)cu_seqlens_q;
    (void)cu_seqlens_k;
    (void)max_seqlen_q;
    (void)max_seqlen_k;
    (void)scale;
    (void)causal;
    (void)window_size_left;
    (void)window_size_right;
    throw std::runtime_error(
        "InfLLM-V2 varlen attention requires ENABLE_INFLLMV2+ENABLE_ATEN or ENABLE_FLASH_ATTN build");
#endif
}

void infllmv2_attention_kvcache_impl(Tensor out,
                                     const Tensor &q,
                                     const Tensor &k_cache,
                                     const Tensor &v_cache,
                                     const Tensor &cache_lens,
                                     float scale,
                                     bool causal,
                                     int window_size_left,
                                     int window_size_right) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k_cache, v_cache, cache_lens);

#if defined(ENABLE_INFLLMV2) && defined(ENABLE_ATEN)
#ifdef ENABLE_NVIDIA_API
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
    auto q_at = infinicore::adaptor::to_aten_tensor(q);
    auto kcache_at = infinicore::adaptor::to_aten_tensor(k_cache);
    auto vcache_at = infinicore::adaptor::to_aten_tensor(v_cache);
    auto seqlens_k_at = std::optional<const at::Tensor>(infinicore::adaptor::to_aten_tensor(cache_lens));
    auto out_at = std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(out));

    c10::optional<const at::Tensor> k_new = c10::nullopt;
    c10::optional<const at::Tensor> v_new = c10::nullopt;
    c10::optional<const at::Tensor> rotary_cos = c10::nullopt;
    c10::optional<const at::Tensor> rotary_sin = c10::nullopt;
    c10::optional<const at::Tensor> cache_batch_idx = c10::nullopt;
    c10::optional<const at::Tensor> leftpad_k = c10::nullopt;
    c10::optional<at::Tensor> block_table = c10::nullopt;
    c10::optional<at::Tensor> alibi_slopes = c10::nullopt;
    c10::optional<at::Tensor> blockmask_ = c10::nullopt;

    // Let FlashAttn/InfLLM-v2 allocate output internally. Passing an explicit out_ tensor
    // can interact badly with internal q reshapes in the seqlen_q==1 GQA fast path.
    c10::optional<at::Tensor> out_kernel_opt = c10::nullopt;
    auto outs = mha_fwd_kvcache(
        q_at,
        kcache_at,
        vcache_at,
        k_new,
        v_new,
        seqlens_k_at,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        leftpad_k,
        block_table,
        alibi_slopes,
        out_kernel_opt,
        scale,
        causal,
        window_size_left,
        window_size_right,
        0.0f,
        false,
        0,
        blockmask_);
    out_at.value().copy_(outs[0]);
    return;

#elif defined(ENABLE_FLASH_ATTN)
    (void)causal;
    (void)window_size_left;
    (void)window_size_right;
    auto device = q->device();
    auto bs = cache_lens->shape()[0];
    auto one = infinicore::Tensor::ones({bs, 1}, cache_lens->dtype(), device);
    auto block_table = one;
    auto seqlens_k = cache_lens;
    auto tmp = infinicore::op::mha_kvcache(
        q,
        k_cache,
        v_cache,
        seqlens_k,
        block_table,
        std::nullopt,
        scale);
    out->copy_(tmp);
    return;
#else
    (void)k_cache;
    (void)v_cache;
    (void)cache_lens;
    (void)scale;
    (void)causal;
    (void)window_size_left;
    (void)window_size_right;
    throw std::runtime_error(
        "InfLLM-V2 kvcache attention requires ENABLE_INFLLMV2+ENABLE_ATEN or ENABLE_FLASH_ATTN build");
#endif
}

void infllmv2_attention_kvcache_update_impl(Tensor out,
                                            const Tensor &q,
                                            const Tensor &k_cache,
                                            const Tensor &v_cache,
                                            const Tensor &k_new,
                                            const Tensor &v_new,
                                            const Tensor &cache_lens,
                                            float scale,
                                            bool causal,
                                            int window_size_left,
                                            int window_size_right) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k_cache, v_cache, k_new, v_new, cache_lens);

#if defined(ENABLE_INFLLMV2) && defined(ENABLE_ATEN)
#ifdef ENABLE_NVIDIA_API
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
    auto q_at = infinicore::adaptor::to_aten_tensor(q);
    auto kcache_at = infinicore::adaptor::to_aten_tensor(k_cache);
    auto vcache_at = infinicore::adaptor::to_aten_tensor(v_cache);
    auto knew_at = infinicore::adaptor::to_aten_tensor(k_new);
    auto vnew_at = infinicore::adaptor::to_aten_tensor(v_new);
    auto seqlens_k_at = std::optional<const at::Tensor>(infinicore::adaptor::to_aten_tensor(cache_lens));
    auto out_at = std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(out));

    c10::optional<const at::Tensor> k_new_opt = std::optional<const at::Tensor>(knew_at);
    c10::optional<const at::Tensor> v_new_opt = std::optional<const at::Tensor>(vnew_at);
    c10::optional<const at::Tensor> rotary_cos = c10::nullopt;
    c10::optional<const at::Tensor> rotary_sin = c10::nullopt;
    c10::optional<const at::Tensor> cache_batch_idx = c10::nullopt;
    c10::optional<const at::Tensor> leftpad_k = c10::nullopt;
    c10::optional<at::Tensor> block_table = c10::nullopt;
    c10::optional<at::Tensor> alibi_slopes = c10::nullopt;
    c10::optional<at::Tensor> blockmask_ = c10::nullopt;

    c10::optional<at::Tensor> out_kernel_opt = c10::nullopt;
    auto outs = mha_fwd_kvcache(
        q_at,
        kcache_at,
        vcache_at,
        k_new_opt,
        v_new_opt,
        seqlens_k_at,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        leftpad_k,
        block_table,
        alibi_slopes,
        out_kernel_opt,
        scale,
        causal,
        window_size_left,
        window_size_right,
        0.0f,
        false,
        0,
        blockmask_);
    out_at.value().copy_(outs[0]);
    return;

#elif defined(ENABLE_FLASH_ATTN)
    (void)k_new;
    (void)v_new;
    // FlashAttn adaptor path currently doesn't support in-place cache update in this wrapper.
    // Fall back to normal kvcache (expects cache already updated by caller).
    infllmv2_attention_kvcache_impl(out, q, k_cache, v_cache, cache_lens, scale, causal, window_size_left, window_size_right);
    return;
#else
    (void)k_cache;
    (void)v_cache;
    (void)k_new;
    (void)v_new;
    (void)cache_lens;
    (void)scale;
    (void)causal;
    (void)window_size_left;
    (void)window_size_right;
    throw std::runtime_error(
        "InfLLM-V2 kvcache_update attention requires ENABLE_INFLLMV2+ENABLE_ATEN build");
#endif
}

} // namespace

InfllmV2AttentionVarlen::InfllmV2AttentionVarlen(Tensor out,
                                                 const Tensor &q,
                                                 const Tensor &k,
                                                 const Tensor &v,
                                                 const Tensor &cu_seqlens_q,
                                                 const Tensor &cu_seqlens_k,
                                                 int max_seqlen_q,
                                                 int max_seqlen_k,
                                                 float scale,
                                                 bool causal,
                                                 int window_size_left,
                                                 int window_size_right) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k, v, cu_seqlens_q, cu_seqlens_k);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(),
                                 out, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale, causal, window_size_left, window_size_right);
}

void InfllmV2AttentionVarlen::execute(Tensor out,
                                      const Tensor &q,
                                      const Tensor &k,
                                      const Tensor &v,
                                      const Tensor &cu_seqlens_q,
                                      const Tensor &cu_seqlens_k,
                                      int max_seqlen_q,
                                      int max_seqlen_k,
                                      float scale,
                                      bool causal,
                                      int window_size_left,
                                      int window_size_right) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        InfllmV2AttentionVarlen,
        out, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale, causal, window_size_left, window_size_right);
}

// NOTE: we implement run/cleanup with explicit types to keep compilation simple.
namespace {
struct VarlenPlanned {
    graph::GraphTensor out, q, k, v, cu_q, cu_k;
    int max_q, max_k;
    float scale;
    bool causal;
    int wleft, wright;
};
void run_varlen_typed(void *planned_meta) {
    auto *p = reinterpret_cast<VarlenPlanned *>(planned_meta);
    infllmv2_attention_varlen_impl(p->out, p->q, p->k, p->v, p->cu_q, p->cu_k, p->max_q, p->max_k, p->scale, p->causal, p->wleft, p->wright);
}

void cleanup_varlen(void **planned_meta_ptr) {
    delete *reinterpret_cast<VarlenPlanned **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

void *plan_varlen_typed(Tensor out,
                        const Tensor &q,
                        const Tensor &k,
                        const Tensor &v,
                        const Tensor &cu_seqlens_q,
                        const Tensor &cu_seqlens_k,
                        int max_seqlen_q,
                        int max_seqlen_k,
                        float scale,
                        bool causal,
                        int window_size_left,
                        int window_size_right) {
    return new VarlenPlanned{graph::GraphTensor(out), graph::GraphTensor(q), graph::GraphTensor(k), graph::GraphTensor(v),
                             graph::GraphTensor(cu_seqlens_q), graph::GraphTensor(cu_seqlens_k),
                             max_seqlen_q, max_seqlen_k, scale, causal, window_size_left, window_size_right};
}

static bool registered_infllmv2_attention_varlen = []() {
    InfllmV2AttentionVarlen::plan_dispatcher().registerAll(&plan_varlen_typed, false);
    InfllmV2AttentionVarlen::run_dispatcher().registerAll(&run_varlen_typed, false);
    InfllmV2AttentionVarlen::cleanup_dispatcher().registerAll(&cleanup_varlen, false);
    return true;
}();
} // namespace

void infllmv2_attention_varlen_(Tensor out,
                                const Tensor &q,
                                const Tensor &k,
                                const Tensor &v,
                                const Tensor &cu_seqlens_q,
                                const Tensor &cu_seqlens_k,
                                int max_seqlen_q,
                                int max_seqlen_k,
                                float scale,
                                bool causal,
                                int window_size_left,
                                int window_size_right) {
    InfllmV2AttentionVarlen::execute(
        out, q, k, v, cu_seqlens_q, cu_seqlens_k,
        max_seqlen_q, max_seqlen_k,
        scale, causal,
        window_size_left, window_size_right);
}

Tensor infllmv2_attention_varlen(const Tensor &q,
                                 const Tensor &k,
                                 const Tensor &v,
                                 const Tensor &cu_seqlens_q,
                                 const Tensor &cu_seqlens_k,
                                 int max_seqlen_q,
                                 int max_seqlen_k,
                                 float scale,
                                 bool causal,
                                 int window_size_left,
                                 int window_size_right) {
    auto out = Tensor::empty(q->shape(), q->dtype(), q->device());
    infllmv2_attention_varlen_(out, q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, scale, causal, window_size_left, window_size_right);
    return out;
}

InfllmV2AttentionKVCache::InfllmV2AttentionKVCache(Tensor out,
                                                   const Tensor &q,
                                                   const Tensor &k_cache,
                                                   const Tensor &v_cache,
                                                   const Tensor &cache_lens,
                                                   float scale,
                                                   bool causal,
                                                   int window_size_left,
                                                   int window_size_right) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k_cache, v_cache, cache_lens);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(),
                                 out, q, k_cache, v_cache, cache_lens, scale, causal, window_size_left, window_size_right);
}

void InfllmV2AttentionKVCache::execute(Tensor out,
                                       const Tensor &q,
                                       const Tensor &k_cache,
                                       const Tensor &v_cache,
                                       const Tensor &cache_lens,
                                       float scale,
                                       bool causal,
                                       int window_size_left,
                                       int window_size_right) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        InfllmV2AttentionKVCache,
        out, q, k_cache, v_cache, cache_lens, scale, causal, window_size_left, window_size_right);
}

namespace {
struct KVCachePlanned {
    graph::GraphTensor out, q, k, v, lens;
    float scale;
    bool causal;
    int wleft, wright;
};
void *plan_kvcache(Tensor out,
                   const Tensor &q,
                   const Tensor &k_cache,
                   const Tensor &v_cache,
                   const Tensor &cache_lens,
                   float scale,
                   bool causal,
                   int window_size_left,
                   int window_size_right) {
    return new KVCachePlanned{graph::GraphTensor(out), graph::GraphTensor(q), graph::GraphTensor(k_cache),
                              graph::GraphTensor(v_cache), graph::GraphTensor(cache_lens),
                              scale, causal, window_size_left, window_size_right};
}

void run_kvcache(void *planned_meta) {
    auto *p = reinterpret_cast<KVCachePlanned *>(planned_meta);
    infllmv2_attention_kvcache_impl(p->out, p->q, p->k, p->v, p->lens, p->scale, p->causal, p->wleft, p->wright);
}

void cleanup_kvcache(void **planned_meta_ptr) {
    delete *reinterpret_cast<KVCachePlanned **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered_infllmv2_attention_kvcache = []() {
    InfllmV2AttentionKVCache::plan_dispatcher().registerAll(&plan_kvcache, false);
    InfllmV2AttentionKVCache::run_dispatcher().registerAll(&run_kvcache, false);
    InfllmV2AttentionKVCache::cleanup_dispatcher().registerAll(&cleanup_kvcache, false);
    return true;
}();
} // namespace

void infllmv2_attention_kvcache_(Tensor out,
                                 const Tensor &q,
                                 const Tensor &k_cache,
                                 const Tensor &v_cache,
                                 const Tensor &cache_lens,
                                 float scale,
                                 bool causal,
                                 int window_size_left,
                                 int window_size_right) {
    InfllmV2AttentionKVCache::execute(
        out, q, k_cache, v_cache, cache_lens,
        scale, causal,
        window_size_left, window_size_right);
}

Tensor infllmv2_attention_kvcache(const Tensor &q,
                                  const Tensor &k_cache,
                                  const Tensor &v_cache,
                                  const Tensor &cache_lens,
                                  float scale,
                                  bool causal,
                                  int window_size_left,
                                  int window_size_right) {
    auto out = Tensor::empty(q->shape(), q->dtype(), q->device());
    infllmv2_attention_kvcache_(out, q, k_cache, v_cache, cache_lens, scale, causal, window_size_left, window_size_right);
    return out;
}

InfllmV2AttentionKVCacheUpdate::InfllmV2AttentionKVCacheUpdate(Tensor out,
                                                               const Tensor &q,
                                                               const Tensor &k_cache,
                                                               const Tensor &v_cache,
                                                               const Tensor &k_new,
                                                               const Tensor &v_new,
                                                               const Tensor &cache_lens,
                                                               float scale,
                                                               bool causal,
                                                               int window_size_left,
                                                               int window_size_right) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k_cache, v_cache, k_new, v_new, cache_lens);
    INFINICORE_GRAPH_OP_DISPATCH(out->device().getType(),
                                 out, q, k_cache, v_cache, k_new, v_new, cache_lens, scale, causal, window_size_left, window_size_right);
}

void InfllmV2AttentionKVCacheUpdate::execute(Tensor out,
                                             const Tensor &q,
                                             const Tensor &k_cache,
                                             const Tensor &v_cache,
                                             const Tensor &k_new,
                                             const Tensor &v_new,
                                             const Tensor &cache_lens,
                                             float scale,
                                             bool causal,
                                             int window_size_left,
                                             int window_size_right) {
    INFINICORE_GRAPH_OP_RECORD_OR_RUN(
        InfllmV2AttentionKVCacheUpdate,
        out, q, k_cache, v_cache, k_new, v_new, cache_lens, scale, causal, window_size_left, window_size_right);
}

namespace {
struct KVCacheUpdatePlanned {
    graph::GraphTensor out, q, k, v, knew, vnew, lens;
    float scale;
    bool causal;
    int wleft, wright;
};
void *plan_kvcache_update(Tensor out,
                          const Tensor &q,
                          const Tensor &k_cache,
                          const Tensor &v_cache,
                          const Tensor &k_new,
                          const Tensor &v_new,
                          const Tensor &cache_lens,
                          float scale,
                          bool causal,
                          int window_size_left,
                          int window_size_right) {
    return new KVCacheUpdatePlanned{graph::GraphTensor(out), graph::GraphTensor(q),
                                    graph::GraphTensor(k_cache), graph::GraphTensor(v_cache),
                                    graph::GraphTensor(k_new), graph::GraphTensor(v_new),
                                    graph::GraphTensor(cache_lens),
                                    scale, causal, window_size_left, window_size_right};
}

void run_kvcache_update(void *planned_meta) {
    auto *p = reinterpret_cast<KVCacheUpdatePlanned *>(planned_meta);
    infllmv2_attention_kvcache_update_impl(p->out, p->q, p->k, p->v, p->knew, p->vnew, p->lens, p->scale, p->causal, p->wleft, p->wright);
}

void cleanup_kvcache_update(void **planned_meta_ptr) {
    delete *reinterpret_cast<KVCacheUpdatePlanned **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered_infllmv2_attention_kvcache_update = []() {
    InfllmV2AttentionKVCacheUpdate::plan_dispatcher().registerAll(&plan_kvcache_update, false);
    InfllmV2AttentionKVCacheUpdate::run_dispatcher().registerAll(&run_kvcache_update, false);
    InfllmV2AttentionKVCacheUpdate::cleanup_dispatcher().registerAll(&cleanup_kvcache_update, false);
    return true;
}();
} // namespace

void infllmv2_attention_kvcache_update_(Tensor out,
                                        const Tensor &q,
                                        const Tensor &k_cache,
                                        const Tensor &v_cache,
                                        const Tensor &k_new,
                                        const Tensor &v_new,
                                        const Tensor &cache_lens,
                                        float scale,
                                        bool causal,
                                        int window_size_left,
                                        int window_size_right) {
    InfllmV2AttentionKVCacheUpdate::execute(
        out, q, k_cache, v_cache, k_new, v_new, cache_lens,
        scale, causal,
        window_size_left, window_size_right);
}

Tensor infllmv2_attention_kvcache_update(const Tensor &q,
                                         const Tensor &k_cache,
                                         const Tensor &v_cache,
                                         const Tensor &k_new,
                                         const Tensor &v_new,
                                         const Tensor &cache_lens,
                                         float scale,
                                         bool causal,
                                         int window_size_left,
                                         int window_size_right) {
    auto out = Tensor::empty(q->shape(), q->dtype(), q->device());
    infllmv2_attention_kvcache_update_(out, q, k_cache, v_cache, k_new, v_new, cache_lens, scale, causal, window_size_left, window_size_right);
    return out;
}

} // namespace infinicore::op
