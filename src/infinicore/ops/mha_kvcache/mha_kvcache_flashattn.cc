#include "infinicore/ops/mha_kvcache.hpp"

#include "infinicore/adaptor/flash_attention_adaptor.hpp"

#include <stdexcept>

#if defined(ENABLE_FLASH_ATTN) && defined(ENABLE_CAMBRICON_API)
#include <algorithm>
#include <cstdint>
#include <vector>
#endif

#ifdef ENABLE_FLASH_ATTN
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#if defined(ENABLE_CAMBRICON_API)
#include <framework/core/stream_guard.h>
#endif
#endif

#if defined(ENABLE_METAX_API) || defined(ENABLE_CAMBRICON_API)
#define INFINICORE_FLASH_OP(name) ::name
#else
#define INFINICORE_FLASH_OP(name) flash::name
#endif

namespace infinicore::op::mha_kvcache_impl::flashattn {

struct PlannedMeta {
    graph::GraphTensor out, q, k_cache, v_cache, seqlens_k, block_table;
    std::optional<graph::GraphTensor> alibi_slopes;
    float scale;
};

void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k_cache,
           const Tensor &v_cache,
           const Tensor &seqlens_k,
           const Tensor &block_table,
           std::optional<Tensor> alibi_slopes,
           float scale) {
    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k_cache),
        graph::GraphTensor(v_cache),
        graph::GraphTensor(seqlens_k),
        graph::GraphTensor(block_table),
        alibi_slopes ? std::optional<graph::GraphTensor>(graph::GraphTensor(*alibi_slopes)) : std::nullopt,
        scale};
}

void run(void *planned_meta) {
#ifdef ENABLE_FLASH_ATTN
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API) || defined(ENABLE_CAMBRICON_API)
#if defined(ENABLE_CAMBRICON_API)
    torch_mlu::mlu::MLUStreamGuard guard(infinicore::adaptor::get_mlu_stream());
#else
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif
#endif
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

#if defined(ENABLE_CAMBRICON_API)
    const bool out_need_copy_back = !p->out->is_contiguous();
    Tensor out_work_ic = out_need_copy_back ? p->out->contiguous() : Tensor(p->out);
    auto out_work = infinicore::adaptor::to_aten_tensor(out_work_ic);
    auto q = infinicore::adaptor::to_aten_tensor(p->q);
    auto k_cache = infinicore::adaptor::to_aten_tensor(p->k_cache);
    auto v_cache = infinicore::adaptor::to_aten_tensor(p->v_cache);
    auto seqlens_k_tensor = infinicore::adaptor::to_aten_tensor(p->seqlens_k);
    auto block_table_tensor = infinicore::adaptor::to_aten_tensor(p->block_table);
    auto alibi_slopes = p->alibi_slopes
                          ? std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(*p->alibi_slopes))
                          : std::nullopt;

    if (q.dim() != 4 || out_work.dim() != 4) {
        throw std::runtime_error("Cambricon flash-attn KV-cache path expects q/out with shape [batch, seqlen_q, heads, head_dim]");
    }

    const auto batch_size = q.size(0);
    const auto seqlen_q = q.size(1);
    const auto num_heads = q.size(2);
    const auto head_size = q.size(3);

    auto seqlens_k_cpu = seqlens_k_tensor.to(at::kCPU);
    auto seqlens_k_data = seqlens_k_cpu.data_ptr<int32_t>();
    std::vector<int32_t> cu_seqlens_q_host(batch_size + 1, 0);
    std::vector<int32_t> cu_seqlens_k_host(batch_size + 1, 0);
    int32_t max_seqlen_k = 0;
    for (int64_t i = 0; i < batch_size; ++i) {
        cu_seqlens_q_host[i + 1] = cu_seqlens_q_host[i] + static_cast<int32_t>(seqlen_q);
        cu_seqlens_k_host[i + 1] = cu_seqlens_k_host[i] + seqlens_k_data[i];
        max_seqlen_k = std::max(max_seqlen_k, seqlens_k_data[i]);
    }

    auto tensor_options = q.options().dtype(at::kInt);
    auto cu_seqlens_q = at::tensor(cu_seqlens_q_host, tensor_options);
    auto cu_seqlens_k = at::tensor(cu_seqlens_k_host, tensor_options);

    auto q_varlen = q.reshape({batch_size * seqlen_q, num_heads, head_size});
    auto out_varlen = out_work.reshape({batch_size * seqlen_q, num_heads, head_size});
    auto out = std::optional<at::Tensor>(out_varlen);
    std::optional<at::Tensor> seqused_k = std::nullopt;
    std::optional<const at::Tensor> leftpad_k = std::nullopt;
    auto block_table = std::optional<at::Tensor>(block_table_tensor);

    INFINICORE_FLASH_OP(mha_varlen_fwd)
    (
        q_varlen,
        k_cache,
        v_cache,
        out,
        cu_seqlens_q,
        cu_seqlens_k,
        seqused_k,
        leftpad_k,
        block_table,
        alibi_slopes,
        static_cast<int>(seqlen_q),
        static_cast<int>(max_seqlen_k),
        0.0,
        p->scale,
        false,
        seqlen_q > 1,
        -1,
        -1,
        0.0,
        false,
        std::nullopt);

    if (out_need_copy_back) {
        p->out->copy_from(out_work_ic);
    }
    return;
#else
    // Paged KV caches must be contiguous for flash-attn; avoid extra copies for q/metadata when already dense.
    const bool out_need_copy_back = !p->out->is_contiguous();
    Tensor out_work = out_need_copy_back ? p->out->contiguous() : Tensor(p->out);
    auto out_tensor = infinicore::adaptor::to_aten_tensor(out_work);
    auto q = infinicore::adaptor::to_aten_tensor(p->q);
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API)
    auto k_cache = infinicore::adaptor::to_aten_tensor(p->k_cache);
#elif defined(ENABLE_QY_API)
    Tensor k_cache_work = p->k_cache->contiguous();
    Tensor v_cache_work = p->v_cache->contiguous();
    auto k_cache = infinicore::adaptor::to_aten_tensor(k_cache_work);
    auto v_cache = infinicore::adaptor::to_aten_tensor(v_cache_work);
#endif
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API)
    auto v_cache = infinicore::adaptor::to_aten_tensor(p->v_cache);
#endif
    auto alibi_slopes = p->alibi_slopes
                          ? std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(*p->alibi_slopes))
                          : std::nullopt;

    auto seqlens_k = std::optional<const at::Tensor>(infinicore::adaptor::to_aten_tensor(p->seqlens_k));
    auto block_table = std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(p->block_table));

    std::optional<const at::Tensor> k_new = std::nullopt;
    std::optional<const at::Tensor> v_new = std::nullopt;
    std::optional<const at::Tensor> rotary_cos = std::nullopt;
    std::optional<const at::Tensor> rotary_sin = std::nullopt;
    std::optional<const at::Tensor> cache_batch_idx = std::nullopt;
    std::optional<const at::Tensor> leftpad_k = std::nullopt;

    const bool use_dynamic_out = q.dim() == 4 && k_cache.dim() == 4
                              && q.size(1) == 1 && q.size(2) > k_cache.size(2)
                              && q.size(3) % 8 == 0 && !alibi_slopes.has_value();

    auto out = use_dynamic_out ? std::optional<at::Tensor>(std::nullopt)
                               : std::optional<at::Tensor>(out_tensor);

#if defined(ENABLE_METAX_API) && defined(INFINICORE_HPCC_VERSION_MAJOR) && (INFINICORE_HPCC_VERSION_MAJOR >= 3)
    std::optional<at::Tensor> flash_attn_mars_ext = std::nullopt;
#endif

    auto result = INFINICORE_FLASH_OP(mha_fwd_kvcache)(
        q,
        k_cache,
        v_cache,
        k_new,
        v_new,
        seqlens_k,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        leftpad_k,
        block_table,
        alibi_slopes,
        out,
        p->scale,
        true,
        -1,
        -1,
        0.0f,
        false,
        0
#if defined(ENABLE_METAX_API) && defined(INFINICORE_HPCC_VERSION_MAJOR) && (INFINICORE_HPCC_VERSION_MAJOR >= 3)
        ,
        flash_attn_mars_ext
#endif
    );

    if (use_dynamic_out) {
        out_tensor.copy_(result[0]);
    }
    if (out_need_copy_back) {
        p->out->copy_from(out_work);
    }
#endif // ENABLE_CAMBRICON_API
#else
    throw std::runtime_error("FlashAttention is not enabled in this build");
#endif
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(MhaKVCache, &plan, &run, &cleanup);

} // namespace infinicore::op::mha_kvcache_impl::flashattn
