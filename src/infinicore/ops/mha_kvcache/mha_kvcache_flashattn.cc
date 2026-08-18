#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
#include "infinicore/ops/mha_kvcache.hpp"

#ifdef ENABLE_INFINIOPS_LINKED_FLASH_ATTN_WITH_KVCACHE
#include "../infiniops_impl.hpp"

#include "base/flash_attn_with_kvcache.h"

#include <cstdint>
#include <vector>
#endif

#include "infinicore/adaptor/flash_attention_adaptor.hpp"

#include <stdexcept>

#ifdef ENABLE_FLASH_ATTN
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
#include <c10/cuda/CUDAGuard.h>
#endif
#endif

#if defined(ENABLE_METAX_API)
#define INFINICORE_FLASH_OP(name) ::name
#else
#define INFINICORE_FLASH_OP(name) flash::name
#endif

namespace infinicore::op::mha_kvcache_impl::flashattn {
namespace {

#ifdef ENABLE_INFINIOPS_LINKED_FLASH_ATTN_WITH_KVCACHE
using TensorMeta = ::infinicore::op::infiniops::TensorMeta;

bool canUseInfiniOps(const Tensor &out,
                     const Tensor &q,
                     const Tensor &k_cache,
                     const Tensor &v_cache,
                     const Tensor &seqlens_k,
                     const Tensor &block_table,
                     const std::optional<Tensor> &alibi_slopes) {
    const auto dtype = q->dtype();
    const auto device_type = out->device().getType();
    if ((device_type != Device::Type::NVIDIA && device_type != Device::Type::METAX)
        || q->ndim() != 4
        || out->ndim() != 4
        || k_cache->ndim() != 4
        || v_cache->ndim() != 4
        || q->size(1) != 1
        || k_cache->shape() != v_cache->shape()
        || out->shape() != q->shape()
        || (dtype != DataType::F16 && dtype != DataType::BF16)
        || out->dtype() != dtype
        || k_cache->dtype() != dtype
        || v_cache->dtype() != dtype
        || q->size(0) == 0
        || q->size(2) == 0
        || k_cache->size(1) == 0
        || k_cache->size(2) == 0
        || q->size(2) % k_cache->size(2) != 0
        || q->size(3) == 0
        || q->size(3) > 256
        || q->size(3) % 8 != 0
        || q->size(3) != k_cache->size(3)
        || q->stride(3) != 1
        || out->stride(3) != 1
        || k_cache->stride(3) != 1
        || v_cache->stride(3) != 1
        || seqlens_k->ndim() != 1
        || seqlens_k->size(0) != q->size(0)
        || seqlens_k->dtype() != DataType::I32
        || !seqlens_k->is_contiguous()
        || block_table->ndim() != 2
        || block_table->size(0) != q->size(0)
        || block_table->dtype() != DataType::I32
        || !block_table->is_contiguous()
        || k_cache->size(1) % 256 != 0) {
        return false;
    }

    if (alibi_slopes
        && ((alibi_slopes.value()->ndim() != 1
             && alibi_slopes.value()->ndim() != 2)
            || alibi_slopes.value()->dtype() != DataType::F32
            || !alibi_slopes.value()->is_contiguous()
            || alibi_slopes.value()->device() != out->device()
            || (alibi_slopes.value()->ndim() == 1
                && alibi_slopes.value()->size(0) != q->size(2))
            || (alibi_slopes.value()->ndim() == 2
                && (alibi_slopes.value()->size(0) != q->size(0)
                    || alibi_slopes.value()->size(1) != q->size(2))))) {
        return false;
    }

    return true;
}
#endif

} // namespace

struct PlannedMeta {
    graph::GraphTensor out, q, k_cache, v_cache, seqlens_k, block_table;
    std::optional<graph::GraphTensor> alibi_slopes;
    float scale;
#ifdef ENABLE_INFINIOPS_LINKED_FLASH_ATTN_WITH_KVCACHE
    bool use_infiniops{false};
    std::optional<TensorMeta> infiniops_out, infiniops_q, infiniops_k_cache,
        infiniops_v_cache, infiniops_seqlens_k, infiniops_block_table;
    std::optional<TensorMeta> infiniops_alibi_slopes;
#endif
};

void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k_cache,
           const Tensor &v_cache,
           const Tensor &seqlens_k,
           const Tensor &block_table,
           std::optional<Tensor> alibi_slopes,
           float scale) {
    auto *planned = new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k_cache),
        graph::GraphTensor(v_cache),
        graph::GraphTensor(seqlens_k),
        graph::GraphTensor(block_table),
        alibi_slopes ? std::optional<graph::GraphTensor>(graph::GraphTensor(*alibi_slopes)) : std::nullopt,
        scale};

#ifdef ENABLE_INFINIOPS_LINKED_FLASH_ATTN_WITH_KVCACHE
    planned->use_infiniops = canUseInfiniOps(
        out, q, k_cache, v_cache, seqlens_k, block_table, alibi_slopes);
    if (planned->use_infiniops) {
        planned->infiniops_out.emplace(out);
        planned->infiniops_q.emplace(q);
        planned->infiniops_k_cache.emplace(k_cache);
        planned->infiniops_v_cache.emplace(v_cache);
        planned->infiniops_seqlens_k.emplace(seqlens_k);
        planned->infiniops_block_table.emplace(block_table);
        if (alibi_slopes) {
            planned->infiniops_alibi_slopes.emplace(*alibi_slopes);
        }
    }
#endif

    return planned;
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

#ifdef ENABLE_INFINIOPS_LINKED_FLASH_ATTN_WITH_KVCACHE
    if (p->use_infiniops) {
        infini::ops::Handle handle;
        handle.set_stream(context::getStream());
        infini::ops::Config config;
        config.set_implementation_index(16);

        const std::optional<infini::ops::Tensor> no_tensor;
        const std::optional<infini::ops::Tensor> cache_seqlens{
            p->infiniops_seqlens_k->tensor(p->seqlens_k)};
        const std::optional<infini::ops::Tensor> block_table{
            p->infiniops_block_table->tensor(p->block_table)};
        const std::optional<infini::ops::Tensor> alibi_slopes = p->alibi_slopes
                                                                  ? std::optional<infini::ops::Tensor>{p->infiniops_alibi_slopes->tensor(*p->alibi_slopes)}
                                                                  : std::nullopt;

        infini::ops::FlashAttnWithKvcache::Call(
            handle,
            config,
            p->infiniops_q->tensor(p->q),
            p->infiniops_k_cache->tensor(p->k_cache),
            p->infiniops_v_cache->tensor(p->v_cache),
            no_tensor,
            no_tensor,
            no_tensor,
            no_tensor,
            cache_seqlens,
            no_tensor,
            no_tensor,
            block_table,
            alibi_slopes,
            std::optional<double>{p->scale},
            true,
            std::vector<std::int64_t>{-1, -1},
            0.0,
            true,
            std::int64_t{0},
            false,
            p->infiniops_out->tensor(p->out),
            no_tensor);
        return;
    }
#endif

#if defined(ENABLE_FLASH_ATTN)
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API) || defined(ENABLE_QY_API)
    c10::cuda::CUDAStreamGuard guard(infinicore::adaptor::get_cuda_stream());
#endif

    // Paged KV caches must be contiguous for flash-attn; avoid extra copies for q/metadata when already dense.
    const bool out_need_copy_back = !p->out->is_contiguous();
    Tensor out_work = out_need_copy_back ? p->out->contiguous() : Tensor(p->out);
    auto out_tensor = infinicore::adaptor::to_aten_tensor(out_work);
    auto q = infinicore::adaptor::to_aten_tensor(p->q);
#if defined(ENABLE_NVIDIA_API) || defined(ENABLE_METAX_API)
    auto k_cache = infinicore::adaptor::to_aten_tensor(p->k_cache);
    auto v_cache = infinicore::adaptor::to_aten_tensor(p->v_cache);
#elif defined(ENABLE_QY_API)
    Tensor k_cache_work = p->k_cache->contiguous();
    Tensor v_cache_work = p->v_cache->contiguous();
    auto k_cache = infinicore::adaptor::to_aten_tensor(k_cache_work);
    auto v_cache = infinicore::adaptor::to_aten_tensor(v_cache_work);
#endif
    auto seqlens_k = std::optional<const at::Tensor>(infinicore::adaptor::to_aten_tensor(p->seqlens_k));
    auto block_table = std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(p->block_table));
    auto alibi_slopes = p->alibi_slopes
                          ? std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(*p->alibi_slopes))
                          : std::nullopt;

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
#endif
