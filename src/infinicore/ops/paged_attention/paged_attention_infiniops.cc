#include "infinicore/ops/paged_attention.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/flash_attn_with_kvcache.h"

#include <cstdint>
#include <optional>
#include <vector>

namespace infinicore::op::paged_attention_impl::infiniop {
void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k_cache,
           const Tensor &v_cache,
           const Tensor &block_tables,
           const Tensor &cache_lens,
           std::optional<Tensor> alibi_slopes,
           float scale);
void run(void *planned_meta);
void cleanup(void **planned_meta_ptr);
} // namespace infinicore::op::paged_attention_impl::infiniop

namespace infinicore::op::paged_attention_impl::infiniops {
namespace {
using TensorMeta = ::infinicore::op::infiniops::TensorMeta;

bool canUseFlashAttention(const Tensor &out,
                          const Tensor &q,
                          const Tensor &k_cache,
                          const Tensor &v_cache,
                          const Tensor &block_tables,
                          const Tensor &cache_lens) {
#ifdef ENABLE_INFINIOPS_LINKED_FLASH_ATTN_WITH_KVCACHE
    const auto dtype = q->dtype();
    return out->device().getType() == Device::Type::NVIDIA
        && q->ndim() == 3
        && out->ndim() == 3
        && k_cache->ndim() == 4
        && v_cache->ndim() == 4
        && block_tables->ndim() == 2
        && cache_lens->ndim() == 1
        && (dtype == DataType::F16 || dtype == DataType::BF16)
        && out->dtype() == dtype
        && k_cache->dtype() == dtype
        && v_cache->dtype() == dtype
        && k_cache->shape() == v_cache->shape()
        && out->size(0) == q->size(0)
        && out->size(1) == q->size(1)
        && out->size(2) == v_cache->size(3)
        && q->size(0) == block_tables->size(0)
        && q->size(0) == cache_lens->size(0)
        && k_cache->size(1) > 0
        && q->size(1) % k_cache->size(1) == 0
        && q->size(2) == k_cache->size(3)
        && q->size(2) == v_cache->size(3)
        && q->size(2) <= 256
        && q->size(2) % 8 == 0
        && k_cache->size(2) % 256 == 0
        && q->stride(2) == 1
        && out->stride(2) == 1
        && k_cache->stride(3) == 1
        && v_cache->stride(3) == 1
        && block_tables->dtype() == DataType::I32
        && cache_lens->dtype() == DataType::I32
        && block_tables->is_contiguous()
        && cache_lens->is_contiguous();
#else
    (void)out;
    (void)q;
    (void)k_cache;
    (void)v_cache;
    (void)block_tables;
    (void)cache_lens;
    return false;
#endif
}

struct PlannedMeta {
    TensorMeta flash_out, flash_q, flash_k_cache, flash_v_cache;
    TensorMeta block_tables, cache_lens;
    std::optional<TensorMeta> alibi_slopes;
    graph::GraphTensor out_tensor, q_tensor, k_cache_tensor, v_cache_tensor, block_tables_tensor, cache_lens_tensor;
    std::optional<graph::GraphTensor> alibi_slopes_tensor;
    void *fallback_meta;
    bool use_flash_attention;
    float scale;
};
} // namespace

void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k_cache,
           const Tensor &v_cache,
           const Tensor &block_tables,
           const Tensor &cache_lens,
           std::optional<Tensor> alibi_slopes,
           float scale) {
    INFINICORE_ASSERT(::infinicore::op::infiniops::isSupportedDevice(out->device().getType()));
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, q, k_cache, v_cache, block_tables, cache_lens);
    if (alibi_slopes) {
        INFINICORE_ASSERT_TENSORS_SAME_DEVICE(out, *alibi_slopes);
    }

    const bool use_flash_attention = canUseFlashAttention(out, q, k_cache, v_cache, block_tables, cache_lens);
    auto flash_out = out->unsqueeze(1);
    auto flash_q = q->unsqueeze(1);
    auto flash_k_cache = k_cache->permute({0, 2, 1, 3});
    auto flash_v_cache = v_cache->permute({0, 2, 1, 3});
    void *fallback_meta = use_flash_attention
                            ? nullptr
                            : paged_attention_impl::infiniop::plan(
                                out, q, k_cache, v_cache, block_tables, cache_lens, alibi_slopes, scale);

    return new PlannedMeta{
        TensorMeta(flash_out), TensorMeta(flash_q), TensorMeta(flash_k_cache), TensorMeta(flash_v_cache),
        TensorMeta(block_tables), TensorMeta(cache_lens),
        alibi_slopes ? std::optional<TensorMeta>{TensorMeta(*alibi_slopes)} : std::nullopt,
        graph::GraphTensor(out), graph::GraphTensor(q), graph::GraphTensor(k_cache), graph::GraphTensor(v_cache), graph::GraphTensor(block_tables), graph::GraphTensor(cache_lens),
        alibi_slopes ? std::optional<graph::GraphTensor>{graph::GraphTensor(*alibi_slopes)} : std::nullopt,
        fallback_meta,
        use_flash_attention,
        scale};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

#ifdef ENABLE_INFINIOPS_LINKED_FLASH_ATTN_WITH_KVCACHE
    if (planned->use_flash_attention) {
        infini::ops::Handle handle;
        handle.set_stream(context::getStream());
        infini::ops::Config config;
        config.set_implementation_index(16);
        const std::optional<infini::ops::Tensor> no_tensor;
        const std::optional<infini::ops::Tensor> cache_lens{
            planned->cache_lens.tensor(planned->cache_lens_tensor)};
        const std::optional<infini::ops::Tensor> block_tables{
            planned->block_tables.tensor(planned->block_tables_tensor)};
        const std::optional<infini::ops::Tensor> alibi_slopes = planned->alibi_slopes
                                                                  ? std::optional<infini::ops::Tensor>{planned->alibi_slopes->tensor(planned->alibi_slopes_tensor.value()->data())}
                                                                  : std::nullopt;
        infini::ops::FlashAttnWithKvcache::Call(
            handle,
            config,
            planned->flash_q.tensor(planned->q_tensor),
            planned->flash_k_cache.tensor(planned->k_cache_tensor),
            planned->flash_v_cache.tensor(planned->v_cache_tensor),
            no_tensor,
            no_tensor,
            no_tensor,
            no_tensor,
            cache_lens,
            no_tensor,
            no_tensor,
            block_tables,
            alibi_slopes,
            std::optional<double>{planned->scale},
            true,
            std::vector<std::int64_t>{-1, -1},
            0.0,
            true,
            std::int64_t{0},
            false,
            planned->flash_out.tensor(planned->out_tensor),
            no_tensor);
        return;
    }
#endif

    INFINICORE_ASSERT(planned->fallback_meta != nullptr);
    paged_attention_impl::infiniop::run(planned->fallback_meta);
}

void cleanup(void **planned_meta_ptr) {
    auto planned = *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    if (planned->fallback_meta != nullptr) {
        paged_attention_impl::infiniop::cleanup(&planned->fallback_meta);
    }
    delete planned;
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    ::infinicore::op::infiniops::registerSupportedDevices(PagedAttention::plan_dispatcher(), &plan);
    ::infinicore::op::infiniops::registerSupportedDevices(PagedAttention::run_dispatcher(), &run);
    ::infinicore::op::infiniops::registerSupportedDevices(PagedAttention::cleanup_dispatcher(), &cleanup);
    return true;
}();
} // namespace infinicore::op::paged_attention_impl::infiniops
#endif
