#if defined(ENABLE_HYGON_API) && defined(ENABLE_FLASH_ATTN)
#include "infinicore/ops/mha_kvcache.hpp"

#include "../../../adaptor/flash_attn/hygon/flash_attn_hygon.hpp"
#include "infinicore/adaptor/aten_adaptor.hpp"

#include <c10/hip/HIPGuard.h>

#include <limits>
#include <stdexcept>
#include <string>

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
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    // Paged KV caches must be contiguous for flash-attn; avoid extra copies for q/metadata when already dense.
    const bool out_need_copy_back = !p->out->is_contiguous();
    Tensor out_work = out_need_copy_back ? p->out->contiguous() : Tensor(p->out);
    auto out_tensor = infinicore::adaptor::to_aten_tensor(out_work);
    auto q = infinicore::adaptor::to_aten_tensor(p->q);
    auto k_cache = infinicore::adaptor::to_aten_tensor(p->k_cache);
    auto v_cache = infinicore::adaptor::to_aten_tensor(p->v_cache);
    auto seqlens_k_tensor = infinicore::adaptor::to_aten_tensor(p->seqlens_k);
    auto block_table_tensor = infinicore::adaptor::to_aten_tensor(p->block_table);
    auto seqlens_k = std::optional<const at::Tensor>(seqlens_k_tensor);
    auto block_table = std::optional<at::Tensor>(block_table_tensor);
    auto alibi_slopes = p->alibi_slopes
                          ? std::optional<at::Tensor>(infinicore::adaptor::to_aten_tensor(*p->alibi_slopes))
                          : std::nullopt;

    const bool use_paged_attention =
        q.dim() == 4 && q.size(1) == 1
        && k_cache.dim() == 4 && v_cache.dim() == 4
        && k_cache.size(0) == v_cache.size(0)
        && k_cache.size(1) == v_cache.size(1)
        && k_cache.size(2) == v_cache.size(3)
        && k_cache.size(3) == v_cache.size(2)
        && k_cache.size(2) == 64
        && q.size(2) % k_cache.size(1) == 0
        && q.size(3) == k_cache.size(3)
        && q.is_contiguous() && k_cache.is_contiguous() && v_cache.is_contiguous()
        && seqlens_k_tensor.dim() == 1 && block_table_tensor.dim() == 2
        && !alibi_slopes.has_value();
    if (use_paged_attention) {
        const auto max_context_len_64 = block_table_tensor.size(1) * k_cache.size(2);
        if (max_context_len_64 > std::numeric_limits<int>::max()) {
            throw std::runtime_error("paged_attention max context length exceeds int range");
        }
        auto paged_out = out_tensor.view({q.size(0), q.size(2), q.size(3)});
        const std::optional<at::Tensor> none = std::nullopt;
        static const std::string kv_cache_dtype = "auto";
        flash::paged_attention(
            paged_out,
            q,
            k_cache,
            v_cache,
            p->scale,
            block_table_tensor,
            seqlens_k_tensor,
            none,
            kv_cache_dtype,
            none,
            none,
            none,
            static_cast<int>(max_context_len_64),
            none);
        if (out_need_copy_back) {
            p->out->copy_from(out_work);
        }
        return;
    }

    std::optional<const at::Tensor> k_new = std::nullopt;
    std::optional<const at::Tensor> v_new = std::nullopt;
    std::optional<const at::Tensor> rotary_cos = std::nullopt;
    std::optional<const at::Tensor> rotary_sin = std::nullopt;
    std::optional<const at::Tensor> cache_batch_idx = std::nullopt;
    std::optional<const at::Tensor> leftpad_k = std::nullopt;
    const bool needs_grouped_out_alias = q.dim() == 4
                                      && k_cache.dim() == 4 && v_cache.dim() == 4
                                      && q.size(1) == 1 && k_cache.size(2) > 0
                                      && q.size(2) > k_cache.size(2)
                                      && q.size(2) % k_cache.size(2) == 0
                                      && q.size(3) == v_cache.size(3)
                                      && q.size(3) % 8 == 0
                                      && v_cache.sizes() == k_cache.sizes()
                                      && !alibi_slopes.has_value();
    auto direct_out = needs_grouped_out_alias
                        ? out_tensor.view({q.size(0), q.size(2) / k_cache.size(2), k_cache.size(2), v_cache.size(3)})
                        : out_tensor;
    auto out = std::optional<at::Tensor>(direct_out);

    auto result = flash::mha_fwd_kvcache(
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
        0);

    if (!result.empty() && result[0].defined()
        && result[0].data_ptr() != out_tensor.data_ptr()) {
        out_tensor.copy_(result[0]);
    }
    if (out_need_copy_back) {
        p->out->copy_from(out_work);
    }
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    MhaKVCache::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    MhaKVCache::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    MhaKVCache::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
    return true;
}();

} // namespace infinicore::op::mha_kvcache_impl::flashattn
#endif // ENABLE_HYGON_API && ENABLE_FLASH_ATTN
