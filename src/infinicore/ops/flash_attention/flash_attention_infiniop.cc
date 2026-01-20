#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/flash_attention.hpp"
#include <infiniop.h>

namespace infinicore::op::flash_attention_impl::infiniop {

thread_local common::OpCache<size_t, infiniopFlashAttentionDescriptor_t> caches(
    100, // capacity
    [](infiniopFlashAttentionDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyFlashAttentionDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor out, Tensor q, Tensor k, Tensor v, std::size_t total_kv_len, float scale, bool is_causal) {
    size_t seed = hash_combine(out, q, k, v, total_kv_len, scale, is_causal);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopFlashAttentionDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateFlashAttentionDescriptor(
            context::getInfiniopHandle(device), &desc,
            out->desc(), q->desc(), k->desc(), v->desc(), total_kv_len,
            scale, static_cast<char>(is_causal)));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetFlashAttentionWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopFlashAttention(
        desc, workspace->data(), workspace_size,
        out->data(), q->data(), k->data(), v->data(), context::getStream()));
}

static bool registered = []() {
    FlashAttention::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::flash_attention_impl::infiniop
