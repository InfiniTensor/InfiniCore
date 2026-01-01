#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/rope.hpp"
#include <infiniop.h>

namespace infinicore::op::rope_impl::infiniop {

thread_local common::OpCache<size_t, infiniopRoPEDescriptor_t> caches(
    100, // capacity
    [](infiniopRoPEDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyRoPEDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor x_out, const Tensor &x, const Tensor &pos, const Tensor &sin_cache, const Tensor &cos_cache, infinicore::nn::RoPE::Algo algo) {
    // Convert infinicore::nn::RoPE::Algo to infiniopRoPEAlgo_t
    infiniopRoPEAlgo_t infiniop_algo;
    switch (algo) {
    case infinicore::nn::RoPE::Algo::GPT_J:
        infiniop_algo = INFINIOP_ROPE_ALGO_GPT_J;
        break;
    case infinicore::nn::RoPE::Algo::GPT_NEOX:
        infiniop_algo = INFINIOP_ROPE_ALGO_GPT_NEOX;
        break;
    default:
        throw std::runtime_error("Unsupported RoPE algorithm: " + std::to_string(static_cast<int>(algo)));
    }

    // Create hash key for descriptor caching
    size_t key = hash_combine(x_out, x, pos, sin_cache, cos_cache);
    hash_combine(key, std::hash<int>()(static_cast<int>(infiniop_algo)));

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(key);
    infiniopRoPEDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateRoPEDescriptor(
            context::getInfiniopHandle(device), &desc,
            x_out->desc(), x->desc(),
            pos->desc(), sin_cache->desc(), cos_cache->desc(),
            infiniop_algo));
        cache.put(key, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetRoPEWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    // InfiniOP reads from x and writes to x_out (handles copying internally)
    INFINICORE_CHECK_ERROR(infiniopRoPE(
        desc, workspace->data(), workspace_size,
        x_out->data(), x->data(), pos->data(),
        sin_cache->data(), cos_cache->data(), context::getStream()));
}

static bool registered = []() {
    RoPE::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::rope_impl::infiniop
