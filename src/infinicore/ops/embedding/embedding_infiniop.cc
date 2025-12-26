#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/embedding.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>

namespace infinicore::op::embedding_impl::infiniop {

thread_local common::OpCache<size_t, infiniopEmbeddingDescriptor_t> caches(
    100, // capacity
    [](infiniopEmbeddingDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyEmbeddingDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor out, Tensor input, Tensor weight) {
    size_t seed = hash_combine(out, input, weight);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopEmbeddingDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateEmbeddingDescriptor(
            context::getInfiniopHandle(device), &desc,
            out->desc(), input->desc(), weight->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    INFINICORE_CHECK_ERROR(infiniopEmbedding(
        desc,
        out->data(),
        input->data(),
        weight->data(),
        context::getStream()));
}

static bool registered = []() {
    Embedding::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::embedding_impl::infiniop
