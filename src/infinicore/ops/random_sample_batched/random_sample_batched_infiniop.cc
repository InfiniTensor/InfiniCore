#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/random_sample_batched.hpp"
#include <infiniop.h>

namespace infinicore::op::random_sample_batched_impl::infiniop_backend {

thread_local common::OpCache<size_t, infiniopRandomSampleBatchedDescriptor_t> caches(
    100, // capacity
    [](infiniopRandomSampleBatchedDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyRandomSampleBatchedDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(
    Tensor result,
    Tensor probs,
    const float *random_val,
    const float *topp,
    const int *topk,
    const float *temperature,
    int batch_size) {
    size_t seed = hash_combine(result, probs, batch_size);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopRandomSampleBatchedDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateRandomSampleBatchedDescriptor(
            context::getInfiniopHandle(device), &desc,
            result->desc(), probs->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetRandomSampleBatchedWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopRandomSampleBatched(
        desc,
        workspace->data(), workspace_size,
        result->data(), probs->data(),
        random_val, topp, topk, temperature,
        batch_size,
        context::getStream()));
}

} // namespace infinicore::op::random_sample_batched_impl::infiniop_backend

namespace infinicore::op {
static bool registered = []() {
    RandomSampleBatched::dispatcher().registerAll(&random_sample_batched_impl::infiniop_backend::calculate, false);
    return true;
}();
} // namespace infinicore::op
