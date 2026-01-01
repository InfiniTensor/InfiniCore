#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/causal_softmax.hpp"
#include "infinicore/ops/common/cache.hpp"
#include <infiniop.h>

namespace infinicore::op::causal_softmax_impl::infiniop {

thread_local common::OpCache<size_t, infiniopCausalSoftmaxDescriptor_t> caches(
    100, // capacity
    [](infiniopCausalSoftmaxDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyCausalSoftmaxDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input) {
    size_t seed = hash_combine(output, input);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopCausalSoftmaxDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateCausalSoftmaxDescriptor(
            context::getInfiniopHandle(device), &desc,
            output->desc(), input->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetCausalSoftmaxWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopCausalSoftmax(
        desc, workspace->data(), workspace_size,
        output->data(), input->data(), context::getStream()));
}

static bool registered = []() {
    CausalSoftmax::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::causal_softmax_impl::infiniop
