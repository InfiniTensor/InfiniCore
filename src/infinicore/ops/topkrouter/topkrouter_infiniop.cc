#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/topkrouter.hpp"

#include <infiniop.h>

namespace infinicore::op::topkrouter_impl::infiniop {

thread_local common::OpCache<size_t, infiniopTopkrouterDescriptor_t> caches(
    100,
    [](infiniopTopkrouterDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyTopkrouterDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor values_output,
               Tensor indices_output,
               Tensor input,
               Tensor correction_bias,
               float routed_scaling_factor,
               size_t topk) {
    size_t seed = hash_combine(values_output, indices_output, input, correction_bias, (size_t)topk);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopTopkrouterDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateTopkrouterDescriptor(
            context::getInfiniopHandle(device), &desc,
            input->desc(), correction_bias->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetTopkrouterWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopTopkrouter(
        desc, workspace->data(), workspace_size,
        values_output->data(), indices_output->data(),
        input->data(), correction_bias->data(),
        routed_scaling_factor, topk, context::getStream()));
}

static bool registered = []() {
    TopKRouter::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::topkrouter_impl::infiniop

