#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/layer_norm.hpp"
#include <infiniop.h>

namespace infinicore::op::layer_norm_impl::infiniop {

thread_local common::OpCache<size_t, infiniopLayerNormDescriptor_t> caches(
    100, // capacity
    [](infiniopLayerNormDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyLayerNormDescriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output,
               Tensor input_standardization,
               Tensor input_std_deviation,
               Tensor input,
               Tensor weight,
               Tensor bias,
               float epsilon) {
    size_t seed = hash_combine(output, input_standardization, input_std_deviation, input, weight, bias, epsilon);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopLayerNormDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateLayerNormDescriptor(
            context::getInfiniopHandle(device), &desc,
            output->desc(),
            input_standardization->desc(),
            input_std_deviation->desc(),
            input->desc(),
            weight ? weight->desc() : nullptr,
            bias ? bias->desc() : nullptr,
            epsilon));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetLayerNormWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopLayerNorm(
        desc, workspace->data(), workspace_size,
        output->data(),
        input_standardization->data(),
        input_std_deviation->data(),
        input->data(),
        weight ? weight->data() : nullptr,
        bias ? bias->data() : nullptr,
        context::getStream()));
}

static bool registered = []() {
    LayerNorm::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::layer_norm_impl::infiniop
