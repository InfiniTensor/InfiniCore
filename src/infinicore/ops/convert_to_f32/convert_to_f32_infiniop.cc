#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/convert_to_f32.hpp"

#include <infiniop.h>

namespace infinicore::op::convert_to_f32_impl::infiniop {

thread_local common::OpCache<size_t, infiniopConvertToF32Descriptor_t> caches(
    100,
    [](infiniopConvertToF32Descriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyConvertToF32Descriptor(desc));
            desc = nullptr;
        }
    });

void calculate(Tensor output, Tensor input) {
    size_t seed = hash_combine(output, input);

    auto device = context::getDevice();
    auto &cache = caches.getCache(device);

    auto desc_opt = cache.get(seed);
    infiniopConvertToF32Descriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateConvertToF32Descriptor(
            context::getInfiniopHandle(device), &desc,
            output->desc(), input->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetConvertToF32WorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopConvertToF32(
        desc, workspace->data(), workspace_size,
        output->data(), input->data(), context::getStream()));
}

static bool registered = []() {
    ConvertToF32::dispatcher().registerAll(&calculate, false);
    return true;
}();

} // namespace infinicore::op::convert_to_f32_impl::infiniop
