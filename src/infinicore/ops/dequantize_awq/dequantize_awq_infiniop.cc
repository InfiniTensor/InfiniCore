#include "../../utils.hpp"
#include "infinicore/common/hash.hpp"
#include "infinicore/ops/common/cache.hpp"
#include "infinicore/ops/dequantize_awq.hpp"
#include <infiniop.h>
#include <iostream>
namespace infinicore::op::dequantize_awq_impl::infiniop {
thread_local common::OpCache<size_t, infiniopDequantizeAWQDescriptor_t> caches(
    100, // capacity
    [](infiniopDequantizeAWQDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyDequantizeAWQDescriptor(desc));
            desc = nullptr;
        }
    });
void calculate(Tensor x, Tensor x_packed, Tensor x_scale, Tensor x_zeros) {
    size_t seed = hash_combine(x, x_packed, x_scale, x_zeros);
    auto device = context::getDevice();
    auto &cache = caches.getCache(device);
    auto desc_opt = cache.get(seed);
    infiniopDequantizeAWQDescriptor_t desc = nullptr;
    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateDequantizeAWQDescriptor(
            context::getInfiniopHandle(device), &desc,
            x->desc(), x_packed->desc(), x_scale->desc(), x_zeros->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }
    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetDequantizeAWQWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);
    INFINICORE_CHECK_ERROR(infiniopDequantizeAWQ(
        desc, workspace->data(), workspace_size,
        x->data(), x_packed->data(), x_scale->data(), x_zeros->data(), context::getStream()));
}
static bool registered = []() {
    DequantizeAWQ::dispatcher().registerAll(&calculate, false);
    return true;
}();
} // namespace infinicore::op::dequantize_awq_impl::infiniop
