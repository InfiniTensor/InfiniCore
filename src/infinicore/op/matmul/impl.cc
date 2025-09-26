#include "infinicore/common/utils.hpp"
#include "infinicore/op/common/cache.hpp"
#include "infinicore/op/matmul.h"
#include <infiniop.h>

namespace infinicore::op {

common::OpDispatcher<Matmul::schema> Matmul::dispatcher;

void Matmul::execute(Tensor c, Tensor a, Tensor b) {
    dispatcher.lookup(context::getDevice().getType())(c, a, b);
}

} // namespace infinicore::op

namespace infinicore::op::matmul_impl {

thread_local common::OpCache<size_t, infiniopGemmDescriptor_t> caches(
    100, // capacity
    [](infiniopGemmDescriptor_t &desc) {
        if (desc != nullptr) {
            INFINICORE_CHECK_ERROR(infiniopDestroyGemmDescriptor(desc));
            desc = nullptr;
        }
    });

void infiniop(Tensor c, Tensor a, Tensor b) {
    size_t seed = hash_combine(c, b, a);

    auto device_type = context::getDevice().getType();
    auto device_index = context::getDevice().getIndex();

    auto &cache = caches.getCache(device_type, device_index);

    auto desc_opt = cache.get(seed);
    infiniopGemmDescriptor_t desc = nullptr;

    if (!desc_opt) {
        INFINICORE_CHECK_ERROR(infiniopCreateGemmDescriptor(context::getInfiniopHandle(), &desc, c->desc(), a->desc(), b->desc()));
        cache.put(seed, desc);
    } else {
        desc = *desc_opt;
    }

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetGemmWorkspaceSize(desc, &workspace_size));
    std::shared_ptr<Memory> workspace = context::allocateMemory(workspace_size);

    INFINICORE_CHECK_ERROR(infiniopGemm(
        desc, workspace->data(), workspace_size,
        c->data(), a->data(), b->data(), 1.f, 0.f, context::getStream()));
}

inline struct MatmulRegistrar {
    MatmulRegistrar() {
        Matmul::dispatcher.registerAll(infiniop);
    }
} matmul_registrar;

} // namespace infinicore::op::matmul_impl

namespace infinicore::op {
Tensor matmul(Tensor a, Tensor b) {
    Shape shape = a->shape();
    Size size = a->ndim();
    shape[size - 1] = b->size(size - 1);
    auto c = Tensor::empty(shape, a->dtype(), a->device());
    matmul_(c, a, b);
    return c;
}

void matmul_(Tensor c, Tensor a, Tensor b) {
    Matmul::execute(c, a, b);
}
} // namespace infinicore::op
