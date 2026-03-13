#include "infinicore/ops/eye.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::eye_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, Eye, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor y;
};

void *plan(Tensor y) {
    size_t seed = hash_combine(y);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, Eye,
        seed, y->desc());

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(y)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    size_t workspace_size = 0;
    INFINICORE_CHECK_ERROR(infiniopGetEyeWorkspaceSize(planned->descriptor->desc, &workspace_size));

    if (workspace_size > 0) {
        auto workspace = context::allocateMemory(workspace_size);
        INFINICORE_CHECK_ERROR(
            infiniopEye(
                planned->descriptor->desc,
                workspace->data(),
                workspace_size,
                planned->y->data(),
                context::getStream()));
    } else {
        INFINICORE_CHECK_ERROR(
            infiniopEye(
                planned->descriptor->desc,
                nullptr,
                0,
                planned->y->data(),
                context::getStream()));
    }
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(Eye, &plan, &run, &cleanup);

} // namespace infinicore::op::eye_impl::infiniop
