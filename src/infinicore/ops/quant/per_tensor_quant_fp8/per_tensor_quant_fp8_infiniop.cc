#include "../../infiniop_impl.hpp"
#include "infinicore/ops/per_tensor_quant_fp8.hpp"

namespace infinicore::op::per_tensor_quant_fp8_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, PerTensorQuantFp8, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, x, x_packed, x_scale;
    const bool is_static;
};

void *plan(const Tensor &x, Tensor x_packed, Tensor x_scale, bool is_static) {
    size_t seed = hash_combine(x, x_packed, x_scale);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, PerTensorQuantFp8,
        seed,
        x_packed->desc(), x_scale->desc(), x->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, PerTensorQuantFp8, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(x),
        graph::GraphTensor(x_packed),
        graph::GraphTensor(x_scale),
        is_static};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    const bool is_static = planned->is_static;
    INFINICORE_CHECK_ERROR(infiniopPerTensorQuantFp8(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->x_packed->data(),
        planned->x_scale->data(),
        planned->x->data(),
        is_static,
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(PerTensorQuantFp8, &plan, &run, &cleanup);

} // namespace infinicore::op::per_tensor_quant_fp8_impl::infiniop
