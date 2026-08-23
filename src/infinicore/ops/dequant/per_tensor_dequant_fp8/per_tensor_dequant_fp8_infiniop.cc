#include "../../infiniop_impl.hpp"
#include "infinicore/ops/per_tensor_dequant_fp8.hpp"

namespace infinicore::op::per_tensor_dequant_fp8_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, PerTensorDequantFp8, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace, x, x_packed, x_scale;
};

void *plan(const Tensor &x, const Tensor &x_packed, const Tensor &x_scale) {
    size_t seed = hash_combine(x, x_packed, x_scale);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, PerTensorDequantFp8,
        seed,
        x->desc(), x_packed->desc(), x_scale->desc());

    INFINIOP_WORKSPACE_TENSOR(workspace, PerTensorDequantFp8, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(x),
        graph::GraphTensor(x_packed),
        graph::GraphTensor(x_scale)};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopPerTensorDequantFp8(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->x->data(),
        planned->x_packed->data(),
        planned->x_scale->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(PerTensorDequantFp8, &plan, &run, &cleanup);

} // namespace infinicore::op::per_tensor_dequant_fp8_impl::infiniop
