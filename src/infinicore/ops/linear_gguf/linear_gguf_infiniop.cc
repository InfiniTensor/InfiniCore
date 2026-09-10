#include "infinicore/ops/linear_gguf.hpp"

// Included directly rather than through <infiniop.h>, so that adding this op
// needs no edit to the umbrella header that lists every other op.
#include "infiniop/ops/linear_gguf.h"

#include "../infiniop_impl.hpp"

namespace infinicore::op::linear_gguf_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, LinearGguf, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor output;
    graph::GraphTensor input;
    graph::GraphTensor weight;
    int64_t ggml_type;
};

void *plan(Tensor output,
           const Tensor &input,
           const Tensor &weight,
           int64_t ggml_type) {
    // The type id is part of the key: the same shapes with different block
    // formats are different kernels, so they must not share a descriptor.
    size_t seed = hash_combine(output, input, weight, static_cast<size_t>(ggml_type));
    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor,
        descriptor,
        LinearGguf,
        seed,
        output->desc(),
        input->desc(),
        weight->desc(),
        ggml_type);
    INFINIOP_WORKSPACE_TENSOR(workspace, LinearGguf, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(output),
        graph::GraphTensor(input),
        graph::GraphTensor(weight),
        ggml_type};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);
    INFINICORE_CHECK_ERROR(infiniopLinearGguf(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->output->data(),
        planned->input->data(),
        planned->weight->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(LinearGguf, &plan, &run, &cleanup);

} // namespace infinicore::op::linear_gguf_impl::infiniop
