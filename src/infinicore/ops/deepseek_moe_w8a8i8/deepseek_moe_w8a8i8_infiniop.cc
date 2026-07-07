#include "infinicore/ops/deepseek_moe_w8a8i8.hpp"

#include "../infiniop_impl.hpp"

namespace infinicore::op::deepseek_moe_w8a8i8_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, DeepseekMoeW8A8I8, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    Tensor workspace_owner, out_owner, hidden_owner, topk_indices_owner, topk_weights_owner;
    graph::GraphTensor workspace, out, hidden, topk_indices, topk_weights;
    std::vector<graph::GraphTensor> gate_weights, up_weights, down_weights;
    std::vector<graph::GraphTensor> gate_weight_scales, up_weight_scales, down_weight_scales;
    std::vector<const void *> gate_ptrs, up_ptrs, down_ptrs;
    std::vector<const void *> gate_scale_ptrs, up_scale_ptrs, down_scale_ptrs;
    std::shared_ptr<Memory> gate_ptrs_device, up_ptrs_device, down_ptrs_device;
    std::shared_ptr<Memory> gate_scale_ptrs_device, up_scale_ptrs_device, down_scale_ptrs_device;
};

static std::vector<graph::GraphTensor> to_graph_tensors(const std::vector<Tensor> &tensors) {
    std::vector<graph::GraphTensor> result;
    result.reserve(tensors.size());
    for (const auto &tensor : tensors) {
        result.emplace_back(tensor);
    }
    return result;
}

static std::vector<const void *> data_ptrs(const std::vector<graph::GraphTensor> &tensors) {
    std::vector<const void *> result;
    result.reserve(tensors.size());
    for (const auto &tensor : tensors) {
        result.push_back(tensor->data());
    }
    return result;
}

void *plan(Tensor out,
           const Tensor &hidden,
           const Tensor &topk_indices,
           const Tensor &topk_weights,
           const std::vector<Tensor> &gate_weights,
           const std::vector<Tensor> &up_weights,
           const std::vector<Tensor> &down_weights,
           const std::vector<Tensor> &gate_weight_scales,
           const std::vector<Tensor> &up_weight_scales,
           const std::vector<Tensor> &down_weight_scales,
           size_t intermediate_size,
           size_t num_experts) {
    size_t seed = hash_combine(out, hidden, topk_indices, topk_weights);
    hash_combine(seed, intermediate_size, num_experts);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, DeepseekMoeW8A8I8, seed,
        out->desc(), hidden->desc(), topk_indices->desc(), topk_weights->desc(),
        intermediate_size, num_experts);

    INFINIOP_WORKSPACE_TENSOR(workspace, DeepseekMoeW8A8I8, descriptor);

    auto planned = new PlannedMeta{
        descriptor,
        workspace,
        out,
        hidden,
        topk_indices,
        topk_weights,
        graph::GraphTensor(workspace),
        graph::GraphTensor(out),
        graph::GraphTensor(hidden),
        graph::GraphTensor(topk_indices),
        graph::GraphTensor(topk_weights),
        to_graph_tensors(gate_weights),
        to_graph_tensors(up_weights),
        to_graph_tensors(down_weights),
        to_graph_tensors(gate_weight_scales),
        to_graph_tensors(up_weight_scales),
        to_graph_tensors(down_weight_scales),
        {},
        {},
        {},
        {},
        {},
        {},
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr,
        nullptr};
    planned->gate_ptrs = data_ptrs(planned->gate_weights);
    planned->up_ptrs = data_ptrs(planned->up_weights);
    planned->down_ptrs = data_ptrs(planned->down_weights);
    planned->gate_scale_ptrs = data_ptrs(planned->gate_weight_scales);
    planned->up_scale_ptrs = data_ptrs(planned->up_weight_scales);
    planned->down_scale_ptrs = data_ptrs(planned->down_weight_scales);

    const size_t ptr_bytes = num_experts * sizeof(void *);
    planned->gate_ptrs_device = context::allocateMemory(ptr_bytes);
    planned->up_ptrs_device = context::allocateMemory(ptr_bytes);
    planned->down_ptrs_device = context::allocateMemory(ptr_bytes);
    planned->gate_scale_ptrs_device = context::allocateMemory(ptr_bytes);
    planned->up_scale_ptrs_device = context::allocateMemory(ptr_bytes);
    planned->down_scale_ptrs_device = context::allocateMemory(ptr_bytes);
    context::memcpyH2D(planned->gate_ptrs_device->data(), planned->gate_ptrs.data(), ptr_bytes, false);
    context::memcpyH2D(planned->up_ptrs_device->data(), planned->up_ptrs.data(), ptr_bytes, false);
    context::memcpyH2D(planned->down_ptrs_device->data(), planned->down_ptrs.data(), ptr_bytes, false);
    context::memcpyH2D(planned->gate_scale_ptrs_device->data(), planned->gate_scale_ptrs.data(), ptr_bytes, false);
    context::memcpyH2D(planned->up_scale_ptrs_device->data(), planned->up_scale_ptrs.data(), ptr_bytes, false);
    context::memcpyH2D(planned->down_scale_ptrs_device->data(), planned->down_scale_ptrs.data(), ptr_bytes, false);
    return planned;
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopDeepseekMoeW8A8I8WithDevicePtrs(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->out->data(),
        planned->hidden->data(),
        planned->topk_indices->data(),
        planned->topk_weights->data(),
        planned->gate_ptrs_device->data(),
        planned->up_ptrs_device->data(),
        planned->down_ptrs_device->data(),
        planned->gate_scale_ptrs_device->data(),
        planned->up_scale_ptrs_device->data(),
        planned->down_scale_ptrs_device->data(),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekMoeW8A8I8, &plan, &run, &cleanup);

} // namespace infinicore::op::deepseek_moe_w8a8i8_impl::infiniop
