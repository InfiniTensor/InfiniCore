#include "infinicore/ops/moe_align.hpp"

#include <infiniop/ops/moe_align.h>

#include "../infiniop_impl.hpp"

namespace infinicore::op::moe_align_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, MoeAlign, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor sorted_token_ids;
    graph::GraphTensor expert_ids;
    graph::GraphTensor num_tokens_post_padded;
    graph::GraphTensor topk_ids;
    bool pad_sorted_token_ids;
};

void *plan(Tensor sorted_token_ids,
           Tensor expert_ids,
           Tensor num_tokens_post_padded,
           const Tensor &topk_ids,
           const size_t num_experts,
           const size_t block_size,
           const bool pad_sorted_token_ids) {
    size_t seed = hash_combine(sorted_token_ids, expert_ids, num_tokens_post_padded, topk_ids, num_experts, block_size);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, MoeAlign,
        seed,
        sorted_token_ids->desc(),
        expert_ids->desc(),
        num_tokens_post_padded->desc(),
        topk_ids->desc(),
        num_experts,
        block_size);

    INFINIOP_WORKSPACE_TENSOR(workspace, MoeAlign, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(sorted_token_ids),
        graph::GraphTensor(expert_ids),
        graph::GraphTensor(num_tokens_post_padded),
        graph::GraphTensor(topk_ids),
        pad_sorted_token_ids};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopMoeAlign(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->sorted_token_ids->data(),
        planned->expert_ids->data(),
        planned->num_tokens_post_padded->data(),
        planned->topk_ids->data(),
        nullptr,
        static_cast<int>(planned->pad_sorted_token_ids),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(MoeAlign, &plan, &run, cleanup);

} // namespace infinicore::op::moe_align_impl::infiniop

namespace infinicore::op::moe_align_with_expert_map_impl::infiniop {

struct Descriptor {
    infiniopMoeAlignDescriptor_t desc = nullptr;

    explicit Descriptor(infiniopMoeAlignDescriptor_t d)
        : desc(d) {}

    Descriptor(const Descriptor &) = delete;
    Descriptor &operator=(const Descriptor &) = delete;

    ~Descriptor() {
        if (desc != nullptr) {
            infiniopDestroyMoeAlignDescriptor(desc);
        }
    }
};

thread_local common::OpCache<size_t, std::shared_ptr<Descriptor>> caches(
    100,
    [](std::shared_ptr<Descriptor> &desc) {
        desc = nullptr;
    });

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    graph::GraphTensor workspace;
    graph::GraphTensor sorted_token_ids;
    graph::GraphTensor expert_ids;
    graph::GraphTensor num_tokens_post_padded;
    graph::GraphTensor topk_ids;
    graph::GraphTensor expert_map;
    bool pad_sorted_token_ids;
};

void *plan(Tensor sorted_token_ids,
           Tensor expert_ids,
           Tensor num_tokens_post_padded,
           const Tensor &topk_ids,
           const Tensor &expert_map,
           const size_t num_experts,
           const size_t block_size,
           const bool pad_sorted_token_ids) {
    size_t seed = hash_combine(sorted_token_ids, expert_ids, num_tokens_post_padded, topk_ids, expert_map, num_experts, block_size);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, MoeAlign,
        seed,
        sorted_token_ids->desc(),
        expert_ids->desc(),
        num_tokens_post_padded->desc(),
        topk_ids->desc(),
        num_experts,
        block_size);

    Tensor workspace;
    {
        auto device = context::getDevice();
        size_t workspace_size = 0;
        INFINICORE_CHECK_ERROR(infiniopGetMoeAlignWorkspaceSize(descriptor->desc, &workspace_size));
        workspace = Tensor::empty({workspace_size}, DataType::U8, device);
    }

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(sorted_token_ids),
        graph::GraphTensor(expert_ids),
        graph::GraphTensor(num_tokens_post_padded),
        graph::GraphTensor(topk_ids),
        graph::GraphTensor(expert_map),
        pad_sorted_token_ids};
}

void run(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopMoeAlign(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->sorted_token_ids->data(),
        planned->expert_ids->data(),
        planned->num_tokens_post_padded->data(),
        planned->topk_ids->data(),
        planned->expert_map->data(),
        static_cast<int>(planned->pad_sorted_token_ids),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(MoeAlignWithExpertMap, &plan, &run, cleanup);

} // namespace infinicore::op::moe_align_with_expert_map_impl::infiniop
