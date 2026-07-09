#include "infinicore/ops/deepseek_moe_w8a8i8.hpp"

#include "../infiniop_impl.hpp"

#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace infinicore::op::deepseek_moe_w8a8i8_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, DeepseekMoeW8A8I8, 100);

struct ExpertPtrTables {
    std::shared_ptr<Memory> device_buffer;
    size_t num_experts;
};

struct PtrTableCacheEntry {
    std::vector<const void *> ptrs;
    std::shared_ptr<ExpertPtrTables> tables;
};

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;
    Tensor workspace_owner, out_owner, hidden_owner, topk_indices_owner, topk_weights_owner;
    graph::GraphTensor workspace, out, hidden, topk_indices, topk_weights;
    std::vector<graph::GraphTensor> gate_weights, up_weights, down_weights;
    std::vector<graph::GraphTensor> gate_weight_scales, up_weight_scales, down_weight_scales;
    std::shared_ptr<ExpertPtrTables> ptr_tables;
};

struct PlannedWithPtrTablesMeta {
    std::shared_ptr<Descriptor> descriptor;
    Tensor workspace_owner, out_owner, hidden_owner, topk_indices_owner, topk_weights_owner, ptr_tables_owner;
    graph::GraphTensor workspace, out, hidden, topk_indices, topk_weights, ptr_tables;
    size_t num_experts;
};

static std::mutex &ptr_table_cache_mutex() {
    static auto *mutex = new std::mutex;
    return *mutex;
}

static std::unordered_map<size_t, std::vector<PtrTableCacheEntry>> &ptr_table_cache() {
    static auto *cache = new std::unordered_map<size_t, std::vector<PtrTableCacheEntry>>;
    return *cache;
}

static std::vector<graph::GraphTensor> to_graph_tensors(const std::vector<Tensor> &tensors) {
    std::vector<graph::GraphTensor> result;
    result.reserve(tensors.size());
    for (const auto &tensor : tensors) {
        result.emplace_back(tensor);
    }
    return result;
}

static void append_data_ptrs(std::vector<const void *> &out,
                             const std::vector<graph::GraphTensor> &tensors) {
    out.reserve(out.size() + tensors.size());
    for (const auto &tensor : tensors) {
        out.push_back(tensor->data());
    }
}

static size_t ptr_table_key(const std::vector<const void *> &ptrs,
                            size_t num_experts,
                            const Device &device) {
    size_t seed = num_experts;
    auto combine = [&seed](uintptr_t value) {
        seed ^= std::hash<uintptr_t>{}(value) + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    };
    combine(static_cast<uintptr_t>(device.getIndex()));
    combine(static_cast<uintptr_t>(device.getType()));
    for (const auto *ptr : ptrs) {
        combine(reinterpret_cast<uintptr_t>(ptr));
    }
    return seed;
}

static std::shared_ptr<ExpertPtrTables> get_or_create_ptr_tables(
    const std::vector<const void *> &ptrs,
    size_t num_experts,
    const Device &device) {
    const size_t key = ptr_table_key(ptrs, num_experts, device);
    std::lock_guard<std::mutex> lock(ptr_table_cache_mutex());
    auto &bucket = ptr_table_cache()[key];
    for (const auto &entry : bucket) {
        if (entry.ptrs == ptrs) {
            return entry.tables;
        }
    }

    const size_t total_bytes = ptrs.size() * sizeof(void *);
    auto tables = std::make_shared<ExpertPtrTables>();
    tables->device_buffer = context::allocateMemory(total_bytes);
    tables->num_experts = num_experts;
    context::memcpyH2D(tables->device_buffer->data(), ptrs.data(), total_bytes, false);
    bucket.push_back(PtrTableCacheEntry{ptrs, tables});
    return tables;
}

static void *ptr_table_at(const std::shared_ptr<ExpertPtrTables> &tables,
                          size_t table_index) {
    const size_t ptr_bytes = tables->num_experts * sizeof(void *);
    return tables->device_buffer->data() + table_index * ptr_bytes;
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
        nullptr};

    std::vector<const void *> ptrs;
    ptrs.reserve(num_experts * 6);
    append_data_ptrs(ptrs, planned->gate_weights);
    append_data_ptrs(ptrs, planned->up_weights);
    append_data_ptrs(ptrs, planned->down_weights);
    append_data_ptrs(ptrs, planned->gate_weight_scales);
    append_data_ptrs(ptrs, planned->up_weight_scales);
    append_data_ptrs(ptrs, planned->down_weight_scales);
    planned->ptr_tables = get_or_create_ptr_tables(ptrs, num_experts, hidden->device());
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
        ptr_table_at(planned->ptr_tables, 0),
        ptr_table_at(planned->ptr_tables, 1),
        ptr_table_at(planned->ptr_tables, 2),
        ptr_table_at(planned->ptr_tables, 3),
        ptr_table_at(planned->ptr_tables, 4),
        ptr_table_at(planned->ptr_tables, 5),
        context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}


static void *ptr_table_tensor_at(const graph::GraphTensor &tables,
                                 size_t num_experts,
                                 size_t table_index) {
    const size_t ptr_bytes = num_experts * sizeof(void *);
    return const_cast<std::byte *>(tables->data()) + table_index * ptr_bytes;
}

void *plan_with_ptr_tables(Tensor out,
                           const Tensor &hidden,
                           const Tensor &topk_indices,
                           const Tensor &topk_weights,
                           const Tensor &ptr_tables,
                           size_t intermediate_size,
                           size_t num_experts) {
    size_t seed = hash_combine(out, hidden, topk_indices, topk_weights, ptr_tables);
    hash_combine(seed, intermediate_size, num_experts);

    INFINIOP_CACHABLE_DESCRIPTOR_GET_OR_CREATE(
        Descriptor, descriptor, DeepseekMoeW8A8I8, seed,
        out->desc(), hidden->desc(), topk_indices->desc(), topk_weights->desc(),
        intermediate_size, num_experts);

    INFINIOP_WORKSPACE_TENSOR(workspace, DeepseekMoeW8A8I8, descriptor);

    auto planned = new PlannedWithPtrTablesMeta{
        descriptor,
        workspace,
        out,
        hidden,
        topk_indices,
        topk_weights,
        ptr_tables,
        graph::GraphTensor(workspace),
        graph::GraphTensor(out),
        graph::GraphTensor(hidden),
        graph::GraphTensor(topk_indices),
        graph::GraphTensor(topk_weights),
        graph::GraphTensor(ptr_tables),
        num_experts};
    return planned;
}

void run_with_ptr_tables(void *planned_meta) {
    auto planned = reinterpret_cast<PlannedWithPtrTablesMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(infiniopDeepseekMoeW8A8I8WithDevicePtrs(
        planned->descriptor->desc,
        planned->workspace->data(),
        planned->workspace->numel(),
        planned->out->data(),
        planned->hidden->data(),
        planned->topk_indices->data(),
        planned->topk_weights->data(),
        ptr_table_tensor_at(planned->ptr_tables, planned->num_experts, 0),
        ptr_table_tensor_at(planned->ptr_tables, planned->num_experts, 1),
        ptr_table_tensor_at(planned->ptr_tables, planned->num_experts, 2),
        ptr_table_tensor_at(planned->ptr_tables, planned->num_experts, 3),
        ptr_table_tensor_at(planned->ptr_tables, planned->num_experts, 4),
        ptr_table_tensor_at(planned->ptr_tables, planned->num_experts, 5),
        context::getStream()));
}

void cleanup_with_ptr_tables(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedWithPtrTablesMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekMoeW8A8I8, &plan, &run, &cleanup);
namespace with_ptr_tables_registration {
INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(DeepseekMoeW8A8I8WithPtrTables, &plan_with_ptr_tables, &run_with_ptr_tables, &cleanup_with_ptr_tables);
} // namespace with_ptr_tables_registration

} // namespace infinicore::op::deepseek_moe_w8a8i8_impl::infiniop
