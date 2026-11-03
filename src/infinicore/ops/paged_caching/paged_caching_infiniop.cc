#include "infinicore/ops/paged_caching.hpp"

#include "../infiniop_impl.hpp"
#include "valid_slots_cache.hpp"
#include <cstdlib>
#include <cstring>
#include <unordered_set>

namespace infinicore::op::paged_caching_impl::infiniop {

INFINIOP_CACHABLE_DESCRIPTOR(Descriptor, PagedCaching, 100);

struct PlannedMeta {
    std::shared_ptr<Descriptor> descriptor;

    graph::GraphTensor workspace, k_cache, v_cache, k, v, slot_mapping;
    Tensor owned_k, owned_v;
    bool validate_graph_slots;
    int64_t slot_capacity;
};

void *plan(Tensor k_cache, Tensor v_cache, const Tensor &k, const Tensor &v, const Tensor &slot_mapping) {
    const char *flag = std::getenv("INFINICORE_ASCEND_ACLNN_CACHE");
    const char *decode = std::getenv("INFINICORE_ASCEND_DECODE_CACHE");
    const bool decode_shape = decode && std::strcmp(decode, "1") == 0
                           && k->size(0) >= 4 && k->size(0) <= 16;
    bool validated = flag && std::strcmp(flag, "1") == 0
                  && k->device().getType() == Device::Type::ASCEND
                  && (k->size(0) > 16 || decode_shape);
    if (validated) {
        const auto *slots = graph::lookup_bound_host_int_array(slot_mapping);
        validated = slots && slots->size() == slot_mapping->numel();
        if (validated) {
            const int64_t capacity = static_cast<int64_t>(k_cache->size(0) * k_cache->size(2));
            const char *cache_validation = std::getenv("INFINICORE_ASCEND_CACHE_SLOT_VALIDATION");
            if (cache_validation && std::strcmp(cache_validation, "1") == 0) {
                static thread_local ValidSlotsCache validated_slots;
                validated = validated_slots.validate(*slots, capacity);
            } else {
                std::unordered_set<int64_t> seen;
                seen.reserve(slots->size());
                for (const int64_t slot : *slots) {
                    if (slot < 0 || slot >= capacity || !seen.insert(slot).second) {
                        validated = false;
                        break;
                    }
                }
            }
        }
    }
    // QKV projections expose strided slices. ACLNN's Norm path requires
    // contiguous sources, so materialize only after validating all slots.
    Tensor k_work = validated ? k->contiguous() : k;
    Tensor v_work = validated ? v->contiguous() : v;
    size_t key = hash_combine(k_cache, v_cache, k_work, v_work, slot_mapping, validated);
    auto device = context::getDevice();
    auto &cache = caches.getCache(device);
    auto descriptor = cache.get(key).value_or(nullptr);
    if (!descriptor) {
        descriptor = std::make_shared<Descriptor>(nullptr);
        const auto create = validated ? infiniopCreatePagedCachingWithValidSlotsDescriptor
                                      : infiniopCreatePagedCachingDescriptor;
        INFINICORE_CHECK_ERROR(create(
            context::getInfiniopHandle(device), &descriptor->desc,
            k_cache->desc(), v_cache->desc(), k_work->desc(), v_work->desc(), slot_mapping->desc()));
        cache.put(key, descriptor);
    }

    INFINIOP_WORKSPACE_TENSOR(workspace, PagedCaching, descriptor);

    return new PlannedMeta{
        descriptor,
        graph::GraphTensor(workspace),
        graph::GraphTensor(k_cache),
        graph::GraphTensor(v_cache),
        graph::GraphTensor(k_work),
        graph::GraphTensor(v_work),
        graph::GraphTensor(slot_mapping),
        validated ? k_work : Tensor{},
        validated ? v_work : Tensor{},
        validated && context::isGraphRecording(),
        static_cast<int64_t>(k_cache->size(0) * k_cache->size(2))};
}

bool needs_slot_validation(void *planned_meta) {
    return reinterpret_cast<PlannedMeta *>(planned_meta)->validate_graph_slots;
}

void validate_slots_before_replay(void *planned_meta, graph::GraphReplayStage stage) {
    if (stage != graph::GraphReplayStage::PREPARE_UPDATE
        && stage != graph::GraphReplayStage::UPDATE) {
        return;
    }
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    const auto *slots = graph::lookup_bound_host_int_array(p->slot_mapping);
    static thread_local ValidSlotsCache replay_slots;
    if (!slots || slots->size() != p->slot_mapping->numel()
        || !replay_slots.validate(*slots, p->slot_capacity)) {
        throw std::runtime_error("PagedCaching graph requires fresh, unique, in-range host slot bindings before replay");
    }
}

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    INFINICORE_CHECK_ERROR(
        infiniopPagedCaching(
            p->descriptor->desc,
            p->workspace->data(),
            p->workspace->numel(),
            p->k_cache->data(),
            p->v_cache->data(),
            p->k->data(),
            p->v->data(),
            p->slot_mapping->data(),
            context::getStream()));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

INFINICORE_GRAPH_OP_REGISTER_ALLDEVICE(PagedCaching, &plan, &run, &cleanup);

} // namespace infinicore::op::paged_caching_impl::infiniop
