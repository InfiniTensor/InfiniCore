#if defined(ENABLE_HYGON_API) && defined(ENABLE_ATEN)
#include "infinicore/ops/rms_rotary_embedding.hpp"

#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/lightop_adaptor.hpp"

#include <c10/hip/HIPGuard.h>

#include <optional>

namespace infinicore::op::rms_rotary_embedding_impl::lightop_hygon {

struct PlannedMeta {
    graph::GraphTensor query;
    graph::GraphTensor key;
    graph::GraphTensor positions;
    graph::GraphTensor cos_sin_cache;
    graph::GraphTensor q_weight;
    graph::GraphTensor k_weight;
    int64_t head_size;
    bool is_neox;
    float epsilon;
};

void *plan(Tensor query,
           Tensor key,
           const Tensor &positions,
           int64_t head_size,
           const Tensor &cos_sin_cache,
           bool is_neox,
           const Tensor &q_weight,
           const Tensor &k_weight,
           float epsilon) {
    infinicore::adaptor::lightop::preload_rms_rotary_embedding();
    return new PlannedMeta{
        graph::GraphTensor(query),
        graph::GraphTensor(key),
        graph::GraphTensor(positions),
        graph::GraphTensor(cos_sin_cache),
        graph::GraphTensor(q_weight),
        graph::GraphTensor(k_weight),
        head_size,
        is_neox,
        epsilon};
}

void run(void *planned_meta) {
    c10::hip::HIPStreamGuard guard(infinicore::adaptor::get_hip_stream());
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);

    auto query = infinicore::adaptor::to_aten_tensor(p->query);
    auto key = infinicore::adaptor::to_aten_tensor(p->key);
    auto positions = infinicore::adaptor::to_aten_tensor(p->positions);
    auto cos_sin_cache = infinicore::adaptor::to_aten_tensor(p->cos_sin_cache);
    auto q_weight = infinicore::adaptor::to_aten_tensor(p->q_weight);
    auto k_weight = infinicore::adaptor::to_aten_tensor(p->k_weight);

    infinicore::adaptor::lightop::rms_rotary_embedding_fuse(
        positions,
        query,
        key,
        p->head_size,
        cos_sin_cache,
        p->is_neox,
        q_weight,
        k_weight,
        std::nullopt,
        std::nullopt,
        static_cast<double>(p->epsilon));
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

static bool registered = []() {
    if (!infinicore::adaptor::lightop::available()) {
        return false;
    }
    RMSRotaryEmbedding::plan_dispatcher().registerDevice(Device::Type::HYGON, &plan);
    RMSRotaryEmbedding::run_dispatcher().registerDevice(Device::Type::HYGON, &run);
    RMSRotaryEmbedding::cleanup_dispatcher().registerDevice(Device::Type::HYGON, &cleanup);
    return true;
}();

} // namespace infinicore::op::rms_rotary_embedding_impl::lightop_hygon

#endif // ENABLE_HYGON_API && ENABLE_ATEN
