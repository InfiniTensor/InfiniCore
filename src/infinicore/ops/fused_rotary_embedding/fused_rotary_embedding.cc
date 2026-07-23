#include "infinicore/ops/fused_rotary_embedding.hpp"

#include "../../utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/graph/graph.hpp"

#include <functional>
#include <memory>
#include <stdexcept>

#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
#include "infinicore/adaptor/aten_adaptor.hpp"
#include "infinicore/adaptor/vllm_iluvatar_adaptor.hpp"
#endif

namespace infinicore::op {
namespace {

class DeferredGraphOperator final : public graph::GraphOperator {
public:
    explicit DeferredGraphOperator(std::function<void()> runner)
        : runner_(std::move(runner)) {}

    void run() const override { runner_(); }

private:
    std::function<void()> runner_;
};

void record_or_run(std::function<void()> runner) {
    auto op = std::make_shared<DeferredGraphOperator>(std::move(runner));
    if (context::isGraphRecording()) {
        context::addGraphOperator(op);
    } else {
        op->run();
    }
}

void validate_fused_rotary_embedding(const Tensor &query,
                                     const Tensor &key,
                                     const Tensor &positions,
                                     int64_t head_size,
                                     const Tensor &cos_sin_cache) {
    if (!query || !key || !positions || !cos_sin_cache) {
        throw std::runtime_error(
            "fused_rotary_embedding expects non-empty query, key, positions, and cos_sin_cache tensors");
    }
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(query, key, positions, cos_sin_cache);
    if (query->dtype() != key->dtype() || query->dtype() != cos_sin_cache->dtype()) {
        throw std::runtime_error(
            "fused_rotary_embedding expects query, key, and cos_sin_cache to have the same dtype");
    }
    if (query->dtype() != DataType::F16 && query->dtype() != DataType::BF16) {
        throw std::runtime_error(
            "fused_rotary_embedding expects float16 or bfloat16 query and key tensors");
    }
    if (positions->dtype() != DataType::I64) {
        throw std::runtime_error("fused_rotary_embedding expects int64 positions");
    }
    if (query->ndim() != 3 || key->ndim() != 3 || positions->ndim() != 1
        || cos_sin_cache->ndim() != 2) {
        throw std::runtime_error(
            "fused_rotary_embedding expects query/key [tokens,heads,dim], positions [tokens], and cache [positions,dim]");
    }
    if (head_size <= 0
        || query->size(0) != key->size(0)
        || query->size(0) != positions->numel()
        || query->size(2) != static_cast<size_t>(head_size)
        || key->size(2) != static_cast<size_t>(head_size)
        || cos_sin_cache->size(1) != static_cast<size_t>(head_size)) {
        throw std::runtime_error(
            "fused_rotary_embedding tensor shapes are incompatible: head_size="
            + std::to_string(head_size)
            + ", query=[" + std::to_string(query->size(0)) + ","
            + std::to_string(query->size(1)) + "," + std::to_string(query->size(2))
            + "], key=[" + std::to_string(key->size(0)) + ","
            + std::to_string(key->size(1)) + "," + std::to_string(key->size(2))
            + "], positions_numel=" + std::to_string(positions->numel())
            + ", cache=[" + std::to_string(cos_sin_cache->size(0)) + ","
            + std::to_string(cos_sin_cache->size(1)) + "]");
    }
}

void run_fused_rotary_embedding(Tensor query,
                                Tensor key,
                                const Tensor &positions,
                                int64_t head_size,
                                const Tensor &cos_sin_cache,
                                bool is_neox) {
#if defined(ENABLE_ATEN) && defined(ENABLE_ILUVATAR_API)
    if (query->device().getType() == Device::Type::ILUVATAR) {
        auto query_at = adaptor::to_aten_tensor(query);
        auto key_at = adaptor::to_aten_tensor(key);
        auto positions_at = adaptor::to_aten_tensor(positions);
        auto cos_sin_cache_at = adaptor::to_aten_tensor(cos_sin_cache);
        adaptor::vllm_iluvatar::rotary_embedding(
            positions_at,
            query_at,
            std::optional<at::Tensor>(key_at),
            head_size,
            cos_sin_cache_at,
            is_neox);
        return;
    }
#endif

    throw std::runtime_error(
        "fused_rotary_embedding currently supports only Iluvatar builds with ATen");
}

} // namespace

void fused_rotary_embedding_(Tensor query,
                             Tensor key,
                             const Tensor &positions,
                             int64_t head_size,
                             const Tensor &cos_sin_cache,
                             bool is_neox) {
    validate_fused_rotary_embedding(query, key, positions, head_size, cos_sin_cache);
    record_or_run([query, key, positions, head_size, cos_sin_cache, is_neox] {
        run_fused_rotary_embedding(
            query,
            key,
            positions,
            head_size,
            cos_sin_cache,
            is_neox);
    });
}

} // namespace infinicore::op
