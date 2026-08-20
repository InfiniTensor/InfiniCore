#include "infinicore/ops/random_sample.hpp"

#include "../../utils.hpp"

#if defined(ENABLE_INFINIOPS_API)    \
    && (defined(ENABLE_NVIDIA_API)   \
        || defined(ENABLE_METAX_API) \
        || defined(ENABLE_HYGON_API) \
        || (defined(ENABLE_ILUVATAR_API) && defined(ENABLE_ATEN)))
#include "../infiniops_impl.hpp"

#include "base/argmax.h"
#endif

namespace infinicore::op {
namespace {

#if defined(ENABLE_INFINIOPS_API)    \
    && (defined(ENABLE_NVIDIA_API)   \
        || defined(ENABLE_METAX_API) \
        || defined(ENABLE_HYGON_API) \
        || (defined(ENABLE_ILUVATAR_API) && defined(ENABLE_ATEN)))
bool tryGreedyWithInfiniOps(Tensor indices, Tensor logits, int topk) {
    const auto dtype = logits->dtype();
    const auto device_type = logits->device().getType();
    if ((device_type != Device::Type::NVIDIA
         && device_type != Device::Type::METAX
         && device_type != Device::Type::ILUVATAR
         && device_type != Device::Type::HYGON)
        || topk != 1
        || logits->ndim() != 1
        || logits->numel() == 0
        || !logits->is_contiguous()
        || (dtype != DataType::F16 && dtype != DataType::BF16 && dtype != DataType::F32)
        || indices->numel() != 1
        || indices->dtype() != DataType::I64
        || !indices->is_contiguous()) {
        return false;
    }

    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    infini::ops::Config config;
    if (device_type != Device::Type::HYGON) {
        config.set_implementation_index(8);
    }
    const std::optional<int64_t> no_dim;
    infini::ops::Argmax::Call(
        handle,
        config,
        infiniops::TensorMeta(logits).tensor(logits),
        no_dim,
        false,
        infiniops::TensorMeta(indices).tensor(indices));
    return true;
}
#endif

} // namespace

common::OpDispatcher<RandomSample::schema> &RandomSample::dispatcher() {
    static common::OpDispatcher<RandomSample::schema> dispatcher_;
    return dispatcher_;
};

void RandomSample::execute(
    Tensor indices, Tensor logits,
    float random_val, float topp, int topk, float temperature) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(indices, logits);
    infinicore::context::setDevice(logits->device());
#if defined(ENABLE_INFINIOPS_API)    \
    && (defined(ENABLE_NVIDIA_API)   \
        || defined(ENABLE_METAX_API) \
        || defined(ENABLE_HYGON_API) \
        || (defined(ENABLE_ILUVATAR_API) && defined(ENABLE_ATEN)))
    if (tryGreedyWithInfiniOps(indices, logits, topk)) {
        return;
    }
#endif

    dispatcher().lookup(logits->device().getType())(
        indices, logits, random_val, topp, topk, temperature);
}

Tensor random_sample(
    Tensor logits,
    float random_val,
    float topp,
    int topk,
    float temperature) {
    auto indices = Tensor::empty({}, DataType::I32, logits->device());
    random_sample_(indices, logits, random_val, topp, topk, temperature);
    return indices;
}

void random_sample_(
    Tensor indices,
    Tensor logits,
    float random_val,
    float topp,
    int topk,
    float temperature) {
    RandomSample::execute(indices, logits, random_val, topp, topk, temperature);
}

} // namespace infinicore::op
