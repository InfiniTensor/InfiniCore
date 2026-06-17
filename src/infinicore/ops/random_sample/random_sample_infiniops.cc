#include "infinicore/ops/random_sample.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/random_sample_infinilm.h"

namespace infinicore::op::random_sample_impl::infiniops {
namespace {

using TensorMeta = ::infinicore::op::infiniops::TensorMeta;

void calculate(Tensor indices, Tensor logits, float random_val, float topp, int topk, float temperature) {
    INFINICORE_ASSERT(indices->device().getType() == Device::Type::NVIDIA);
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(indices, logits);

    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    infini::ops::Config config;

    TensorMeta indices_meta(indices);
    TensorMeta logits_meta(logits);
    infini::ops::Operator<infini::ops::RandomSampleInfinilm>::Call(
        handle,
        config,
        logits_meta.tensor(logits),
        random_val,
        topp,
        static_cast<int64_t>(topk),
        temperature,
        indices_meta.tensor(indices));
}

} // namespace

static bool registered = []() {
    RandomSample::dispatcher().registerDevice(Device::Type::NVIDIA, &calculate);
    return true;
}();

} // namespace infinicore::op::random_sample_impl::infiniops
#endif
