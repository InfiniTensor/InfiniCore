#include "infinicore/ops/softmax.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/softmax_infinilm.h"

#include <optional>

namespace infinicore::op::softmax_impl::infiniops {
namespace {

using TensorMeta = ::infinicore::op::infiniops::TensorMeta;

void calculate(Tensor output, Tensor input, int axis) {
    INFINICORE_ASSERT(output->device().getType() == Device::Type::NVIDIA);
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);

    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    infini::ops::Config config;

    TensorMeta output_meta(output);
    TensorMeta input_meta(input);
    infini::ops::SoftmaxInfinilm::Call(
        handle,
        config,
        input_meta.tensor(input),
        static_cast<int64_t>(axis),
        std::optional<infini::ops::DataType>{},
        output_meta.tensor(output));
}

} // namespace

static bool registered = []() {
    Softmax::dispatcher().registerDevice(Device::Type::NVIDIA, &calculate);
    return true;
}();

} // namespace infinicore::op::softmax_impl::infiniops
#endif
