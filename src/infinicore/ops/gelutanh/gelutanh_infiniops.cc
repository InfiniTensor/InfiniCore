#include "infinicore/ops/gelutanh.hpp"

#ifdef ENABLE_INFINIOPS_API
#include "../infiniops_impl.hpp"

#include "base/gelutanh_infinilm.h"

namespace infinicore::op::gelutanh_impl::infiniops {
namespace {

using TensorMeta = ::infinicore::op::infiniops::TensorMeta;

void calculate(Tensor output, Tensor input) {
    INFINICORE_ASSERT(output->device().getType() == Device::Type::NVIDIA);
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(output, input);

    infini::ops::Handle handle;
    handle.set_stream(context::getStream());
    infini::ops::Config config;

    TensorMeta output_meta(output);
    TensorMeta input_meta(input);
    infini::ops::generated_dispatch::CallGelutanhInfinilm(
        handle,
        config,
        input_meta.tensor(input),
        output_meta.tensor(output));
}

} // namespace

static bool registered = []() {
    GeluTanh::dispatcher().registerDevice(Device::Type::NVIDIA, &calculate);
    return true;
}();

} // namespace infinicore::op::gelutanh_impl::infiniops
#endif
