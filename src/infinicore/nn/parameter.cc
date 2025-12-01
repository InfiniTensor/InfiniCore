#include "infinicore/nn/parameter.hpp"

#include "infinicore/context/context.hpp"

#include <cstring>

namespace infinicore::nn {
Parameter::Parameter()
    : Tensor(Tensor::empty({}, DataType::F32, Device(Device::Type::CPU, 0), false)) {
}

Parameter::Parameter(
    const Shape &shape,
    const DataType &dtype,
    const Device &device)
    : Tensor(Tensor::empty(shape, dtype, device, false)) {
}

void Parameter::load_blob(const void *data) {
    auto buffer = Tensor::empty(impl_->shape(), impl_->dtype(), Device(Device::Type::CPU, 0), true);
    std::memcpy(buffer->data(), data, buffer->nbytes());

    // If parameter is on CPU, use direct memcpy; otherwise use H2D
    if (impl_->device().getType() == Device::Type::CPU) {
        infinicore::context::memcpyH2H(impl_->data(), buffer->data(), buffer->nbytes());
    } else {
        infinicore::context::memcpyH2D(impl_->data(), buffer->data(), buffer->nbytes());
        infinicore::context::syncStream();
    }
}
} // namespace infinicore::nn
