#include "infinicore/nn/parameter.hpp"

#include "infinicore/context/context.hpp"

#include <cstring>

namespace infinicore::nn {
Parameter::Parameter(
    const Shape &shape,
    const DataType &dtype,
    const Device &device)
    : Tensor(Tensor::empty(shape, dtype, device, false)) {
}

void Parameter::load_blob(const void *data) {
    auto buffer = Tensor::empty(impl_->shape(), impl_->dtype(), Device(Device::Type::CPU, 0), true);
    std::memcpy(buffer->data(), data, buffer->nbytes());
    infinicore::context::memcpyH2D(impl_->data(), buffer->data(), buffer->nbytes());
    infinicore::context::syncStream();
}
} // namespace infinicore::nn
