#include <infinicore.hpp>

namespace infinicore {

Tensor::Tensor(const Shape &shape, const DataType &dtype, const Device &device) : shape_{shape}, dtype_{dtype}, device_{device} {}

const Tensor::Shape &Tensor::getShape() const {
    return shape_;
}

const DataType &Tensor::getDtype() const {
    return dtype_;
}

const Device &Tensor::getDevice() const {
    return device_;
}

} // namespace infinicore
