#include <infinicore.hpp>

namespace infinicore {

Tensor::Tensor(const Shape &shape, const DataType &dtype, const Device &device) : shape_{shape}, dtype_{dtype}, device_{device} {}

const Tensor::Shape &Tensor::get_shape() const {
    return shape_;
}

const DataType &Tensor::get_dtype() const {
    return dtype_;
}

const Device &Tensor::get_device() const {
    return device_;
}

} // namespace infinicore
