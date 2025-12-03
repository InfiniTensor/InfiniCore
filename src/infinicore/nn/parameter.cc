#include "infinicore/nn/parameter.hpp"

#include "infinicore/context/context.hpp"

#include <cstring>
#include <stdexcept>

namespace infinicore::nn {
Parameter::Parameter()
    : Tensor(Tensor::empty({}, DataType::F32, Device(Device::Type::CPU, 0), false)) {
}

inline Shape get_partipion_shape_(const Shape &shape, Size tp_dim, Size tp_size) {
    if (tp_size <= 1) {
        return shape;
    }
    Shape part_shape = shape;
    if (tp_dim < shape.size()) {
        if (shape[tp_dim] % tp_size != 0) {
            throw std::runtime_error("Tensor dimension " + std::to_string(tp_dim) + " with size " + std::to_string(shape[tp_dim]) + " is not divisible by tensor parallel size " + std::to_string(tp_size) + ".");
        }
        part_shape[tp_dim] = shape[tp_dim] / tp_size;
    }
    return part_shape;
}

Parameter::Parameter(
    const Shape &shape,
    const DataType &dtype,
    const Device &device,
    Size tp_dim,
    Size tp_rank,
    Size tp_size)
    : Tensor(Tensor::empty(get_partipion_shape_(shape, tp_dim, tp_size), dtype, device, false)), tp_dim_(tp_dim), tp_rank_(tp_rank), tp_size_(tp_size) {
    if (tp_rank_ >= tp_size_) {
        throw std::runtime_error("Tensor parallel rank " + std::to_string(tp_rank_) + " must be less than tensor parallel size " + std::to_string(tp_size_) + ".");
    }
}

void Parameter::load_blob(const void *data) {
    Shape expected_shape = Shape(impl_->shape());
    expected_shape[tp_dim_] *= tp_size_;
    auto buffer = Tensor::empty(expected_shape, impl_->dtype(), Device(Device::Type::CPU, 0), true);
    std::memcpy(buffer->data(), data, buffer->nbytes());
    this->load(buffer);
}

void Parameter::load(const Tensor &tensor) {
    Shape expected_shape = Shape(impl_->shape());
    expected_shape[tp_dim_] *= tp_size_;

    if (expected_shape != tensor->shape()) {
        throw std::runtime_error("Shape mismatch when loading tensor into parameter.");
    }
    if (impl_->dtype() != tensor->dtype()) {
        throw std::runtime_error("Dtype mismatch when loading tensor into parameter.");
    }
    if (tp_size_ > 1) {
        impl_->copy_from(tensor->narrow({{tp_dim_, tp_rank_ * impl_->size(tp_dim_), impl_->size(tp_dim_)}}));

    } else {
        impl_->copy_from(tensor);
    }
    infinicore::context::syncStream();
}
} // namespace infinicore::nn
