#include "tensor.hpp"

namespace infinicore::py {

Tensor::Tensor(const infinicore::Tensor &tensor) : tensor_{tensor} {}

Tensor Tensor::to(const Device &device) const {
    return Tensor{tensor_->to(device)};
}

void Tensor::copy_(const Tensor &src) {
    return tensor_->copy_from(src.tensor_);
}

// Query methods
Shape Tensor::shape() const {
    return tensor_->shape();
}

Strides Tensor::strides() const {
    return tensor_->strides();
}

DataType Tensor::dtype() const {
    return tensor_->dtype();
}

Device Tensor::device() const {
    return tensor_->device();
}

bool Tensor::is_contiguous() const {
    return tensor_->is_contiguous();
}

Size Tensor::ndim() const {
    return tensor_->ndim();
}

Size Tensor::numel() const {
    return tensor_->numel();
}

bool Tensor::is_pinned() const {
    return tensor_->is_pinned();
}

std::string Tensor::info() const {
    return tensor_->info();
}

Tensor Tensor::narrow(const std::vector<TensorSliceParams> &slices) const {
    return Tensor{tensor_->narrow(slices)};
}

Tensor Tensor::permute(const Shape &shape) {
    return Tensor{tensor_->permute(shape)};
}

Tensor Tensor::view(const Shape &new_shape) const {
    return Tensor{tensor_->view(new_shape)};
}

Tensor Tensor::as_strided(const Shape &new_shape, const Strides &new_strides) const {
    return Tensor{tensor_->as_strided(new_shape, new_strides)};
}

Tensor empty(const Shape &shape,
             const DataType &dtype,
             const Device &device,
             bool pin_memory) {
    return Tensor{infinicore::Tensor::empty(shape, dtype, device, pin_memory)};
}

Tensor zeros(const Shape &shape,
             const DataType &dtype,
             const Device &device,
             bool pin_memory) {
    return Tensor{infinicore::Tensor::zeros(shape, dtype, device, pin_memory)};
}

Tensor ones(const Shape &shape,
            const DataType &dtype,
            const Device &device,
            bool pin_memory) {
    return Tensor{infinicore::Tensor::ones(shape, dtype, device, pin_memory)};
}

Tensor from_blob(uintptr_t raw_ptr,
                 Shape &shape,
                 const DataType &dtype,
                 const Device &device) {
    return Tensor{infinicore::Tensor::from_blob(reinterpret_cast<void *>(raw_ptr), shape, dtype, device)};
}

} // namespace infinicore::py
