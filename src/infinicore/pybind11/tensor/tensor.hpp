#pragma once
#include "infinicore.hpp"

#include "../device/device.hpp"
#include "../dtype/dtype.hpp"

namespace infinicore::py {

class Tensor {
public:
    Tensor(const infinicore::Tensor &tensor);
    Tensor to(const Device &device) const;
    void copy_(const Tensor &src);

    const infinicore::Tensor &get() const { return tensor_; }
    infinicore::Tensor &get() { return tensor_; }

    Shape shape() const;
    Strides strides() const;
    DataType dtype() const;
    Device device() const;
    bool is_contiguous() const;
    Size ndim() const;
    Size numel() const;
    bool is_pinned() const;
    std::string info() const;

    Tensor narrow(const std::vector<TensorSliceParams> &slices) const;
    Tensor permute(const Shape &shape);
    Tensor view(const Shape &new_shape) const;
    Tensor as_strided(const Shape &new_shape, const Strides &new_strides) const;

private:
    infinicore::Tensor tensor_;
};

Tensor empty(const Shape &shape,
             const DataType &dtype,
             const Device &device,
             bool pin_memory = false);

Tensor zeros(const Shape &shape,
             const DataType &dtype,
             const Device &device,
             bool pin_memory = false);

Tensor ones(const Shape &shape,
            const DataType &dtype,
            const Device &device,
            bool pin_memory = false);

Tensor from_blob(uintptr_t raw_ptr,
                 Shape &shape,
                 const DataType &dtype,
                 const Device &device);

} // namespace infinicore::py
