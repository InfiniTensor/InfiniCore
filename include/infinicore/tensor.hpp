#ifndef __INFINICORE_TENSOR_API_HPP__
#define __INFINICORE_TENSOR_API_HPP__

#include <vector>

#include "device.hpp"
#include "dtype.hpp"

namespace infinicore {

class Tensor {
public:
    using Size = std::size_t;

    using Stride = std::ptrdiff_t;

    using Shape = std::vector<Size>;

    using Strides = std::vector<Stride>;

    Tensor(const Shape &shape, const DataType &dtype, const Device &device);

    const Shape &get_shape() const;

    const DataType &get_dtype() const;

    const Device &get_device() const;

private:
    Shape shape_;

    DataType dtype_;

    Device device_;
};

} // namespace infinicore

#endif
