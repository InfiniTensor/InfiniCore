#pragma once

#include "device.hpp"
#include "dtype.hpp"
#include "memory.hpp"

#include <algorithm>
#include <cassert>
#include <functional>
#include <memory>
#include <vector>

#include <infiniop.h>
namespace infinicore {

using Size = std::size_t;
using Stride = std::ptrdiff_t;
using Shape = std::vector<Size>;
using Strides = std::vector<Stride>;

class TensorImpl;

struct TensorMetaData {
    Shape shape;
    Strides strides;
    DataType dtype;
};

struct TensorData {
    size_t offset;
    std::shared_ptr<Memory> memory;
};

struct SliceParams {
    size_t dim;
    size_t start;
    size_t len;
};

class Tensor {
public:
    static Tensor empty(const Shape &shape,
                        const DataType &dtype,
                        const Device &device);

    static Tensor zeros(const Shape &shape,
                        const DataType &dtype,
                        const Device &device);

    static Tensor ones(const Shape &shape,
                       const DataType &dtype,
                       const Device &device);

    Tensor(const Tensor &) = default;
    Tensor(Tensor &&) = default;
    Tensor &operator=(const Tensor &) = default;
    Tensor &operator=(Tensor &&) = default;

    TensorImpl *operator->();
    const TensorImpl *operator->() const;

protected:
    explicit Tensor(std::shared_ptr<TensorImpl> impl) : impl_(std::move(impl)) {}
    std::shared_ptr<TensorImpl> impl_;
    friend class TensorImpl;
};

class TensorImpl {

public:
    TensorImpl();

    std::byte *data();
    const std::byte *data() const;

    const Shape &shape() const;

    const Strides &strides() const;

    Size size(size_t dim) const;

    Stride stride(size_t dim) const;

    DataType dtype() const;

    Device device() const;

    infiniopTensorDescriptor_t desc() const;

    /**
     * Returns a new tensor that is a narrowed version of the current tensor.
     * The returned tensor shares the same underlying storage with the original tensor.
     *
     * @param slices A vector of slice parameters specifying the dimension, start index,
     *               and length for each dimension to narrow
     * @return A new tensor with narrowed dimensions
     *
     * Example:
     *   // Narrow dimension 0 from index 2 to 5 (length 3)
     *   // and dimension 1 from index 1 to 3 (length 2)
     *   tensor.narrow({{0, 2, 3}, {1, 1, 2}});
     */
    Tensor narrow(const std::vector<SliceParams> &slices) const;

    /**
     * Returns a new tensor with the dimensions permuted (reordered) according to the given order.
     * The returned tensor shares the same underlying storage with the original tensor.
     *
     * @param order The desired ordering of dimensions
     * @return A new tensor with permuted dimensions
     *
     * Example:
     *   // For a 3D tensor with shape [2, 3, 4], permute to [2, 0, 1]
     *   // This swaps the dimensions: dim0->dim2, dim1->dim0, dim2->dim1
     *   tensor->permute({2, 0, 1});
     */
    Tensor permute(const std::vector<size_t> &order) const;

    /**
     * Returns a new tensor with the same data but a different shape.
     * The returned tensor shares the same underlying storage with the original tensor.
     * The tensor is rearranged if the new shape is not compatible with the current shape.
     *
     * @param new_shape The desired new shape
     * @return A new tensor with the specified shape
     *
     * Example:
     *   // Reshape a 2x3 tensor (6 elements) to a 3x2 tensor
     *   tensor->view({3, 2});
     */
    Tensor view(const std::vector<size_t> &new_shape) const;

    /**
     * Insecurely returns a new tensor with the specified shape and contiguous strides.
     * The returned tensor shares the same underlying storage with the original tensor.
     *
     * @param new_shape The desired new shape
     * @return A new tensor with the specified shape and contiguous strides
     *
     * Example:
     *   // View a tensor as a new shape with contiguous memory layout
     *   tensor->view_as({4, 5});
     */
    Tensor view_as(const std::vector<size_t> &new_shape) const;

    /**
     * Insecurely returns a new tensor with the specified shape and strides.
     * The returned tensor shares the same underlying storage with the original tensor.
     *
     * @param new_shape The desired new shape
     * @param new_strides The desired new strides
     * @return A new tensor with the specified shape and strides
     *
     * Example:
     *   // Create a non-contiguous view with custom strides
     *   tensor->view_as({2, 3}, {6, 2}); // Stride of 6 for dim0, 2 for dim1
     */
    Tensor view_as(const std::vector<size_t> &new_shape, const std::vector<Stride> &new_strides) const;

protected:
    static std::shared_ptr<TensorImpl> empty(const Shape &shape, const DataType &dtype, const Device &device);
    static std::shared_ptr<TensorImpl> zeros(const Shape &shape, const DataType &dtype, const Device &device);
    static std::shared_ptr<TensorImpl> ones(const Shape &shape, const DataType &dtype, const Device &device);

    friend class Tensor;

private:
    TensorMetaData meta_;
    TensorData data_;
};

} // namespace infinicore
