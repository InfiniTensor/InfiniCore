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
    infiniopTensorDescriptor_t desc;

    TensorMetaData(const Shape &shape, const Strides &strides, const DataType &dtype);
};

struct TensorData {
    size_t offset;
    std::shared_ptr<Memory> memory;
};

struct TensorSliceParams {
    size_t dim;
    size_t start;
    Size len;
};

class Tensor {
public:
    static Tensor empty(const Shape &shape,
                        const DataType &dtype,
                        const Device &device,
                        bool pin_memory = false);

    static Tensor zeros(const Shape &shape,
                        const DataType &dtype,
                        const Device &device,
                        bool pin_memory = false);

    static Tensor ones(const Shape &shape,
                       const DataType &dtype,
                       const Device &device,
                       bool pin_memory = false);

    static Tensor from_blob(void *raw_ptr,
                            const Shape &shape,
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
    TensorImpl(const Shape &shape, const DataType &dtype);
    TensorImpl(const Shape &shape, const Strides &strides, const DataType &dtype);

    std::byte *data();
    const std::byte *data() const;

    const Shape &shape() const;

    const Strides &strides() const;

    bool is_contiguous() const;

    Size ndim() const;

    Size numel() const;

    Size size(size_t dim) const;

    Stride stride(size_t dim) const;

    DataType dtype() const;

    Device device() const;

    infiniopTensorDescriptor_t desc() const;

    bool is_pinned() const;

    std::string info() const;

    ///
    /// Data Transfer APIs
    ///

    /**
     * Returns a new tensor with the same data on a different device.
     * If the new device passed is same as the current device, the original tensor is returned.
     *
     * @param device The device of the new tensor
     *
     * @return A new tensor with the same data on the specified device
     */
    Tensor to(Device device) const;

    /**
     * Copy Data from another tensor to this tensor.
     * Currently, only contigous tensors of the same dtype and shape are supported.
     *
     * @param src The source tensor to copy from
     *
     * @return A new tensor with the same data on the specified device
     */
    void copy_from(Tensor src);

    ///
    /// View APIs
    ///

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
    Tensor narrow(const std::vector<TensorSliceParams> &slices) const;

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
    Tensor permute(const Shape &order) const;

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
    Tensor view(const Shape &new_shape) const;

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
     *   tensor->as_strided({2, 3}, {6, 2}); // Stride of 6 for dim0, 2 for dim1
     */
    Tensor as_strided(const Shape &new_shape, const Strides &new_strides) const;

protected:
    static std::shared_ptr<TensorImpl> empty(
        const Shape &shape,
        const DataType &dtype,
        const Device &device,
        bool pin_memory = false);

    static std::shared_ptr<TensorImpl> zeros(
        const Shape &shape,
        const DataType &dtype,
        const Device &device,
        bool pin_memory = false);

    static std::shared_ptr<TensorImpl> ones(
        const Shape &shape,
        const DataType &dtype,
        const Device &device,
        bool pin_memory = false);

    static std::shared_ptr<TensorImpl> from_blob(
        void *raw_ptr,
        const Shape &shape,
        const DataType &dtype,
        const Device &device);

    friend class Tensor;

private:
    TensorMetaData meta_;
    TensorData data_;

    void copy_from(const TensorImpl *src);
};

} // namespace infinicore
