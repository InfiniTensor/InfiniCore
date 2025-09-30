#include "infinicore/tensor.hpp"
#include "infinicore/common/utils.hpp"
#include "infinicore/context/context.hpp"
#include "infinicore/dtype.hpp"

#include <spdlog/spdlog.h>

namespace {
// Helper function to calculate contiguous strides
inline infinicore::Strides calculate_contiguous_strides(const infinicore::Shape &shape) {
    infinicore::Strides strides(shape.size());
    infinicore::Stride stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}
} // namespace

namespace infinicore {
TensorImpl *Tensor::operator->() { return impl_.get(); }

const TensorImpl *Tensor::operator->() const { return impl_.get(); }

Tensor Tensor::empty(const Shape &shape,
                     const DataType &dtype,
                     const Device &device,
                     bool pin_memory) {
    return Tensor{TensorImpl::empty(shape, dtype, device, pin_memory)};
}

Tensor Tensor::zeros(const Shape &shape,
                     const DataType &dtype,
                     const Device &device,
                     bool pin_memory) {
    return Tensor{TensorImpl::zeros(shape, dtype, device, pin_memory)};
}

Tensor Tensor::ones(const Shape &shape,
                    const DataType &dtype,
                    const Device &device,
                    bool pin_memory) {
    return Tensor{TensorImpl::ones(shape, dtype, device, pin_memory)};
}

Tensor Tensor::from_blob(void *raw_ptr, const Shape &shape, const DataType &dtype, const Device &device) {
    return Tensor{TensorImpl::from_blob(raw_ptr, shape, dtype, device)};
}

TensorMetaData::TensorMetaData(const Shape &_shape, const Strides &_strides, const DataType &_dtype)
    : shape(_shape), strides(_strides), dtype(_dtype) {
    INFINICORE_CHECK_ERROR(infiniopCreateTensorDescriptor(&desc, shape.size(), shape.data(), strides.data(), (infiniDtype_t)dtype));
}

TensorImpl::TensorImpl(const Shape &shape, const DataType &dtype)
    : meta_(TensorMetaData(shape, calculate_contiguous_strides(shape), dtype)) {}

TensorImpl::TensorImpl(const Shape &shape, const Strides &strides, const DataType &dtype)
    : meta_(TensorMetaData(shape, strides, dtype)) {}

std::byte *TensorImpl::data() {
    return data_.memory->data() + data_.offset;
}

const std::byte *TensorImpl::data() const {
    return data_.memory->data() + data_.offset;
}

const Shape &TensorImpl::shape() const {
    return meta_.shape;
}

const Strides &TensorImpl::strides() const {
    return meta_.strides;
}

Size TensorImpl::ndim() const {
    return meta_.shape.size();
}

bool TensorImpl::is_contiguous() const {
    Stride expected_stride = 1;
    for (int i = meta_.shape.size() - 1; i >= 0; --i) {
        if (meta_.strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= meta_.shape[i];
    }
    return true;
}

Size TensorImpl::numel() const {
    Size total = 1;
    for (const auto &dim : meta_.shape) {
        total *= dim;
    }
    return total;
}

Size TensorImpl::size(size_t dim) const {
    return meta_.shape[dim];
}

Stride TensorImpl::stride(size_t dim) const {
    return meta_.strides[dim];
}

DataType TensorImpl::dtype() const {
    return meta_.dtype;
}

Device TensorImpl::device() const {
    return data_.memory->device();
}

infiniopTensorDescriptor_t TensorImpl::desc() const {
    return meta_.desc;
}

bool TensorImpl::is_pinned() const {
    return data_.memory->is_pinned();
}

std::string TensorImpl::info() const {
    std::stringstream ss;

    ss << "Tensor: "
       << "shape[ ";
    for (auto s : this->shape()) {
        ss << s << " ";
    }
    ss << "] strides[ ";
    for (auto s : this->strides()) {
        ss << s << " ";
    }
    ss << "] dtype=" << toString(this->dtype());

    return ss.str();
}

std::shared_ptr<TensorImpl> TensorImpl::empty(const Shape &shape,
                                              const DataType &dtype,
                                              const Device &device,
                                              bool pin_memory) {
    auto t = std::shared_ptr<TensorImpl>(new TensorImpl(shape, dtype));
    t->data_.offset = 0;

    context::setDevice(device);

    if (device == Device::Type::CPU) {
        if (pin_memory) {
            if (context::getDevice() == Device::Type::CPU) {
                spdlog::warn("Tensor memory is not pinned by any device with CPU runtime.");
                t->data_.memory = context::allocateHostMemory(t->numel() * dsize(dtype));
            } else {
                t->data_.memory = context::allocatePinnedHostMemory(t->numel() * dsize(dtype));
            }
        } else {
            t->data_.memory = context::allocateHostMemory(t->numel() * dsize(dtype));
        }
    } else {
        t->data_.memory = context::allocateMemory(t->numel() * dsize(dtype));
    }

    return t;
}

std::shared_ptr<TensorImpl> TensorImpl::zeros(const Shape &shape,
                                              const DataType &dtype,
                                              const Device &device,
                                              bool pin_memory) {
    // TODO: Implement this.
    return empty(shape, dtype, device, pin_memory);
}
std::shared_ptr<TensorImpl> TensorImpl::ones(const Shape &shape,
                                             const DataType &dtype,
                                             const Device &device,
                                             bool pin_memory) {
    // TODO: Implement this.
    return empty(shape, dtype, device, pin_memory);
}

std::shared_ptr<TensorImpl> TensorImpl::from_blob(
    void *raw_ptr,
    const Shape &shape,
    const DataType &dtype,
    const Device &device) {
    auto t = std::shared_ptr<TensorImpl>(new TensorImpl(shape, dtype));
    t->data_.offset = 0;
    t->data_.memory = std::make_shared<Memory>((std::byte *)raw_ptr, t->numel() * dsize(dtype), device, nullptr);
    return t;
}

} // namespace infinicore
