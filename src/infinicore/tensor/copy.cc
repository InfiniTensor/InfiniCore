#include "infinicore/context/context.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/tensor.hpp"

#include <spdlog/spdlog.h>

namespace infinicore {
Tensor TensorImpl::to(Device device) const {
    if (device == data_.memory->device()) {
        auto _t = std::make_shared<TensorImpl>(meta_.shape, meta_.strides, meta_.dtype);
        _t->data_ = data_;
        return Tensor(_t);
    } else {
        std::shared_ptr<TensorImpl> _t = empty(meta_.shape, meta_.dtype, device, true);
        _t->copy_from(this);
        return Tensor(_t);
    }
}

void TensorImpl::copy_from(Tensor src) {
    copy_from(src.impl_.get());
}

void TensorImpl::copy_from(const TensorImpl *src) {
    if (!(this->is_contiguous() && src->is_contiguous())) {
        spdlog::error("Only contiguous tensors are supported for copy.");
        std::abort();
    }
    if (this->device() == Device::Type::CPU && src->device() == Device::Type::CPU) {
        context::memcpyH2H(this->data(), src->data(), this->data_.memory->size());
    } else if (this->device() == Device::Type::CPU) {
        context::memcpyD2H(this->data(), src->data(), this->data_.memory->size());
    } else if (src->device() == Device::Type::CPU) {
        context::memcpyH2D(this->data(), src->data(), this->data_.memory->size());
    } else {
        context::memcpyD2D(this->data(), src->data(), this->data_.memory->size());
    }
}

} // namespace infinicore
