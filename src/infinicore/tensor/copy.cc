#include "infinicore/context/context.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/tensor.hpp"

#include <spdlog/spdlog.h>

namespace infinicore {
Tensor TensorImpl::to(Device device) const {
    if (device == data_.memory->device()) {
        return Tensor(const_cast<TensorImpl *>(this)->shared_from_this());
    } else {
        std::shared_ptr<TensorImpl> _t = empty(meta_.shape, meta_.dtype, device, true);
        _t->copy_from(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()));
        return Tensor(_t);
    }
}

void TensorImpl::copy_from(Tensor src) {
    if (src->shape() != this->shape()) {
        throw std::runtime_error("Cannot copy from tensor with different shape");
    }
    if (this->device().getType() == src->device().getType()) {
        op::rearrange_(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()), src);
    } else {
        if (!src->is_contiguous()) {
            src = src->contiguous();
        }
        if (this->device().getType() == Device::Type::CPU) {
            if (this->is_contiguous()) {
                context::memcpyD2H(this->data(), src->data(), this->data_.memory->size());
            } else {
                auto local_src = Tensor::empty(this->shape(), this->dtype(), this->device());
                context::memcpyD2H(local_src->data(), src->data(), this->data_.memory->size());
                op::rearrange_(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()), local_src);
            }
        } else if (src->device().getType() == Device::Type::CPU) {
            if (this->is_contiguous()) {
                context::memcpyH2D(this->data(), src->data(), this->data_.memory->size());
            } else {
                auto local_src = Tensor::empty(this->shape(), this->dtype(), this->device());
                context::memcpyH2D(local_src->data(), src->data(), this->data_.memory->size());
                op::rearrange_(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()), local_src);
            }
        }
    }
}

Tensor TensorImpl::contiguous() const {
    if (is_contiguous()) {
        return Tensor(const_cast<TensorImpl *>(this)->shared_from_this());
    } else {
        return op::rearrange(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()));
    }
}

} // namespace infinicore
