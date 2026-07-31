#include "infinicore/context/context.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/tensor.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
namespace infinicore {
Tensor TensorImpl::to(Device device) const {
    if (device == data_.memory->device()) {
        return Tensor(const_cast<TensorImpl *>(this)->shared_from_this());
    } else {
        std::shared_ptr<TensorImpl> _t = empty(meta_.shape, meta_.dtype, device);
        _t->copy_from(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()));
        return Tensor(_t);
    }
}

void TensorImpl::copy_from(Tensor src) {
    if (src->shape() != this->shape()) {
        throw std::runtime_error(
            "Cannot copy from tensor with different shape. Src: " + src->info() + " Dst: " + this->info());
    }
    if (src->dtype() != this->dtype()) {
        throw std::runtime_error(
            "Cannot copy from tensor with different dtype. Src: " + src->info() + " Dst: " + this->info());
    }
    if (src->nbytes() != this->nbytes()) {
        throw std::runtime_error(
            "Cannot copy from tensor with different byte size. Src: " + src->info() + " Dst: " + this->info());
    }
    if (this->device() == src->device()) {
        op::rearrange_(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()), src);
    } else {
        if (!src->is_contiguous()) {
            src = src->contiguous();
        }

        const size_t copy_size = this->nbytes();
        if (this->device().getType() == Device::Type::CPU) {
            if (this->is_contiguous()) {
                context::setDevice(src->device());
                context::memcpyD2H(this->data(), src->data(), copy_size, false);
            } else {
                auto local_src = Tensor::empty(this->shape(), this->dtype(), this->device());
                context::setDevice(src->device());
                context::memcpyD2H(local_src->data(), src->data(), copy_size, false);
                op::rearrange_(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()), local_src);
            }
        } else if (src->device().getType() == Device::Type::CPU) {
            context::setDevice(this->device());
            if (this->is_contiguous()) {
                context::memcpyH2D(this->data(), src->data(), copy_size);
            } else {
                auto local_src = Tensor::empty(this->shape(), this->dtype(), this->device());
                context::memcpyH2D(local_src->data(), src->data(), copy_size);
                op::rearrange_(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()), local_src);
            }
        } else {
            if (this->device().getType() != src->device().getType()) {
                throw std::runtime_error(
                    "Cannot copy directly between different accelerator backends. Src: " + src->info() + " Dst: " + this->info());
            }
            context::setDevice(this->device());
            if (this->is_contiguous()) {
                context::memcpyD2D(this->data(), src->data(), copy_size);
            } else {
                auto local_src = Tensor::empty(this->shape(), this->dtype(), this->device());
                context::memcpyD2D(local_src->data(), src->data(), copy_size);
                op::rearrange_(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()), local_src);
            }
        }
    }
}

void TensorImpl::copy_from_async(Tensor src) {
    if (src->shape() != this->shape()) {
        throw std::runtime_error(
            "Cannot copy asynchronously from tensor with different shape. Src: " + src->info() + " Dst: " + this->info());
    }
    if (src->dtype() != this->dtype()) {
        throw std::runtime_error(
            "Cannot copy asynchronously from tensor with different dtype. Src: " + src->info() + " Dst: " + this->info());
    }
    if (src->nbytes() != this->nbytes()) {
        throw std::runtime_error(
            "Cannot copy asynchronously from tensor with different byte size. Src: " + src->info() + " Dst: " + this->info());
    }

    if (this->device() == src->device()) {
        op::rearrange_(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()), src);
        return;
    }
    if (!this->is_contiguous() || !src->is_contiguous()) {
        throw std::runtime_error(
            "Asynchronous cross-device copy requires contiguous tensors. Src: " + src->info() + " Dst: " + this->info());
    }

    const size_t copy_size = this->nbytes();
    if (this->device().getType() == Device::Type::CPU) {
        if (!this->is_pinned()) {
            throw std::runtime_error("Asynchronous D2H copy requires pinned destination memory");
        }
        context::setDevice(src->device());
        context::memcpyD2H(this->data(), src->data(), copy_size, true);
    } else if (src->device().getType() == Device::Type::CPU) {
        if (!src->is_pinned()) {
            throw std::runtime_error("Asynchronous H2D copy requires pinned source memory");
        }
        context::setDevice(this->device());
        context::memcpyH2D(this->data(), src->data(), copy_size, true);
    } else {
        if (this->device().getType() != src->device().getType()) {
            throw std::runtime_error(
                "Cannot copy asynchronously between different accelerator backends. Src: " + src->info() + " Dst: " + this->info());
        }
        context::setDevice(this->device());
        context::memcpyD2D(this->data(), src->data(), copy_size, true);
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
