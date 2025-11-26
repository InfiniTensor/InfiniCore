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
        throw std::runtime_error("Cannot copy from tensor with different shape");
    }
    if (this->device() == src->device()) {

        // If both tensors are contiguous, use direct memcpy (much faster and avoids rearrange issues)
        if (this->is_contiguous() && src->is_contiguous()) {
            // Use nbytes() to get the actual tensor size
            size_t copy_size = std::min(this->nbytes(), src->nbytes());

            // For CPU-to-CPU copies, use regular memcpy. For device-to-device, use D2D memcpy
            if (this->device().getType() == Device::Type::CPU) {
                context::memcpyH2H(this->data(), src->data(), copy_size);
            } else {
                // Set context to the device for D2D operations
                context::setDevice(this->device());
                context::memcpyD2D(this->data(), src->data(), copy_size);
            }
        } else {
            // Set context to the device before rearrange
            context::setDevice(this->device());
            op::rearrange_(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()), src);
        }
    } else {
        if (!src->is_contiguous()) {
            src = src->contiguous();
        }
        if (this->device().getType() == Device::Type::CPU) {
            // Set context to source device before copying
            context::setDevice(src->device());

            // Use nbytes() to get the actual tensor size, not the full memory size
            size_t copy_size = std::min(this->nbytes(), src->nbytes());
            if (this->is_contiguous()) {
                // Verify device is still set to source device before memcpy
                Device current_device = context::getDevice();
                if (current_device != src->device()) {
                    context::setDevice(src->device());
                    current_device = context::getDevice();
                }
                context::memcpyD2H(this->data(), src->data(), copy_size);
            } else {
                auto local_src = Tensor::empty(this->shape(), this->dtype(), this->device());
                context::memcpyD2H(local_src->data(), src->data(), this->data_.memory->size());
                op::rearrange_(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()), local_src);
            }
        } else if (src->device().getType() == Device::Type::CPU) {
            // Use nbytes() to get the actual tensor size
            size_t copy_size = std::min(this->nbytes(), src->nbytes());

            // Set context to destination device before copying
            context::setDevice(this->device());

            if (this->is_contiguous()) {
                context::memcpyH2D(this->data(), src->data(), copy_size);
            } else {
                auto local_src = Tensor::empty(this->shape(), this->dtype(), this->device());
                context::memcpyH2D(local_src->data(), src->data(), copy_size);
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
