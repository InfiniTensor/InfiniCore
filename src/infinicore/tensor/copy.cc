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
    if (this->device() == src->device()) {
        context::setDevice(this->device());
        op::rearrange_(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()), src);
    } else {
        if (src->dtype() != this->dtype()) {
            throw std::runtime_error(
                "Cannot copy between devices with different dtypes. Src: " + src->info() + " Dst: " + this->info());
        }
        const size_t copy_size = this->nbytes();
        if (copy_size == 0) {
            return;
        }
        if (!src->is_contiguous()) {
            context::setDevice(src->device());
            src = src->contiguous();
        }

        if (this->device().getType() == Device::Type::CPU) {
            if (this->is_contiguous()) {
                context::setDevice(src->device());
                context::memcpyD2H(this->data(), src->data(), copy_size);
            } else {
                auto local_src = Tensor::empty(this->shape(), this->dtype(), this->device());
                context::setDevice(src->device());
                context::memcpyD2H(local_src->data(), src->data(), copy_size);
                context::setDevice(this->device());
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
            const auto dst_device = this->device();
            const auto src_device = src->device();
            if (dst_device.getType() != src_device.getType()) {
                throw std::runtime_error(
                    "Cannot peer-copy tensors between different accelerator types. Src: " + src->info() + " Dst: " + this->info());
            }

            auto dst = Tensor(const_cast<TensorImpl *>(this)->shared_from_this());
            auto contiguous_dst = this->is_contiguous()
                                    ? dst
                                    : Tensor::empty(this->shape(), this->dtype(), dst_device);
            try {
                // cudaMemcpyPeerAsync does not wait for work on another GPU.
                // Complete source work first, then enqueue and finish the copy
                // on the destination runtime before handing off the tensor.
                context::setDevice(src_device);
                context::syncStream();
                context::setDevice(dst_device);
                context::memcpyPeerD2D(
                    contiguous_dst->data(), src->data(), src_device, copy_size);
                context::syncStream();
            } catch (...) {
                context::setDevice(dst_device);
                throw;
            }

            if (!this->is_contiguous()) {
                op::rearrange_(dst, contiguous_dst);
                context::syncStream();
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
