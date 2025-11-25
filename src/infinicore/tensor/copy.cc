#include "infinicore/context/context.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/ops.hpp"
#include "infinicore/tensor.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <spdlog/spdlog.h>

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
    if (this->device().getType() == src->device().getType() && this->device().getIndex() == src->device().getIndex()) {
        SPDLOG_DEBUG("[COPY] copy_from: Same device type and index");

        // If both tensors are contiguous, use direct memcpy (much faster and avoids rearrange issues)
        if (this->is_contiguous() && src->is_contiguous()) {
            // Use nbytes() to get the actual tensor size
            size_t copy_size = std::min(this->nbytes(), src->nbytes());

            // For CPU-to-CPU copies, use regular memcpy. For device-to-device, use D2D memcpy
            if (this->device().getType() == Device::Type::CPU) {
                SPDLOG_DEBUG("[COPY] copy_from: Both contiguous on CPU, using regular memcpy, size={}", copy_size);
                std::memcpy(this->data(), src->data(), copy_size);
                SPDLOG_DEBUG("[COPY] copy_from: CPU memcpy completed");
            } else {
                // Set context to the device for D2D operations
                context::setDevice(this->device());
                SPDLOG_DEBUG("[COPY] copy_from: Both contiguous, using direct D2D memcpy, size={}", copy_size);
                context::memcpyD2D(this->data(), src->data(), copy_size);
                SPDLOG_DEBUG("[COPY] copy_from: D2D memcpy completed");
            }
        } else {
            SPDLOG_DEBUG("[COPY] copy_from: Not both contiguous, using rearrange_");
            // Set context to the device before rearrange
            context::setDevice(this->device());
            op::rearrange_(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()), src);
        }
    } else {
        SPDLOG_DEBUG("[COPY] copy_from: Different device types, preparing for cross-device copy");
        if (!src->is_contiguous()) {
            SPDLOG_DEBUG("[COPY] copy_from: Making src contiguous");
            src = src->contiguous();
        }
        if (this->device().getType() == Device::Type::CPU) {
            SPDLOG_DEBUG("[COPY] copy_from: Copying from device to CPU (D2H)");

            // Set context to source device before copying
            SPDLOG_DEBUG("[COPY] copy_from: Setting context to source device");
            context::setDevice(src->device());

            // Use nbytes() to get the actual tensor size, not the full memory size
            size_t copy_size = std::min(this->nbytes(), src->nbytes());
            SPDLOG_DEBUG("[COPY] copy_from: Copy size={} (dst nbytes={}, src nbytes={})",
                         copy_size, this->nbytes(), src->nbytes());

            if (this->is_contiguous()) {
                SPDLOG_DEBUG("[COPY] copy_from: Direct D2H copy, size={}", copy_size);
                SPDLOG_DEBUG("[COPY] copy_from: Source device: {}, Destination device: {}",
                             src->device().toString(), this->device().toString());
                // Verify device is still set to source device before memcpy
                Device current_device = context::getDevice();
                SPDLOG_DEBUG("[COPY] copy_from: Current context device before memcpy: {}", current_device.toString());
                if (current_device != src->device()) {
                    SPDLOG_DEBUG("[COPY] copy_from: Device was switched from {} to {}, restoring source device",
                                 src->device().toString(), current_device.toString());
                    context::setDevice(src->device());
                    current_device = context::getDevice();
                    SPDLOG_DEBUG("[COPY] copy_from: Current context device after restore: {}", current_device.toString());
                }
                SPDLOG_DEBUG("[COPY] copy_from: About to call context::memcpyD2H with source_device={}", src->device().toString());
                context::memcpyD2H(this->data(), src->data(), copy_size);
                SPDLOG_DEBUG("[COPY] copy_from: context::memcpyD2H completed");
            } else {
                SPDLOG_DEBUG("[COPY] copy_from: D2H copy with rearrange");
                // Save source device before creating CPU tensor (which will switch device to CPU)
                Device source_device = src->device();

                // Create CPU tensor - this will switch context to CPU
                auto local_src = Tensor::empty(this->shape(), this->dtype(), this->device());

                // CRITICAL: Restore source device for D2H memcpy (required for device-to-host operations)
                // The context::memcpyD2H will now automatically handle CPU runtime by switching to device runtime,
                // but we still restore the source device here for correctness
                context::setDevice(source_device);

                // Verify device is correct before memcpy
                Device verify_device = context::getDevice();
                if (verify_device != source_device) {
                    SPDLOG_WARN("[COPY] copy_from: Device verification failed! Expected {}, got {}. "
                                "memcpyD2H will automatically handle this by switching to device runtime.",
                                source_device.toString(), verify_device.toString());
                    // Don't throw - let the automatic fix in memcpyD2H handle it
                } else {
                    SPDLOG_DEBUG("[COPY] copy_from: Device verified as {} before D2H memcpy", source_device.toString());
                }

                // memcpyD2H will now automatically switch to device runtime if current is CPU
                context::memcpyD2H(local_src->data(), src->data(), copy_size);
                op::rearrange_(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()), local_src);
            }
        } else if (src->device().getType() == Device::Type::CPU) {
            SPDLOG_DEBUG("[COPY] copy_from: Copying from CPU to device (H2D)");
            SPDLOG_DEBUG("[COPY] copy_from: dst contiguous={}", this->is_contiguous());
            SPDLOG_DEBUG("[COPY] copy_from: dst data ptr={}, src data ptr={}",
                         static_cast<void *>(this->data()), static_cast<void *>(src->data()));

            // Use nbytes() to get the actual tensor size
            size_t copy_size = std::min(this->nbytes(), src->nbytes());
            SPDLOG_DEBUG("[COPY] copy_from: size={} (dst nbytes={}, src nbytes={})",
                         copy_size, this->nbytes(), src->nbytes());

            // Set context to destination device before copying
            SPDLOG_DEBUG("[COPY] copy_from: Setting context to destination device");
            context::setDevice(this->device());

            if (this->is_contiguous()) {
                SPDLOG_DEBUG("[COPY] copy_from: Direct H2D copy");
                context::memcpyH2D(this->data(), src->data(), copy_size);
                SPDLOG_DEBUG("[COPY] copy_from: H2D copy completed");
            } else {
                SPDLOG_DEBUG("[COPY] copy_from: H2D copy with rearrange");
                auto local_src = Tensor::empty(this->shape(), this->dtype(), this->device());
                context::memcpyH2D(local_src->data(), src->data(), copy_size);
                op::rearrange_(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()), local_src);
            }
        }
    }
    SPDLOG_DEBUG("[COPY] copy_from: Complete");
}

Tensor TensorImpl::contiguous() const {
    if (is_contiguous()) {
        return Tensor(const_cast<TensorImpl *>(this)->shared_from_this());
    } else {
        return op::rearrange(Tensor(const_cast<TensorImpl *>(this)->shared_from_this()));
    }
}

} // namespace infinicore
