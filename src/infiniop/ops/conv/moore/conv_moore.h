#ifndef __CONV_MOORE_H__
#define __CONV_MOORE_H__

#include "conv_mudnn.h"

namespace op::conv::moore {

// Descriptor class for CONV operations on Moore devices.
// This class acts as a wrapper to select mudnn backend.
// It encapsulates the backend-specific Descriptor implementation and provides
// a unified interface for workspace query and CONV calculation.
class Descriptor final : public InfiniopDescriptor {
public:
    // Destructor: deletes the backend-specific descriptor.
    ~Descriptor() {
        delete reinterpret_cast<mudnn::Descriptor *>(_impl);
    }

    // Returns the required workspace size for the CONV operation.
    size_t workspaceSize() const {
        return reinterpret_cast<mudnn::Descriptor *>(_impl)->workspaceSize();
    }

    // Static factory method to create a Descriptor instance.
    // This method chooses the backend (mudnn) and constructs
    // the corresponding implementation internally.
    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t y_desc,
        infiniopTensorDescriptor_t x_desc,
        infiniopTensorDescriptor_t w_desc,
        infiniopTensorDescriptor_t b_desc,
        const void *pads,
        const void *strides,
        const void *dilations,
        size_t n) {
        auto desc = new Descriptor(handle->device, handle->device_id);

        // Backend selection strategy:
        // Currently defaulting to MUDNN.
        // Can be modified to choose based on environment variables or runtime parameters.
        desc->_backend = Backend::MUDNN;

        mudnn::Descriptor *impl;
        auto status = mudnn::Descriptor::create(handle, &impl, y_desc, x_desc, w_desc, b_desc, pads, strides, dilations, n);
        if (status != INFINI_STATUS_SUCCESS) {
            delete desc;
            return status;
        }
        desc->_impl = impl;

        *desc_ptr = desc;
        return INFINI_STATUS_SUCCESS;
    }

    // Unified CONV calculation interface.
    // Calls the corresponding backend's calculate function internally.
    infiniStatus_t calculate(
        void *workspace, size_t workspace_size,
        void *y,
        const void *x,
        const void *w,
        const void *bias,
        void *stream) const {
        return reinterpret_cast<mudnn::Descriptor *>(_impl)
            ->calculate(workspace, workspace_size, y, x, w, bias, stream);
    }

private:
    // Private constructor: ensures users cannot directly instantiate Descriptor.
    // Instances must be created via the static create() factory method.
    Descriptor(infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id}, _impl(nullptr) {}

    // Enum to indicate which backend is being used internally.
    enum class Backend { MUDNN };

    Backend _backend; // Currently selected MUDNN backend
    void *_impl;      // Pointer to backend-specific descriptor (mudnn::Descriptor*)
};

} // namespace op::conv::moore

#endif // __CONV_MOORE_H__