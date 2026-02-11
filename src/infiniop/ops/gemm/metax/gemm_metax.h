#ifndef __GEMM_METAX_H__
#define __GEMM_METAX_H__

#include <cstdlib>
#include <string>
#include "hcdnn/gemm_hcdnn.h"
#include "hcblas/gemm_hcblas.h"

namespace op::gemm::metax {

// Descriptor class for GEMM operations on Metax devices.
// This class acts as a wrapper to select either hcdnn or hcblas backend.
// It encapsulates the backend-specific Descriptor implementation and provides
// a unified interface for workspace query and GEMM calculation.
class Descriptor final : public InfiniopDescriptor {
public:
    // Destructor: deletes the backend-specific descriptor.
    ~Descriptor() {
        if (_backend == Backend::HCBLAS) {
            delete reinterpret_cast<hcblas::Descriptor *>(_impl);
        } else {
            delete reinterpret_cast<hcdnn::Descriptor *>(_impl);
        }
    }

    // Returns the required workspace size for the GEMM operation.
    size_t workspaceSize() const {
        if (_backend == Backend::HCBLAS) {
            return reinterpret_cast<hcblas::Descriptor *>(_impl)->workspaceSize();
        } else {
            return reinterpret_cast<hcdnn::Descriptor *>(_impl)->workspaceSize();
        }
    }

    // Static factory method to create a Descriptor instance.
    // This method chooses the backend (hcdnn or hcblas) and constructs
    // the corresponding implementation internally.
    static infiniStatus_t create(
        infiniopHandle_t handle,
        Descriptor **desc_ptr,
        infiniopTensorDescriptor_t c_desc,
        infiniopTensorDescriptor_t a_desc,
        infiniopTensorDescriptor_t b_desc) {
        auto desc = new Descriptor(handle->device, handle->device_id);

        // Backend selection strategy:
        // Check environment variable INFINIOP_GEMM_METAX_BACKEND first (HCBLAS or HCDNN)
        // Otherwise default to HCDNN.
        const char* backend_env = std::getenv("INFINIOP_GEMM_METAX_BACKEND");
        if (backend_env != nullptr && std::string(backend_env) == "HCBLAS") {
            desc->_backend = Backend::HCBLAS;
        } else {
            desc->_backend = Backend::HCDNN;
        }

        if (desc->_backend == Backend::HCBLAS) {
            hcblas::Descriptor *impl;
            auto status = hcblas::Descriptor::create(handle, &impl, c_desc, a_desc, b_desc);
            if (status != INFINI_STATUS_SUCCESS) {
                delete desc;
                return status;
            }
            desc->_impl = impl;
        } else {
            hcdnn::Descriptor *impl;
            auto status = hcdnn::Descriptor::create(handle, &impl, c_desc, a_desc, b_desc);
            if (status != INFINI_STATUS_SUCCESS) {
                delete desc;
                return status;
            }
            desc->_impl = impl;
        }

        *desc_ptr = desc;
        return INFINI_STATUS_SUCCESS;
    }

    // Unified GEMM calculation interface.
    // Calls the corresponding backend's calculate function internally.
    infiniStatus_t calculate(
        void *workspace, size_t workspace_size,
        void *c, float beta,
        const void *a, const void *b,
        float alpha,
        void *stream) const {
        if (_backend == Backend::HCBLAS) {
            return reinterpret_cast<hcblas::Descriptor *>(_impl)
                ->calculate(workspace, workspace_size, c, beta, a, b, alpha, stream);
        } else {
            return reinterpret_cast<hcdnn::Descriptor *>(_impl)
                ->calculate(workspace, workspace_size, c, beta, a, b, alpha, stream);
        }
    }

private:
    // Private constructor: ensures users cannot directly instantiate Descriptor.
    // Instances must be created via the static create() factory method.
    Descriptor(infiniDevice_t device_type, int device_id)
        : InfiniopDescriptor{device_type, device_id}, _impl(nullptr) {}

    // Enum to indicate which backend is being used internally.
    enum class Backend { HCBLAS,
                         HCDNN };

    Backend _backend; // Currently selected HCBLAS/HCDNN backend
    void *_impl;      // Pointer to backend-specific descriptor (hcblas::Descriptor* or hcdnn::Descriptor*)
};

} // namespace op::gemm::metax

#endif // __GEMM_METAX_H__
