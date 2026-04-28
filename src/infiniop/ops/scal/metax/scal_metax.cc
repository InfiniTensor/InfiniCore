#include "scal_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"

namespace op::scal::metax {

struct Descriptor::Opaque {
    std::shared_ptr<device::metax::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t alpha_desc,
    infiniopTensorDescriptor_t x_desc) {

    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    auto info = ScalInfo::createScalInfo(alpha_desc, x_desc);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        info.take(),
        0,
        new Opaque{handle->internal()},
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *alpha,
    void *x,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;

    const size_t size = _info.getSize();
    const ptrdiff_t incx = _info.getIncx();
    const infiniDtype_t data_type = _info.getDtype();

    CHECK_STATUS(_opaque->internal->useMcblas(
        (hcStream_t)stream,
        [&](hcblasHandle_t handle) {
            CHECK_MCBLAS(hcblasSetPointerMode(handle, HCBLAS_POINTER_MODE_DEVICE));

            switch (data_type) {
            case INFINI_DTYPE_F32:
                CHECK_MCBLAS(hcblasSscal(handle, size, (const float *)alpha, (float *)x, incx));
                break;
            case INFINI_DTYPE_F64:
                CHECK_MCBLAS(hcblasDscal(handle, size, (const double *)alpha, (double *)x, incx));
                break;
            default:
                return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
            }

            return INFINI_STATUS_SUCCESS;
        }));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::scal::metax
