#include "blas_dot_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"

namespace op::blas_dot::metax {

struct Descriptor::Opaque {
    std::shared_ptr<device::metax::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t result_desc) {

    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    auto info = BlasDotInfo::createBlasDotInfo(x_desc, y_desc, result_desc);
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
    const void *x,
    const void *y,
    void *result,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;

    const size_t size = _info.getSize();
    const ptrdiff_t incx = _info.getIncx();
    const ptrdiff_t incy = _info.getIncy();
    const infiniDtype_t data_type = _info.getDtype();

    CHECK_STATUS(_opaque->internal->useMcblas(
        (hcStream_t)stream,
        [&](hcblasHandle_t handle) {
            CHECK_MCBLAS(hcblasSetPointerMode(handle, HCBLAS_POINTER_MODE_DEVICE));

            switch (data_type) {
            case INFINI_DTYPE_F32:
                CHECK_MCBLAS(hcblasSdot(handle, size, (const float *)x, incx, (const float *)y, incy, (float *)result));
                break;
            case INFINI_DTYPE_F64:
                CHECK_MCBLAS(hcblasDdot(handle, size, (const double *)x, incx, (const double *)y, incy, (double *)result));
                break;
            default:
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }

            return INFINI_STATUS_SUCCESS;
        }));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::blas_dot::metax
