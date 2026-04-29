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

    hpccDataType x_type, y_type, result_type;
    hpccDataType execution_type;

    switch (data_type) {
    case INFINI_DTYPE_F16:
        x_type = y_type = result_type = HPCC_R_16F;
        execution_type = HPCC_R_32F;
        break;
    case INFINI_DTYPE_BF16:
        x_type = y_type = result_type = HPCC_R_16BF;
        execution_type = HPCC_R_32F;
        break;
    case INFINI_DTYPE_F32:
        x_type = y_type = result_type = HPCC_R_32F;
        execution_type = HPCC_R_32F;
        break;
    case INFINI_DTYPE_F64:
        x_type = y_type = result_type = HPCC_R_64F;
        execution_type = HPCC_R_64F;
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    CHECK_STATUS(_opaque->internal->useMcblas(
        (hcStream_t)stream,
        [&](hcblasHandle_t handle) {
            CHECK_MCBLAS(hcblasSetPointerMode(handle, HCBLAS_POINTER_MODE_DEVICE));

            CHECK_MCBLAS(hcblasDotEx(
                handle,
                size,
                x,
                x_type,
                incx,
                y,
                y_type,
                incy,
                result,
                result_type,
                execution_type));

            return INFINI_STATUS_SUCCESS;
        }));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::blas_dot::metax
