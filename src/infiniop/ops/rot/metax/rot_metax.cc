#include "rot_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"

namespace op::rot::metax {

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
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t s_desc) {

    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    auto result = RotInfo::createRotInfo(x_desc, y_desc, c_desc, s_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        result.take(),
        0,
        new Opaque{handle->internal()},
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *x,
    void *y,
    const void *c,
    const void *s,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;

    const int n = utils::cast<int>(_info.n);
    const int incx = utils::cast<int>(_info.incx);
    const int incy = utils::cast<int>(_info.incy);
    const infiniDtype_t data_type = _info.data_type;

    hpccDataType x_type, y_type, cs_type;
    hpccDataType execution_type;

    switch (data_type) {
    case INFINI_DTYPE_F16:
        x_type = y_type = cs_type = HPCC_R_16F;
        execution_type = HPCC_R_32F;
        break;
    case INFINI_DTYPE_BF16:
        x_type = y_type = cs_type = HPCC_R_16BF;
        execution_type = HPCC_R_32F;
        break;
    case INFINI_DTYPE_F32:
        x_type = y_type = cs_type = HPCC_R_32F;
        execution_type = HPCC_R_32F;
        break;
    case INFINI_DTYPE_F64:
        x_type = y_type = cs_type = HPCC_R_64F;
        execution_type = HPCC_R_64F;
        break;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    CHECK_STATUS(_opaque->internal->useMcblas(
        (hcStream_t)stream,
        [&](hcblasHandle_t handle) {
            CHECK_MCBLAS(hcblasSetPointerMode(
                handle,
                HCBLAS_POINTER_MODE_DEVICE));

            CHECK_MCBLAS(hcblasRotEx(
                handle,
                n,
                x,
                x_type,
                incx,
                y,
                y_type,
                incy,
                c,
                s,
                cs_type,
                execution_type));

            return INFINI_STATUS_SUCCESS;
        }));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::rot::metax
