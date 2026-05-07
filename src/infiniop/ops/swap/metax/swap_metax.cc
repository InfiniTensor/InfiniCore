#include "swap_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"

namespace op::swap::metax {

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
    infiniopTensorDescriptor_t y_desc) {

    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    auto result = SwapInfo::createSwapInfo(x_desc, y_desc);
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
    void *stream) const {

    (void)workspace;
    (void)workspace_size;

    const int n = utils::cast<int>(_info.n);
    const int incx = utils::cast<int>(_info.incx);
    const int incy = utils::cast<int>(_info.incy);
    const infiniDtype_t data_type = _info.data_type;

    CHECK_STATUS(_opaque->internal->useMcblas(
        (hcStream_t)stream,
        [&](hcblasHandle_t handle) {
            CHECK_MCBLAS(hcblasSetPointerMode(handle, HCBLAS_POINTER_MODE_DEVICE));

            switch (data_type) {
            case INFINI_DTYPE_F32:
                CHECK_MCBLAS(hcblasSswap(handle, n, (float *)x, incx, (float *)y, incy));
                break;
            case INFINI_DTYPE_F64:
                CHECK_MCBLAS(hcblasDswap(handle, n, (double *)x, incx, (double *)y, incy));
                break;
            default:
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }

            return INFINI_STATUS_SUCCESS;
        }));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::swap::metax
