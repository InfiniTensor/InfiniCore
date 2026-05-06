#include "rotmg_metax.h"
#include "../../../devices/metax/metax_common.h"
#include "../../../devices/metax/metax_handle.h"

namespace op::rotmg::metax {

struct Descriptor::Opaque {
    std::shared_ptr<device::metax::Handle::Internal> internal;
};

Descriptor::~Descriptor() {
    delete _opaque;
}

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t d1_desc,
    infiniopTensorDescriptor_t d2_desc,
    infiniopTensorDescriptor_t x1_desc,
    infiniopTensorDescriptor_t y1_desc,
    infiniopTensorDescriptor_t param_desc) {

    auto handle = reinterpret_cast<device::metax::Handle *>(handle_);
    auto result = RotmgInfo::createRotmgInfo(d1_desc, d2_desc, x1_desc, y1_desc, param_desc);
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
    void *d1,
    void *d2,
    void *x1,
    const void *y1,
    void *param,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;

    const infiniDtype_t data_type = _info.data_type;

    CHECK_STATUS(_opaque->internal->useMcblas(
        (hcStream_t)stream,
        [&](hcblasHandle_t handle) {
            CHECK_MCBLAS(hcblasSetPointerMode(handle, HCBLAS_POINTER_MODE_DEVICE));

            switch (data_type) {
            case INFINI_DTYPE_F32:
                CHECK_MCBLAS(hcblasSrotmg(handle, (float *)d1, (float *)d2, (float *)x1, (const float *)y1, (float *)param));
                break;
            case INFINI_DTYPE_F64:
                CHECK_MCBLAS(hcblasDrotmg(handle, (double *)d1, (double *)d2, (double *)x1, (const double *)y1, (double *)param));
                break;
            default:
                return INFINI_STATUS_BAD_TENSOR_DTYPE;
            }

            return INFINI_STATUS_SUCCESS;
        }));

    return INFINI_STATUS_SUCCESS;
}

} // namespace op::rotmg::metax
