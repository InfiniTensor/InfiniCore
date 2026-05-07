#include "blas_copy_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::blas_copy::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = BlasCopyInfo::createBlasCopyInfo(x_desc, y_desc);
    CHECK_RESULT(result);

    *desc_ptr = new Descriptor(
        result.take(),
        0,
        nullptr,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
infiniStatus_t calculateBlasCopy(
    const BlasCopyInfo &info,
    const Tdata *x,
    Tdata *y) {

    const size_t n = info.n;

    for (size_t i = 0; i < n; ++i) {
        ptrdiff_t x_idx = utils::cast<ptrdiff_t>(i) * info.incx;
        ptrdiff_t y_idx = utils::cast<ptrdiff_t>(i) * info.incy;
        y[y_idx] = x[x_idx];
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_BLAS_COPY(TDATA)      \
    calculateBlasCopy(_info,            \
                      (const TDATA *)x, \
                      (TDATA *)y)

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *x,
    void *y,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.data_type) {
    case INFINI_DTYPE_F16:
        return CALCULATE_BLAS_COPY(fp16_t);
    case INFINI_DTYPE_F32:
        return CALCULATE_BLAS_COPY(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_BLAS_COPY(double);
    case INFINI_DTYPE_BF16:
        return CALCULATE_BLAS_COPY(bf16_t);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_BLAS_COPY

} // namespace op::blas_copy::cpu
