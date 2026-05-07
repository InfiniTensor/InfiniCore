#include "blas_dot_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::blas_dot::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t result_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = BlasDotInfo::createBlasDotInfo(x_desc, y_desc, result_desc);
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
infiniStatus_t calculateBlasDot(
    const BlasDotInfo &info,
    const Tdata *x,
    const Tdata *y,
    Tdata *result) {

    const size_t n = info.n;
    const ptrdiff_t incx = info.incx;
    const ptrdiff_t incy = info.incy;

    ptrdiff_t ix = (incx < 0) ? (1 - utils::cast<ptrdiff_t>(n)) * incx : 0;
    ptrdiff_t iy = (incy < 0) ? (1 - utils::cast<ptrdiff_t>(n)) * incy : 0;

    if constexpr (std::is_same<Tdata, fp16_t>::value || std::is_same<Tdata, bf16_t>::value) {
        float total = 0.0f;

        for (size_t i = 0; i < n; ++i) {
            total += utils::cast<float>(x[ix]) * utils::cast<float>(y[iy]);
            ix += incx;
            iy += incy;
        }

        result[0] = utils::cast<Tdata>(total);
    } else {
        Tdata total = utils::cast<Tdata>(0);

        for (size_t i = 0; i < n; ++i) {
            total += x[ix] * y[iy];
            ix += incx;
            iy += incy;
        }

        result[0] = total;
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_BLAS_DOT(TDATA)      \
    calculateBlasDot(_info,            \
                     (const TDATA *)x, \
                     (const TDATA *)y, \
                     (TDATA *)result)

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *x,
    const void *y,
    void *result,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.data_type) {
    case INFINI_DTYPE_F16:
        return CALCULATE_BLAS_DOT(fp16_t);
    case INFINI_DTYPE_F32:
        return CALCULATE_BLAS_DOT(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_BLAS_DOT(double);
    case INFINI_DTYPE_BF16:
        return CALCULATE_BLAS_DOT(bf16_t);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_BLAS_DOT

} // namespace op::blas_dot::cpu
