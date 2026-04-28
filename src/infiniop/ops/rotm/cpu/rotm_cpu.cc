#include "rotm_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::rotm::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t param_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto info = RotmInfo::createRotmInfo(x_desc, y_desc, param_desc);
    CHECK_RESULT(info);

    *desc_ptr = new Descriptor(
        info.take(),
        0,
        nullptr,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
infiniStatus_t calculateRotm(
    const RotmInfo &info,
    Tdata *x,
    Tdata *y,
    const Tdata *param) {

    using Tcompute = std::conditional_t<std::is_same_v<Tdata, double>, double, float>;

    const Tcompute zero = utils::cast<Tcompute>(0.0f);
    const Tcompute two = utils::cast<Tcompute>(2.0f);

    Tcompute sflag = utils::cast<Tcompute>(param[0]);

    if (info.getSize() == 0 || (sflag + two == zero)) {
        return INFINI_STATUS_SUCCESS;
    }

    const ptrdiff_t size = static_cast<ptrdiff_t>(info.getSize());
    const ptrdiff_t incx = info.getIncx();
    const ptrdiff_t incy = info.getIncy();
    const ptrdiff_t kx = incx >= 0 ? 0 : (size - 1) * (-incx);
    const ptrdiff_t ky = incy >= 0 ? 0 : (size - 1) * (-incy);

    Tcompute sh11 = zero;
    Tcompute sh12 = zero;
    Tcompute sh21 = zero;
    Tcompute sh22 = zero;

    if (incx == incy && incx > 0) {
        const ptrdiff_t nsteps = size * incx;
        if (sflag < zero) {
            sh11 = utils::cast<Tcompute>(param[1]);
            sh12 = utils::cast<Tcompute>(param[3]);
            sh21 = utils::cast<Tcompute>(param[2]);
            sh22 = utils::cast<Tcompute>(param[4]);
            for (ptrdiff_t i = 0; i < nsteps; i += incx) {
                const Tcompute w = utils::cast<Tcompute>(x[i]);
                const Tcompute z = utils::cast<Tcompute>(y[i]);
                x[i] = utils::cast<Tdata>(w * sh11 + z * sh12);
                y[i] = utils::cast<Tdata>(w * sh21 + z * sh22);
            }
        } else if (sflag == zero) {
            sh12 = utils::cast<Tcompute>(param[3]);
            sh21 = utils::cast<Tcompute>(param[2]);
            for (ptrdiff_t i = 0; i < nsteps; i += incx) {
                const Tcompute w = utils::cast<Tcompute>(x[i]);
                const Tcompute z = utils::cast<Tcompute>(y[i]);
                x[i] = utils::cast<Tdata>(w + z * sh12);
                y[i] = utils::cast<Tdata>(w * sh21 + z);
            }
        } else {
            sh11 = utils::cast<Tcompute>(param[1]);
            sh22 = utils::cast<Tcompute>(param[4]);
            for (ptrdiff_t i = 0; i < nsteps; i += incx) {
                const Tcompute w = utils::cast<Tcompute>(x[i]);
                const Tcompute z = utils::cast<Tcompute>(y[i]);
                x[i] = utils::cast<Tdata>(w * sh11 + z);
                y[i] = utils::cast<Tdata>(-w + sh22 * z);
            }
        }
    } else {
        ptrdiff_t ix = kx;
        ptrdiff_t iy = ky;

        if (sflag < zero) {
            sh11 = utils::cast<Tcompute>(param[1]);
            sh12 = utils::cast<Tcompute>(param[3]);
            sh21 = utils::cast<Tcompute>(param[2]);
            sh22 = utils::cast<Tcompute>(param[4]);
            for (ptrdiff_t i = 0; i < size; ++i) {
                const Tcompute w = utils::cast<Tcompute>(x[ix]);
                const Tcompute z = utils::cast<Tcompute>(y[iy]);
                x[ix] = utils::cast<Tdata>(w * sh11 + z * sh12);
                y[iy] = utils::cast<Tdata>(w * sh21 + z * sh22);
                ix += incx;
                iy += incy;
            }
        } else if (sflag == zero) {
            sh12 = utils::cast<Tcompute>(param[3]);
            sh21 = utils::cast<Tcompute>(param[2]);
            for (ptrdiff_t i = 0; i < size; ++i) {
                const Tcompute w = utils::cast<Tcompute>(x[ix]);
                const Tcompute z = utils::cast<Tcompute>(y[iy]);
                x[ix] = utils::cast<Tdata>(w + z * sh12);
                y[iy] = utils::cast<Tdata>(w * sh21 + z);
                ix += incx;
                iy += incy;
            }
        } else {
            sh11 = utils::cast<Tcompute>(param[1]);
            sh22 = utils::cast<Tcompute>(param[4]);
            for (ptrdiff_t i = 0; i < size; ++i) {
                const Tcompute w = utils::cast<Tcompute>(x[ix]);
                const Tcompute z = utils::cast<Tcompute>(y[iy]);
                x[ix] = utils::cast<Tdata>(w * sh11 + z);
                y[iy] = utils::cast<Tdata>(-w + sh22 * z);
                ix += incx;
                iy += incy;
            }
        }
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_ROTM(TDATA) \
    calculateRotm(_info,      \
                  (TDATA *)x, \
                  (TDATA *)y, \
                  (const TDATA *)param)

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *x,
    void *y,
    const void *param,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.getDtype()) {
    case INFINI_DTYPE_F16:
        return CALCULATE_ROTM(fp16_t);
    case INFINI_DTYPE_BF16:
        return CALCULATE_ROTM(bf16_t);
    case INFINI_DTYPE_F32:
        return CALCULATE_ROTM(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_ROTM(double);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_ROTM

} // namespace op::rotm::cpu
