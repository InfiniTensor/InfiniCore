#include "rot_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

namespace op::rot::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t s_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = RotInfo::createRotInfo(x_desc, y_desc, c_desc, s_desc);
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
infiniStatus_t calculateRot(
    const RotInfo &info,
    Tdata *x,
    Tdata *y,
    const Tdata *c,
    const Tdata *s) {

    using Tcompute = std::conditional_t<std::is_same_v<Tdata, double>, double, float>;

    const Tcompute c_val = utils::cast<Tcompute>(c[0]);
    const Tcompute s_val = utils::cast<Tcompute>(s[0]);

    const size_t n = info.n;
    const ptrdiff_t incx = info.incx;
    const ptrdiff_t incy = info.incy;

    if (n == 0) {
        return INFINI_STATUS_SUCCESS;
    }

    const ptrdiff_t ix = incx >= 0 ? 0 : utils::cast<ptrdiff_t>(n - 1) * (-incx);
    const ptrdiff_t iy = incy >= 0 ? 0 : utils::cast<ptrdiff_t>(n - 1) * (-incy);

    for (size_t i = 0; i < n; ++i) {
        const ptrdiff_t x_idx = ix + utils::cast<ptrdiff_t>(i) * incx;
        const ptrdiff_t y_idx = iy + utils::cast<ptrdiff_t>(i) * incy;

        const Tcompute x_val = utils::cast<Tcompute>(x[x_idx]);
        const Tcompute y_val = utils::cast<Tcompute>(y[y_idx]);
        const Tcompute temp = c_val * x_val + s_val * y_val;
        y[y_idx] = utils::cast<Tdata>(c_val * y_val - s_val * x_val);
        x[x_idx] = utils::cast<Tdata>(temp);
    }

    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_ROT(TDATA)       \
    calculateRot(_info,            \
                 (TDATA *)x,       \
                 (TDATA *)y,       \
                 (const TDATA *)c, \
                 (const TDATA *)s)

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
    (void)stream;

    switch (_info.data_type) {
    case INFINI_DTYPE_F16:
        return CALCULATE_ROT(fp16_t);
    case INFINI_DTYPE_F32:
        return CALCULATE_ROT(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_ROT(double);
    case INFINI_DTYPE_BF16:
        return CALCULATE_ROT(bf16_t);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_ROT

} // namespace op::rot::cpu
