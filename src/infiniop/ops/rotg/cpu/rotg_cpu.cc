#include "rotg_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

#include <cmath>

namespace op::rotg::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t y_desc,
    infiniopTensorDescriptor_t c_desc,
    infiniopTensorDescriptor_t s_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = RotgInfo::createRotgInfo(x_desc, y_desc, c_desc, s_desc);
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
infiniStatus_t calculateRotg(
    Tdata *x,
    Tdata *y,
    Tdata *c,
    Tdata *s) {

    using Tcompute = std::conditional_t<std::is_same_v<Tdata, double>, double, float>;

    const Tcompute zero = utils::cast<Tcompute>(0.0f);
    const Tcompute one = utils::cast<Tcompute>(1.0f);

    Tcompute x_val = utils::cast<Tcompute>(x[0]);
    Tcompute y_val = utils::cast<Tcompute>(y[0]);

    const Tcompute anorm = std::abs(x_val);
    const Tcompute bnorm = std::abs(y_val);

    if (bnorm == zero) {
        c[0] = utils::cast<Tdata>(one);
        s[0] = utils::cast<Tdata>(zero);
        y[0] = utils::cast<Tdata>(zero);
        return INFINI_STATUS_SUCCESS;
    }

    if (anorm == zero) {
        c[0] = utils::cast<Tdata>(zero);
        s[0] = utils::cast<Tdata>(one);
        x[0] = utils::cast<Tdata>(y_val);
        y[0] = utils::cast<Tdata>(one);
        return INFINI_STATUS_SUCCESS;
    }

    const Tcompute sigma = anorm > bnorm ? std::copysign(one, x_val) : std::copysign(one, y_val);
    const Tcompute r = sigma * std::hypot(x_val, y_val);
    const Tcompute c_val = x_val / r;
    const Tcompute s_val = y_val / r;

    Tcompute z;
    if (anorm > bnorm) {
        z = s_val;
    } else if (c_val != zero) {
        z = one / c_val;
    } else {
        z = one;
    }

    x[0] = utils::cast<Tdata>(r);
    y[0] = utils::cast<Tdata>(z);
    c[0] = utils::cast<Tdata>(c_val);
    s[0] = utils::cast<Tdata>(s_val);
    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_ROTG(TDATA) \
    calculateRotg((TDATA *)x, \
                  (TDATA *)y, \
                  (TDATA *)c, \
                  (TDATA *)s)

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    void *x,
    void *y,
    void *c,
    void *s,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.data_type) {
    case INFINI_DTYPE_F16:
        return CALCULATE_ROTG(fp16_t);
    case INFINI_DTYPE_F32:
        return CALCULATE_ROTG(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_ROTG(double);
    case INFINI_DTYPE_BF16:
        return CALCULATE_ROTG(bf16_t);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_ROTG

} // namespace op::rotg::cpu
