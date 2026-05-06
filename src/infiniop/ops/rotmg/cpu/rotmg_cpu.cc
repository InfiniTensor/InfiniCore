#include "rotmg_cpu.h"
#include "../../../devices/cpu/common_cpu.h"

#include <cmath>

namespace op::rotmg::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t d1_desc,
    infiniopTensorDescriptor_t d2_desc,
    infiniopTensorDescriptor_t x1_desc,
    infiniopTensorDescriptor_t y1_desc,
    infiniopTensorDescriptor_t param_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = RotmgInfo::createRotmgInfo(d1_desc, d2_desc, x1_desc, y1_desc, param_desc);
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
infiniStatus_t calculateRotmg(
    Tdata *d1,
    Tdata *d2,
    Tdata *x1,
    const Tdata *y1,
    Tdata *param) {

    using Tcompute = std::conditional_t<std::is_same_v<Tdata, double>, double, float>;

    const Tcompute zero = utils::cast<Tcompute>(0.0f);
    const Tcompute one = utils::cast<Tcompute>(1.0f);
    const Tcompute two = utils::cast<Tcompute>(2.0f);
    const Tcompute gam = utils::cast<Tcompute>(4096.0f);
    const Tcompute gamsq = utils::cast<Tcompute>(1.67772e7f);
    const Tcompute rgamsq = utils::cast<Tcompute>(5.96046e-8f);

    Tcompute d1_val = utils::cast<Tcompute>(d1[0]);
    Tcompute d2_val = utils::cast<Tcompute>(d2[0]);
    Tcompute x1_val = utils::cast<Tcompute>(x1[0]);
    const Tcompute y1_val = utils::cast<Tcompute>(y1[0]);

    Tcompute sflag;
    Tcompute sh11 = zero;
    Tcompute sh12 = zero;
    Tcompute sh21 = zero;
    Tcompute sh22 = zero;

    if (d1_val < zero) {
        sflag = -one;
        d1_val = zero;
        d2_val = zero;
        x1_val = zero;
    } else {
        const Tcompute sp2 = d2_val * y1_val;
        if (sp2 == zero) {
            param[0] = utils::cast<Tdata>(-two);
            return INFINI_STATUS_SUCCESS;
        }

        const Tcompute sp1 = d1_val * x1_val;
        const Tcompute sq2 = sp2 * y1_val;
        const Tcompute sq1 = sp1 * x1_val;

        if (std::abs(sq1) > std::abs(sq2)) {
            sh21 = -y1_val / x1_val;
            sh12 = sp2 / sp1;
            const Tcompute su = one - sh12 * sh21;

            if (su > zero) {
                sflag = zero;
                d1_val = d1_val / su;
                d2_val = d2_val / su;
                x1_val = x1_val * su;
            } else {
                sflag = -one;
                sh11 = zero;
                sh12 = zero;
                sh21 = zero;
                sh22 = zero;
                d1_val = zero;
                d2_val = zero;
                x1_val = zero;
            }
        } else {
            if (sq2 < zero) {
                sflag = -one;
                d1_val = zero;
                d2_val = zero;
                x1_val = zero;
            } else {
                sflag = one;
                sh11 = sp1 / sp2;
                sh22 = x1_val / y1_val;
                const Tcompute su = one + sh11 * sh22;
                const Tcompute stemp = d2_val / su;
                d2_val = d1_val / su;
                d1_val = stemp;
                x1_val = y1_val * su;
            }
        }

        if (d1_val != zero) {
            while (d1_val <= rgamsq || d1_val >= gamsq) {
                if (sflag == zero) {
                    sh11 = one;
                    sh22 = one;
                    sflag = -one;
                } else {
                    sh21 = -one;
                    sh12 = one;
                    sflag = -one;
                }
                if (d1_val <= rgamsq) {
                    d1_val = d1_val * gam * gam;
                    x1_val = x1_val / gam;
                    sh11 = sh11 / gam;
                    sh12 = sh12 / gam;
                } else {
                    d1_val = d1_val / (gam * gam);
                    x1_val = x1_val * gam;
                    sh11 = sh11 * gam;
                    sh12 = sh12 * gam;
                }
            }
        }

        if (d2_val != zero) {
            while (std::abs(d2_val) <= rgamsq || std::abs(d2_val) >= gamsq) {
                if (sflag == zero) {
                    sh11 = one;
                    sh22 = one;
                    sflag = -one;
                } else {
                    sh21 = -one;
                    sh12 = one;
                    sflag = -one;
                }
                if (std::abs(d2_val) <= rgamsq) {
                    d2_val = d2_val * gam * gam;
                    sh21 = sh21 / gam;
                    sh22 = sh22 / gam;
                } else {
                    d2_val = d2_val / (gam * gam);
                    sh21 = sh21 * gam;
                    sh22 = sh22 * gam;
                }
            }
        }
    }

    if (sflag < zero) {
        param[1] = utils::cast<Tdata>(sh11);
        param[2] = utils::cast<Tdata>(sh21);
        param[3] = utils::cast<Tdata>(sh12);
        param[4] = utils::cast<Tdata>(sh22);
    } else if (sflag == zero) {
        param[2] = utils::cast<Tdata>(sh21);
        param[3] = utils::cast<Tdata>(sh12);
    } else {
        param[1] = utils::cast<Tdata>(sh11);
        param[4] = utils::cast<Tdata>(sh22);
    }

    param[0] = utils::cast<Tdata>(sflag);
    d1[0] = utils::cast<Tdata>(d1_val);
    d2[0] = utils::cast<Tdata>(d2_val);
    x1[0] = utils::cast<Tdata>(x1_val);
    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_ROTMG(TDATA)        \
    calculateRotmg((TDATA *)d1,       \
                   (TDATA *)d2,       \
                   (TDATA *)x1,       \
                   (const TDATA *)y1, \
                   (TDATA *)param)

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
    (void)stream;

    switch (_info.data_type) {
    case INFINI_DTYPE_F16:
        return CALCULATE_ROTMG(fp16_t);
    case INFINI_DTYPE_F32:
        return CALCULATE_ROTMG(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_ROTMG(double);
    case INFINI_DTYPE_BF16:
        return CALCULATE_ROTMG(bf16_t);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_ROTMG

} // namespace op::rotmg::cpu
