#include "nrm2_cpu.h"
#include "../../../devices/cpu/common_cpu.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace op::nrm2::cpu {

Descriptor::~Descriptor() = default;

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle_,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t x_desc,
    infiniopTensorDescriptor_t result_desc) {

    auto handle = reinterpret_cast<device::cpu::Handle *>(handle_);
    auto result = Nrm2Info::createNrm2Info(x_desc, result_desc);
    CHECK_RESULT(result);

    // Create descriptor
    *desc_ptr = new Descriptor(
        result.take(),
        0,
        nullptr,
        handle->device,
        handle->device_id);

    return INFINI_STATUS_SUCCESS;
}

template <typename Tdata>
infiniStatus_t calculateNrm2(
    const Nrm2Info &info,
    const Tdata *x,
    Tdata *result) {

    using Tcompute = std::conditional_t<std::is_same_v<Tdata, double>, double, float>;

    const size_t n = info.n;
    const ptrdiff_t incx = info.incx;

    // Blue's scaling constants (float vs double)
    constexpr Tcompute tsml = [] {
        if constexpr (std::is_same_v<Tcompute, float>) {
            return Tcompute(0x1p-63f); // 2^-63
        } else {
            return Tcompute(0x1p-511); // 2^-511
        }
    }();
    constexpr Tcompute tbig = [] {
        if constexpr (std::is_same_v<Tcompute, float>) {
            return Tcompute(0x1p52f); // 2^52
        } else {
            return Tcompute(0x1p486); // 2^486
        }
    }();
    constexpr Tcompute ssml = [] {
        if constexpr (std::is_same_v<Tcompute, float>) {
            return Tcompute(0x1p75f); // 2^75
        } else {
            return Tcompute(0x1p600); // 2^600
        }
    }();
    constexpr Tcompute sbig = [] {
        if constexpr (std::is_same_v<Tcompute, float>) {
            return Tcompute(0x1p-76f); // 2^-76
        } else {
            return Tcompute(0x1p-601); // 2^-601
        }
    }();

    Tcompute scl = Tcompute(1);
    Tcompute sumsq = Tcompute(0);

    bool notbig = true;
    Tcompute asml = Tcompute(0);
    Tcompute amed = Tcompute(0);
    Tcompute abig = Tcompute(0);

    // 0-based index; handle negative stride
    ptrdiff_t ix = (incx < 0) ? (ptrdiff_t(1) - utils::cast<ptrdiff_t>(n)) * incx : 0;

    for (size_t i = 0; i < n; ++i) {
        Tcompute ax = std::abs(utils::cast<Tcompute>(x[ix]));

        if (ax > tbig) {
            const Tcompute y = ax * sbig;
            abig += y * y;
            notbig = false;
        } else if (ax < tsml) {
            if (notbig) {
                const Tcompute y = ax * ssml;
                asml += y * y;
            }
        } else {
            amed += ax * ax;
        }

        ix += incx;
    }

    if (abig > Tcompute(0)) {
        if (amed > Tcompute(0) || std::isinf(amed) || std::isnan(amed)) {
            abig += (amed * sbig) * sbig;
        }
        scl = Tcompute(1) / sbig;
        sumsq = abig;
    } else if (asml > Tcompute(0)) {
        if (amed > Tcompute(0) || std::isinf(amed) || std::isnan(amed)) {
            amed = std::sqrt(amed);
            asml = std::sqrt(asml) / ssml;

            const Tcompute ymin = std::min(amed, asml);
            const Tcompute ymax = std::max(amed, asml);

            scl = Tcompute(1);
            sumsq = (ymax * ymax) * (Tcompute(1) + (ymin / ymax) * (ymin / ymax));
        } else {
            scl = Tcompute(1) / ssml;
            sumsq = asml;
        }
    } else {
        scl = Tcompute(1);
        sumsq = amed;
    }

    result[0] = utils::cast<Tdata>(scl * std::sqrt(sumsq));
    return INFINI_STATUS_SUCCESS;
}

#define CALCULATE_NRM2(TDATA)       \
    calculateNrm2(_info,            \
                  (const TDATA *)x, \
                  (TDATA *)result)

infiniStatus_t Descriptor::calculate(
    void *workspace,
    size_t workspace_size,
    const void *x,
    void *result,
    void *stream) const {

    (void)workspace;
    (void)workspace_size;
    (void)stream;

    switch (_info.data_type) {
    case INFINI_DTYPE_F16:
        return CALCULATE_NRM2(fp16_t);
    case INFINI_DTYPE_F32:
        return CALCULATE_NRM2(float);
    case INFINI_DTYPE_F64:
        return CALCULATE_NRM2(double);
    case INFINI_DTYPE_BF16:
        return CALCULATE_NRM2(bf16_t);
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

#undef CALCULATE_NRM2

} // namespace op::nrm2::cpu
