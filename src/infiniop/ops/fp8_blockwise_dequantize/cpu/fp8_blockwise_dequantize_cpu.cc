#include "fp8_blockwise_dequantize_cpu.h"

#include "../../../../utils/custom_types.h"
#include "../../../devices/cpu/cpu_handle.h"

#include <cmath>
#include <cstdint>
#include <limits>

namespace op::fp8_blockwise_dequantize::cpu {
namespace {

// Decode one FP8 E4M3FN byte: 1 sign bit, 4 exponent bits (bias 7), 3 mantissa
// bits. E4M3FN has no infinity; exponent 15 with mantissa 7 is NaN.
inline float decode_e4m3(uint8_t value) {
    const int exponent = (value >> 3) & 0xf;
    const int mantissa = value & 0x7;
    if (exponent == 0xf && mantissa == 0x7) {
        return std::numeric_limits<float>::quiet_NaN();
    }
    const float decoded = exponent == 0
                            ? std::ldexp(static_cast<float>(mantissa), -9)
                            : std::ldexp(1.0f + static_cast<float>(mantissa) * 0.125f, exponent - 7);
    return value & 0x80 ? -decoded : decoded;
}

template <typename T>
void dequantize(T *out,
                const uint8_t *q,
                const float *scales,
                const Fp8BlockwiseDequantizeInfo &info) {
#ifdef ENABLE_OMP
#pragma omp parallel for
#endif
    for (ptrdiff_t row = 0; row < static_cast<ptrdiff_t>(info.rows); ++row) {
        const size_t scale_row = row / info.block_rows;
        for (size_t col = 0; col < info.cols; ++col) {
            const size_t index = row * info.cols + col;
            const float scale = scales[scale_row * info.scales_cols + col / info.block_cols];
            out[index] = utils::cast<T>(decode_e4m3(q[index]) * scale);
        }
    }
}

} // namespace

struct Descriptor::Opaque {};

Descriptor::~Descriptor() { delete _opaque; }

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t scales_desc) {
    auto info = Fp8BlockwiseDequantizeInfo::create(out_desc, q_desc, scales_desc);
    CHECK_RESULT(info);
    *desc_ptr = new Descriptor(new Opaque{}, info.take(), handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void *, size_t, void *out,
    const void *q, const void *scales, void *) const {
    auto q_ptr = reinterpret_cast<const uint8_t *>(q);
    auto scales_ptr = reinterpret_cast<const float *>(scales);
    switch (_info.output_dtype) {
    case INFINI_DTYPE_F16:
        dequantize(reinterpret_cast<fp16_t *>(out), q_ptr, scales_ptr, _info);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_BF16:
        dequantize(reinterpret_cast<bf16_t *>(out), q_ptr, scales_ptr, _info);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F32:
        dequantize(reinterpret_cast<float *>(out), q_ptr, scales_ptr, _info);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::fp8_blockwise_dequantize::cpu
