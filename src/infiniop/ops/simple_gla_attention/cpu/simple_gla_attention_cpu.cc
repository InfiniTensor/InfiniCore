#include "simple_gla_attention_cpu.h"
#include "../../../handle.h"
#include "../../../tensor.h"
#include "../../../../utils.h"

#include <cmath>
#include <vector>

namespace op::simple_gla_attention::cpu {

namespace {

template <typename Tq>
void simple_gla_attention_cpu_run(
    Tq *out_ptr,
    const Tq *q_ptr,
    const Tq *k_ptr,
    const Tq *v_ptr,
    const float *g_ptr,
    float scale,
    size_t B,
    size_t T,
    size_t H,
    size_t D) {

    const size_t stride_b = T * H * D;
    const size_t stride_t = H * D;
    const size_t stride_h = D;

    std::vector<float> gate(H);
    for (size_t h = 0; h < H; ++h) {
        gate[h] = std::exp(g_ptr[h]);
    }

    std::vector<float> S(B * H * D * D, 0.f);

    for (size_t t = 0; t < T; ++t) {
        const size_t t_offset = t * stride_t;

        for (size_t b = 0; b < B; ++b) {
            const size_t b_offset = b * stride_b + t_offset;
            for (size_t h = 0; h < H; ++h) {
                const float g = gate[h];
                float *S_bh = S.data() + (b * H + h) * (D * D);

                for (size_t i = 0; i < D * D; ++i) {
                    S_bh[i] *= g;
                }

                for (size_t dk = 0; dk < D; ++dk) {
                    size_t qk_idx = b_offset + h * stride_h + dk;
                    float k_val = utils::cast<float>(k_ptr[qk_idx]);
                    for (size_t dv = 0; dv < D; ++dv) {
                        size_t qv_idx = b_offset + h * stride_h + dv;
                        float v_val = utils::cast<float>(v_ptr[qv_idx]);
                        S_bh[dk * D + dv] += k_val * v_val;
                    }
                }
            }
        }

        for (size_t b = 0; b < B; ++b) {
            const size_t b_offset = b * stride_b + t_offset;
            for (size_t h = 0; h < H; ++h) {
                const float *S_bh = S.data() + (b * H + h) * (D * D);
                for (size_t dv = 0; dv < D; ++dv) {
                    float acc = 0.f;
                    for (size_t dk = 0; dk < D; ++dk) {
                        size_t q_idx = b_offset + h * stride_h + dk;
                        float q_val = utils::cast<float>(q_ptr[q_idx]) * scale;
                        acc += q_val * S_bh[dk * D + dv];
                    }
                    size_t out_idx = b_offset + h * stride_h + dv;
                    out_ptr[out_idx] = utils::cast<Tq>(acc);
                }
            }
        }
    }
}

} // namespace

infiniStatus_t Descriptor::create(
    infiniopHandle_t handle,
    Descriptor **desc_ptr,
    infiniopTensorDescriptor_t out_desc,
    infiniopTensorDescriptor_t q_desc,
    infiniopTensorDescriptor_t k_desc,
    infiniopTensorDescriptor_t v_desc,
    infiniopTensorDescriptor_t g_gamma_desc) {

    if (desc_ptr == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }

    if (handle->device != INFINI_DEVICE_CPU) {
        return INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED;
    }

    const auto q_dtype = q_desc->dtype();
    if (q_dtype != INFINI_DTYPE_F32 && q_dtype != INFINI_DTYPE_F16 && q_dtype != INFINI_DTYPE_BF16) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    if (out_desc->dtype() != q_dtype || k_desc->dtype() != q_dtype || v_desc->dtype() != q_dtype) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
    if (g_gamma_desc->dtype() != INFINI_DTYPE_F32) {
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }

    if (q_desc->ndim() != 4 || !q_desc->isContiguous()) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }
    if (!out_desc->isContiguous() || !k_desc->isContiguous() || !v_desc->isContiguous() || !g_gamma_desc->isContiguous()) {
        return INFINI_STATUS_BAD_TENSOR_STRIDES;
    }

    auto qs = q_desc->shape();
    if (out_desc->shape() != qs || k_desc->shape() != qs || v_desc->shape() != qs) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    const size_t B = qs[0];
    const size_t T = qs[1];
    const size_t H = qs[2];
    const size_t D = qs[3];
    if (g_gamma_desc->ndim() != 1 || g_gamma_desc->shape()[0] != H) {
        return INFINI_STATUS_BAD_TENSOR_SHAPE;
    }

    *desc_ptr = new Descriptor(q_dtype, B, T, H, D, handle->device, handle->device_id);
    return INFINI_STATUS_SUCCESS;
}

infiniStatus_t Descriptor::calculate(
    void * /*workspace*/,
    size_t /*workspace_size*/,
    void *out,
    void const *q,
    void const *k,
    void const *v,
    void const *g_gamma,
    float scale,
    void * /*stream*/) const {

    if (out == nullptr || q == nullptr || k == nullptr || v == nullptr || g_gamma == nullptr) {
        return INFINI_STATUS_NULL_POINTER;
    }

    auto *g_ptr = reinterpret_cast<const float *>(g_gamma);

    switch (_q_dtype) {
    case INFINI_DTYPE_F32:
        simple_gla_attention_cpu_run(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            g_ptr,
            scale,
            _B,
            _T,
            _H,
            _D);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_F16:
        simple_gla_attention_cpu_run(
            reinterpret_cast<fp16_t *>(out),
            reinterpret_cast<const fp16_t *>(q),
            reinterpret_cast<const fp16_t *>(k),
            reinterpret_cast<const fp16_t *>(v),
            g_ptr,
            scale,
            _B,
            _T,
            _H,
            _D);
        return INFINI_STATUS_SUCCESS;
    case INFINI_DTYPE_BF16:
        simple_gla_attention_cpu_run(
            reinterpret_cast<bf16_t *>(out),
            reinterpret_cast<const bf16_t *>(q),
            reinterpret_cast<const bf16_t *>(k),
            reinterpret_cast<const bf16_t *>(v),
            g_ptr,
            scale,
            _B,
            _T,
            _H,
            _D);
        return INFINI_STATUS_SUCCESS;
    default:
        return INFINI_STATUS_BAD_TENSOR_DTYPE;
    }
}

} // namespace op::simple_gla_attention::cpu
