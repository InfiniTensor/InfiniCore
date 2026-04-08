#include "infinicore/ops/simple_gla_attention.hpp"

#include "../../../utils.h"
#include "../../utils.hpp"
#include "infinicore/context/context.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace infinicore::op {

namespace {

// Read one element from tensor at flat index, convert to float.
template <typename T>
inline float read_float(const std::byte *ptr, size_t idx) {
    return static_cast<float>(*reinterpret_cast<const T *>(ptr + idx * sizeof(T)));
}

inline float read_float_at(const std::byte *ptr, size_t idx, DataType dtype) {
    switch (dtype) {
    case DataType::F32:
        return read_float<float>(ptr, idx);
    case DataType::F16:
        return _f16_to_f32(*reinterpret_cast<const fp16_t *>(ptr + idx * 2));
    case DataType::BF16:
        return _bf16_to_f32(*reinterpret_cast<const bf16_t *>(ptr + idx * 2));
    default:
        throw std::runtime_error("simple_gla_attention: unsupported dtype (need F32, F16, or BF16)");
    }
}

// Write one float to tensor at flat index.
inline void write_float_at(std::byte *ptr, size_t idx, DataType dtype, float val) {
    switch (dtype) {
    case DataType::F32:
        *reinterpret_cast<float *>(ptr + idx * 4) = val;
        break;
    case DataType::F16:
        *reinterpret_cast<fp16_t *>(ptr + idx * 2) = _f32_to_f16(val);
        break;
    case DataType::BF16:
        *reinterpret_cast<bf16_t *>(ptr + idx * 2) = _f32_to_bf16(val);
        break;
    default:
        throw std::runtime_error("simple_gla_attention: unsupported dtype (need F32, F16, or BF16)");
    }
}

void simple_gla_attention_cpu_impl(Tensor &out,
                                   const Tensor &q,
                                   const Tensor &k,
                                   const Tensor &v,
                                   const Tensor &g_gamma,
                                   float scale) {
    const auto &q_shape = q->shape();
    const size_t B = q_shape[0];
    const size_t T = q_shape[1];
    const size_t H = q_shape[2];
    const size_t D = q_shape[3];

    INFINICORE_ASSERT(k->shape() == q_shape && v->shape() == q_shape);
    INFINICORE_ASSERT(g_gamma->shape().size() == 1 && g_gamma->shape()[0] == H);

    const DataType dtype = q->dtype();
    const std::byte *q_ptr = q->data();
    const std::byte *k_ptr = k->data();
    const std::byte *v_ptr = v->data();
    const std::byte *g_ptr = g_gamma->data();
    std::byte *out_ptr = out->data();

    // Contiguous layout (B, T, H, D): index (b,t,h,d) = b*T*H*D + t*H*D + h*D + d
    const size_t stride_b = T * H * D;
    const size_t stride_t = H * D;
    const size_t stride_h = D;

    // Gate (H,) in float
    std::vector<float> gate(H);
    for (size_t h = 0; h < H; ++h) {
        gate[h] = std::exp(read_float_at(g_ptr, h, g_gamma->dtype()));
    }

    // State S: (B, H, D, D) in float, row-major
    std::vector<float> S(B * H * D * D, 0.f);

    for (size_t t = 0; t < T; ++t) {
        const size_t t_offset = t * stride_t;

        // 1. S = S * gate + outer(k_t, v_t)
        // k_t (b,h,d_k), v_t (b,h,d_v) -> kv(b,h,d_k,d_v) = k_t(b,h,d_k) * v_t(b,h,d_v)
        for (size_t b = 0; b < B; ++b) {
            const size_t b_offset = b * stride_b + t_offset;
            for (size_t h = 0; h < H; ++h) {
                const float g = gate[h];
                float *S_bh = S.data() + (b * H + h) * (D * D);

                // Scale S by gate
                for (size_t i = 0; i < D * D; ++i) {
                    S_bh[i] *= g;
                }

                // Add outer(k_t, v_t)
                for (size_t dk = 0; dk < D; ++dk) {
                    size_t qk_idx = b_offset + h * stride_h + dk;
                    float k_val = read_float_at(k_ptr, qk_idx, dtype);
                    for (size_t dv = 0; dv < D; ++dv) {
                        size_t qv_idx = b_offset + h * stride_h + dv;
                        float v_val = read_float_at(v_ptr, qv_idx, dtype);
                        S_bh[dk * D + dv] += k_val * v_val;
                    }
                }
            }
        }

        // 2. o_t = (q_t * scale) @ S  -> (B, H, D) for each (b,h): o[b,h,:] = scale * (q_t[b,h,:] @ S[b,h,:,:])
        for (size_t b = 0; b < B; ++b) {
            const size_t b_offset = b * stride_b + t_offset;
            for (size_t h = 0; h < H; ++h) {
                const float *S_bh = S.data() + (b * H + h) * (D * D);
                for (size_t dv = 0; dv < D; ++dv) {
                    float acc = 0.f;
                    for (size_t dk = 0; dk < D; ++dk) {
                        size_t q_idx = b_offset + h * stride_h + dk;
                        float q_val = read_float_at(q_ptr, q_idx, dtype) * scale;
                        acc += q_val * S_bh[dk * D + dv];
                    }
                    size_t out_idx = b_offset + h * stride_h + dv;
                    write_float_at(out_ptr, out_idx, dtype, acc);
                }
            }
        }
    }
}

void simple_gla_attention_cpu_calculate(Tensor &out, const Tensor &q, const Tensor &k,
                                        const Tensor &v, const Tensor &g_gamma, float scale) {
    simple_gla_attention_cpu_impl(out, q, k, v, g_gamma, scale);
}

static bool register_cpu = []() {
    SimpleGlaAttention::dispatcher().registerDevice(Device::Type::CPU, &simple_gla_attention_cpu_calculate,
                                                    false);
    return true;
}();

} // namespace

common::OpDispatcher<SimpleGlaAttention::schema> &SimpleGlaAttention::dispatcher() {
    static common::OpDispatcher<SimpleGlaAttention::schema> dispatcher_;
    return dispatcher_;
}

void SimpleGlaAttention::execute(Tensor &out, const Tensor &q, const Tensor &k, const Tensor &v,
                                 const Tensor &g_gamma, float scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q, k, v, g_gamma);
    infinicore::context::setDevice(q->device());
    auto device_type = infinicore::context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);
    if (func == nullptr) {
        throw std::runtime_error("simple_gla_attention: no implementation for device type " + std::to_string(static_cast<int>(device_type)));
    }
    func(out, q, k, v, g_gamma, scale);
}

Tensor simple_gla_attention(const Tensor &q,
                            const Tensor &k,
                            const Tensor &v,
                            const Tensor &g_gamma,
                            float scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q, k, v, g_gamma);

    const auto &q_shape = q->shape();
    INFINICORE_ASSERT(q_shape.size() == 4);
    INFINICORE_ASSERT(k->shape() == q_shape && v->shape() == q_shape);
    INFINICORE_ASSERT(g_gamma->shape().size() == 1 && g_gamma->shape()[0] == q_shape[2]);

    auto q_cont = q->contiguous();
    auto k_cont = k->contiguous();
    auto v_cont = v->contiguous();
    auto g_cont = g_gamma->contiguous();

    auto out = Tensor::empty(q_shape, q->dtype(), q->device());
    SimpleGlaAttention::execute(out, q_cont, k_cont, v_cont, g_cont, scale);
    return out;
}

} // namespace infinicore::op
