#include "infinicore/ops/simple_gla_decode_step.hpp"

#include "infinicore/context/context.hpp"
#include "../../../utils.h"
#include "../../utils.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>

namespace infinicore::op {

namespace {

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
        throw std::runtime_error("simple_gla_decode_step: q/k/v need F32, F16, or BF16");
    }
}

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
        throw std::runtime_error("simple_gla_decode_step: out dtype unsupported");
    }
}

void simple_gla_decode_step_cpu_impl(Tensor &out, Tensor &state, const Tensor &q, const Tensor &k,
                                     const Tensor &v, const Tensor &g_gamma, float scale) {
    const auto &q_shape = q->shape();
    INFINICORE_ASSERT(q_shape.size() == 4 && q_shape[1] == 1);
    INFINICORE_ASSERT(k->shape() == q_shape && v->shape() == q_shape);

    const size_t B = q_shape[0];
    const size_t H = q_shape[2];
    const size_t D = q_shape[3];

    INFINICORE_ASSERT(state->shape().size() == 4 && state->shape()[0] == B && state->shape()[1] == H &&
                      state->shape()[2] == D && state->shape()[3] == D);
    INFINICORE_ASSERT(state->dtype() == DataType::F32);
    INFINICORE_ASSERT(g_gamma->shape().size() == 1 && g_gamma->shape()[0] == H);

    const auto &out_shape = out->shape();
    INFINICORE_ASSERT(out_shape == q_shape);
    INFINICORE_ASSERT(out->dtype() == q->dtype());

    const DataType q_dtype = q->dtype();
    const std::byte *q_ptr = q->data();
    const std::byte *k_ptr = k->data();
    const std::byte *v_ptr = v->data();
    const std::byte *g_ptr = g_gamma->data();
    std::byte *out_ptr = out->data();
    float *s_ptr = reinterpret_cast<float *>(state->data());

    const size_t stride_b = H * D;
    const size_t stride_h = D;

    std::vector<float> gate(H);
    for (size_t h = 0; h < H; ++h) {
        gate[h] = std::exp(read_float_at(g_ptr, h, g_gamma->dtype()));
    }

    const size_t t_offset = 0;

    for (size_t b = 0; b < B; ++b) {
        const size_t b_offset = b * stride_b + t_offset;
        for (size_t h = 0; h < H; ++h) {
            const float g = gate[h];
            float *S_bh = s_ptr + (b * H + h) * (D * D);

            for (size_t i = 0; i < D * D; ++i) {
                S_bh[i] *= g;
            }

            for (size_t dk = 0; dk < D; ++dk) {
                size_t k_idx = b_offset + h * stride_h + dk;
                float k_val = read_float_at(k_ptr, k_idx, q_dtype);
                for (size_t dv = 0; dv < D; ++dv) {
                    size_t v_idx = b_offset + h * stride_h + dv;
                    float v_val = read_float_at(v_ptr, v_idx, q_dtype);
                    S_bh[dk * D + dv] += k_val * v_val;
                }
            }
        }
    }

    for (size_t b = 0; b < B; ++b) {
        const size_t b_offset = b * stride_b + t_offset;
        for (size_t h = 0; h < H; ++h) {
            const float *S_bh = s_ptr + (b * H + h) * (D * D);
            for (size_t dv = 0; dv < D; ++dv) {
                float acc = 0.f;
                for (size_t dk = 0; dk < D; ++dk) {
                    size_t q_idx = b_offset + h * stride_h + dk;
                    float q_val = read_float_at(q_ptr, q_idx, q_dtype) * scale;
                    acc += q_val * S_bh[dk * D + dv];
                }
                size_t out_idx = b_offset + h * stride_h + dv;
                write_float_at(out_ptr, out_idx, q_dtype, acc);
            }
        }
    }
}

void simple_gla_decode_step_cpu_calculate(Tensor &out, Tensor &state, const Tensor &q, const Tensor &k,
                                          const Tensor &v, const Tensor &g_gamma, float scale) {
    simple_gla_decode_step_cpu_impl(out, state, q, k, v, g_gamma, scale);
}

static bool register_cpu = []() {
    SimpleGlaDecodeStep::dispatcher().registerDevice(Device::Type::CPU, &simple_gla_decode_step_cpu_calculate,
                                                      false);
    return true;
}();

} // namespace

common::OpDispatcher<SimpleGlaDecodeStep::schema> &SimpleGlaDecodeStep::dispatcher() {
    static common::OpDispatcher<SimpleGlaDecodeStep::schema> dispatcher_;
    return dispatcher_;
}

void SimpleGlaDecodeStep::execute(Tensor &out, Tensor &state, const Tensor &q, const Tensor &k, const Tensor &v,
                                  const Tensor &g_gamma, float scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q, k, v, g_gamma, state, out);

    const auto &q_shape = q->shape();
    INFINICORE_ASSERT(q_shape.size() == 4 && q_shape[1] == 1);
    INFINICORE_ASSERT(k->shape() == q_shape && v->shape() == q_shape);
    INFINICORE_ASSERT(out->shape() == q_shape && out->dtype() == q->dtype());

    const size_t B = q_shape[0];
    const size_t H = q_shape[2];
    const size_t D = q_shape[3];
    INFINICORE_ASSERT(state->shape().size() == 4 && state->shape()[0] == B && state->shape()[1] == H &&
                      state->shape()[2] == D && state->shape()[3] == D);
    INFINICORE_ASSERT(state->dtype() == DataType::F32);
    INFINICORE_ASSERT(g_gamma->shape().size() == 1 && g_gamma->shape()[0] == H);
    INFINICORE_ASSERT(state->is_contiguous() && out->is_contiguous());

    infinicore::context::setDevice(q->device());
    auto device_type = infinicore::context::getDevice().getType();
    auto func = dispatcher().lookup(device_type);
    if (func == nullptr) {
        throw std::runtime_error("simple_gla_decode_step: no implementation for device type " +
                                 std::to_string(static_cast<int>(device_type)));
    }
    func(out, state, q, k, v, g_gamma, scale);
}

Tensor simple_gla_decode_step(const Tensor &q, const Tensor &k, const Tensor &v, Tensor &state,
                              const Tensor &g_gamma, float scale) {
    INFINICORE_ASSERT_TENSORS_SAME_DEVICE(q, k, v, g_gamma, state);

    const auto &q_shape = q->shape();
    INFINICORE_ASSERT(q_shape.size() == 4 && q_shape[1] == 1);
    INFINICORE_ASSERT(k->shape() == q_shape && v->shape() == q_shape);

    auto q_cont = q->contiguous();
    auto k_cont = k->contiguous();
    auto v_cont = v->contiguous();
    auto g_cont = g_gamma->contiguous();

    auto out = Tensor::empty(q_shape, q->dtype(), q->device());
    SimpleGlaDecodeStep::execute(out, state, q_cont, k_cont, v_cont, g_cont, scale);
    return out;
}

} // namespace infinicore::op
