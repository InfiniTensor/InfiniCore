#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/dot.hpp"
#include "infinicore/tensor.hpp"

namespace infinicore::op::dot_impl::cpu {

void calculate(Tensor c, Tensor a, Tensor b) {
    auto a_shapes = a->shape();
    auto b_shapes = b->shape();
    if (a->ndim() != 1 || b->ndim() != 1) {
        throw std::runtime_error("Dot CPU only supports 1-D tensors for a and b.");
    }
    if (a_shapes[0] != b_shapes[0]) {
        throw std::runtime_error("Dot CPU requires a and b to have the same length.");
    }

    auto dtype = a->dtype();
    if (dtype != b->dtype()) {
        throw std::runtime_error("Dot CPU requires a and b to have the same dtype.");
    }

    const size_t len = a_shapes[0];
    const auto a_stride = a->strides()[0];
    const auto b_stride = b->strides()[0];
    const auto a_element_size = a->element_size();
    const auto b_element_size = b->element_size();
    const auto c_element_size = c->element_size();

    auto a_base = a->data();
    auto b_base = b->data();
    auto c_base = c->data();

    double acc = 0.0;
    for (size_t i = 0; i < len; ++i) {
        const size_t a_off = i * static_cast<size_t>(a_stride);
        const size_t b_off = i * static_cast<size_t>(b_stride);
        if (dtype == DataType::F32) {
            auto *ap = reinterpret_cast<float *>(a_base + a_off * a_element_size);
            auto *bp = reinterpret_cast<float *>(b_base + b_off * b_element_size);
            acc += static_cast<double>((*ap) * (*bp));
        } else if (dtype == DataType::F64) {
            auto *ap = reinterpret_cast<double *>(a_base + a_off * a_element_size);
            auto *bp = reinterpret_cast<double *>(b_base + b_off * b_element_size);
            acc += (*ap) * (*bp);
        } else if (dtype == DataType::F16) {
            auto *ap = reinterpret_cast<fp16_t *>(a_base + a_off * a_element_size);
            auto *bp = reinterpret_cast<fp16_t *>(b_base + b_off * b_element_size);
            float av = utils::cast<float>(*ap);
            float bv = utils::cast<float>(*bp);
            acc += static_cast<double>(av * bv);
        } else if (dtype == DataType::BF16) {
            auto *ap = reinterpret_cast<bf16_t *>(a_base + a_off * a_element_size);
            auto *bp = reinterpret_cast<bf16_t *>(b_base + b_off * b_element_size);
            float av = utils::cast<float>(*ap);
            float bv = utils::cast<float>(*bp);
            acc += static_cast<double>(av * bv);
        } else {
            throw std::runtime_error("Unsupported dtype for dot CPU.");
        }
    }

    if (dtype == DataType::F32) {
        auto *cp = reinterpret_cast<float *>(c_base);
        *cp = static_cast<float>(acc);
    } else if (dtype == DataType::F64) {
        auto *cp = reinterpret_cast<double *>(c_base);
        *cp = acc;
    } else if (dtype == DataType::F16) {
        auto *cp = reinterpret_cast<fp16_t *>(c_base);
        *cp = utils::cast<fp16_t>(static_cast<float>(acc));
    } else if (dtype == DataType::BF16) {
        auto *cp = reinterpret_cast<bf16_t *>(c_base);
        *cp = utils::cast<bf16_t>(static_cast<float>(acc));
    }
}

static bool registered = []() {
    Dot::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::dot_impl::cpu
