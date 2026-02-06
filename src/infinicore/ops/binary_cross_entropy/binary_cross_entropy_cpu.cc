#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/binary_cross_entropy.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <optional>
#include <vector>

namespace infinicore::op::binary_cross_entropy_impl::cpu {

inline float bf16_to_f32(uint16_t val) {
    uint32_t bits = static_cast<uint32_t>(val) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// PyTorch CPU 在逐元素输出时行为接近截断
inline uint16_t f32_to_bf16_trunc(float val) {
    if (std::isnan(val)) {
        return 0x7FC0;
    }
    uint32_t bits;
    std::memcpy(&bits, &val, sizeof(bits));
    return static_cast<uint16_t>(bits >> 16);
}

// Reduction 结果 RNE 舍入
inline uint16_t f32_to_bf16_rne(float val) {
    union {
        float f;
        uint32_t u;
    } x;
    x.f = val;
    if (std::isnan(val)) {
        return 0x7FC0;
    }

    uint32_t lsb = (x.u >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    x.u += rounding_bias;
    return static_cast<uint16_t>(x.u >> 16);
}

template <typename T, typename AccT>
void bce_kernel(const Tensor &input, const Tensor &target, std::optional<Tensor> weight, Tensor &output, std::string reduction) {

    const void *input_raw = input->data();
    const void *target_raw = target->data();
    T *output_data = reinterpret_cast<T *>(output->data());

    const void *weight_raw = nullptr;
    if (weight.has_value() && weight.value()) {
        weight_raw = weight.value()->data();
    }

    size_t numel = input->numel();
    auto input_strides = input->strides();
    auto target_strides = target->strides();
    auto shape = input->shape();
    int ndim = input->ndim();

    bool contiguous = input->is_contiguous() && target->is_contiguous();
    if (weight_raw && !weight.value()->is_contiguous()) {
        contiguous = false;
    }

    auto dtype = input->dtype();

    AccT total_loss = 0;

    auto read_val = [&](const void *ptr, size_t offset) -> AccT {
        if (dtype == DataType::BF16) {
            const uint16_t *p = reinterpret_cast<const uint16_t *>(ptr);
            return static_cast<AccT>(bf16_to_f32(p[offset]));
        } else {
            return utils::cast<AccT>(reinterpret_cast<const T *>(ptr)[offset]);
        }
    };

    auto write_val_elementwise = [&](size_t offset, AccT val) {
        if (dtype == DataType::BF16) {
            reinterpret_cast<uint16_t *>(output_data)[offset] = f32_to_bf16_trunc(static_cast<float>(val));
        } else {
            output_data[offset] = utils::cast<T>(val);
        }
    };

    const AccT eps = static_cast<AccT>(1e-12);
    const AccT one = static_cast<AccT>(1.0);

    if (contiguous) {
#pragma omp parallel for reduction(+ : total_loss)
        for (size_t i = 0; i < numel; ++i) {
            AccT x = read_val(input_raw, i);
            AccT y = read_val(target_raw, i);

            AccT term1 = std::max(x, eps);
            AccT term2 = std::max(one - x, eps);

            AccT loss = -(y * std::log(term1) + (one - y) * std::log(term2));

            if (weight_raw) {
                AccT w = read_val(weight_raw, i);
                loss *= w;
            }

            if (reduction == "none") {
                write_val_elementwise(i, loss);
            } else {
                total_loss += loss;
            }
        }
    } else {
#pragma omp parallel for reduction(+ : total_loss)
        for (size_t i = 0; i < numel; ++i) {
            size_t temp_idx = i;
            size_t input_offset = 0;
            size_t target_offset = 0;
            size_t weight_offset = 0;

            for (int d = ndim - 1; d >= 0; --d) {
                size_t coord = temp_idx % shape[d];
                temp_idx /= shape[d];
                input_offset += coord * input_strides[d];
                target_offset += coord * target_strides[d];
                if (weight_raw) {
                    weight_offset += coord * weight.value()->strides()[d];
                }
            }

            AccT x = read_val(input_raw, input_offset);
            AccT y = read_val(target_raw, target_offset);

            AccT term1 = std::max(x, eps);
            AccT term2 = std::max(one - x, eps);

            AccT loss = -(y * std::log(term1) + (one - y) * std::log(term2));

            if (weight_raw) {
                AccT w = read_val(weight_raw, weight_offset);
                loss *= w;
            }

            if (reduction == "none") {
                write_val_elementwise(i, loss);
            } else {
                total_loss += loss;
            }
        }
    }

    if (reduction != "none") {
        if (reduction == "mean") {
            total_loss /= static_cast<AccT>(numel);
        }

        if (dtype == DataType::BF16) {
            *reinterpret_cast<uint16_t *>(output_data) = f32_to_bf16_rne(static_cast<float>(total_loss));
        } else {
            *output_data = utils::cast<T>(total_loss);
        }
    }
}

void calculate(Tensor input, Tensor target, std::optional<Tensor> weight, Tensor output, std::string reduction) {
    auto dtype = input->dtype();

    if (dtype == DataType::F32) {
        bce_kernel<float, float>(input, target, weight, output, reduction);
    } else if (dtype == DataType::F16) {
        bce_kernel<fp16_t, float>(input, target, weight, output, reduction);
    } else if (dtype == DataType::BF16) {
        bce_kernel<uint16_t, float>(input, target, weight, output, reduction);
    } else if (dtype == DataType::F64) {
        bce_kernel<double, double>(input, target, weight, output, reduction);
    } else {
        throw std::runtime_error("Unsupported dtype for binary_cross_entropy");
    }
}

static bool registered = []() {
    BinaryCrossEntropy::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::binary_cross_entropy_impl::cpu