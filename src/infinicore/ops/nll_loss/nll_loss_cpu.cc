#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/nll_loss.hpp"
#include <cmath>
#include <cstring>
#include <omp.h>
#include <optional>
#include <vector>

namespace infinicore::op::nll_loss_impl::cpu {

inline float bf16_to_f32(uint16_t val) {
    uint32_t bits = static_cast<uint32_t>(val) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

template <typename T, typename TargetT = int64_t>
void nll_loss_kernel(const Tensor &input, const Tensor &target, std::optional<Tensor> weight, Tensor &output, int64_t ignore_index) {

    const void *input_raw = input->data();
    const TargetT *target_data = reinterpret_cast<const TargetT *>(target->data());
    T *output_data = reinterpret_cast<T *>(output->data());
    const void *weight_raw = nullptr;

    if (weight.has_value() && weight.value()) {
        weight_raw = weight.value()->data();
    }

    auto input_strides = input->strides();
    size_t batch_size = input->shape()[0];
    size_t n_classes = input->shape()[1];

    int64_t input_stride_n = input_strides[0];
    int64_t input_stride_c = input_strides[1];
    int64_t target_stride = target->strides()[0];
    int64_t weight_stride = (weight.has_value() && weight.value()) ? weight.value()->strides()[0] : 0;

    auto dtype = input->dtype();
    double total_loss = 0.0;
    double total_weight = 0.0;

#pragma omp parallel for reduction(+ : total_loss, total_weight)
    for (size_t i = 0; i < batch_size; ++i) {
        TargetT t = target_data[i * target_stride];

        if (t == ignore_index) {
            continue;
        }

        if (t < 0 || t >= static_cast<TargetT>(n_classes)) {
            continue;
        }

        double w_val = 1.0;
        if (weight_raw) {
            if (dtype == DataType::BF16) {
                const uint16_t *w_ptr = reinterpret_cast<const uint16_t *>(weight_raw);
                w_val = static_cast<double>(bf16_to_f32(w_ptr[t * weight_stride]));
            } else {
                const T *w_ptr = reinterpret_cast<const T *>(weight_raw);
                w_val = utils::cast<double>(w_ptr[t * weight_stride]);
            }
        }

        size_t offset = i * input_stride_n + t * input_stride_c;
        double logit_val = 0.0;

        if (dtype == DataType::BF16) {
            const uint16_t *in_ptr = reinterpret_cast<const uint16_t *>(input_raw);
            logit_val = static_cast<double>(bf16_to_f32(in_ptr[offset]));
        } else {
            const T *in_ptr = reinterpret_cast<const T *>(input_raw);
            logit_val = utils::cast<double>(in_ptr[offset]);
        }

        total_loss += (-logit_val * w_val);
        total_weight += w_val;
    }

    if (total_weight > 0) {
        float res_f = static_cast<float>(total_loss / total_weight);
        if (dtype == DataType::BF16) {
            uint32_t bits;
            std::memcpy(&bits, &res_f, sizeof(bits));
            uint16_t bf16_val = static_cast<uint16_t>(bits >> 16);
            *reinterpret_cast<uint16_t *>(output_data) = bf16_val;
        } else {
            *output_data = utils::cast<T>(res_f);
        }
    } else {
        if (dtype == DataType::BF16) {
            *reinterpret_cast<uint16_t *>(output_data) = 0;
        } else {
            *output_data = utils::cast<T>(0.0f);
        }
    }
}

void calculate(Tensor input, Tensor target, std::optional<Tensor> weight, Tensor output, int64_t ignore_index) {
    auto dtype = input->dtype();

    if (dtype == DataType::F32) {
        nll_loss_kernel<float>(input, target, weight, output, ignore_index);
    } else if (dtype == DataType::F16) {
        nll_loss_kernel<fp16_t>(input, target, weight, output, ignore_index);
    } else if (dtype == DataType::BF16) {
        nll_loss_kernel<uint16_t>(input, target, weight, output, ignore_index);
    } else {
        throw std::runtime_error("Unsupported dtype for nll_loss");
    }
}

static bool registered = []() {
    NLLLoss::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::nll_loss_impl::cpu