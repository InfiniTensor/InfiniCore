#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/gcd.hpp"
#include <cmath>
#include <numeric>
#include <omp.h>
#include <vector>

namespace infinicore::op::gcd_impl::cpu {

template <typename T>
T compute_gcd(T a, T b) {
    return std::gcd(std::abs(a), std::abs(b));
}

template <typename T>
void gcd_contiguous(const T *input_ptr, const T *other_ptr, T *output_ptr, size_t numel) {
#pragma omp parallel for
    for (size_t i = 0; i < numel; ++i) {
        auto a = utils::cast<int64_t>(input_ptr[i]);
        auto b = utils::cast<int64_t>(other_ptr[i]);
        output_ptr[i] = utils::cast<T>(compute_gcd(a, b));
    }
}

template <typename T>
void gcd_strided(const T *input_base, const T *other_base, T *output_base,
                 const std::vector<size_t> &shape,
                 const std::vector<int64_t> &input_strides,
                 const std::vector<int64_t> &other_strides,
                 const std::vector<int64_t> &output_strides) {

    size_t numel = 1;
    for (auto s : shape) {
        numel *= s;
    }
    int ndim = shape.size();

#pragma omp parallel for
    for (size_t i = 0; i < numel; ++i) {
        size_t temp_idx = i;
        size_t input_offset = 0;
        size_t other_offset = 0;
        size_t output_offset = 0;

        for (int d = ndim - 1; d >= 0; --d) {
            size_t coord = temp_idx % shape[d];
            temp_idx /= shape[d];

            input_offset += coord * input_strides[d];
            other_offset += coord * other_strides[d];
            output_offset += coord * output_strides[d];
        }

        auto a = utils::cast<int64_t>(input_base[input_offset]);
        auto b = utils::cast<int64_t>(other_base[other_offset]);

        output_base[output_offset] = utils::cast<T>(compute_gcd(a, b));
    }
}

void calculate(Tensor input, Tensor other, Tensor output) {
    if (input->shape() != other->shape() || input->shape() != output->shape()) {
        throw std::runtime_error("GCD CPU implementation requires all tensors to have the same shape.");
    }

    bool all_contiguous = input->is_contiguous() && other->is_contiguous() && output->is_contiguous();
    auto dtype = input->dtype();
    size_t numel = input->numel();

    if (dtype == DataType::I64) {
        if (all_contiguous) {
            gcd_contiguous<int64_t>(
                reinterpret_cast<int64_t *>(input->data()),
                reinterpret_cast<int64_t *>(other->data()),
                reinterpret_cast<int64_t *>(output->data()), numel);
        } else {
            gcd_strided<int64_t>(
                reinterpret_cast<int64_t *>(input->data()),
                reinterpret_cast<int64_t *>(other->data()),
                reinterpret_cast<int64_t *>(output->data()),
                input->shape(), input->strides(), other->strides(), output->strides());
        }
    } else if (dtype == DataType::I32) {
        if (all_contiguous) {
            gcd_contiguous<int32_t>(
                reinterpret_cast<int32_t *>(input->data()),
                reinterpret_cast<int32_t *>(other->data()),
                reinterpret_cast<int32_t *>(output->data()), numel);
        } else {
            gcd_strided<int32_t>(
                reinterpret_cast<int32_t *>(input->data()),
                reinterpret_cast<int32_t *>(other->data()),
                reinterpret_cast<int32_t *>(output->data()),
                input->shape(), input->strides(), other->strides(), output->strides());
        }
    } else {
        throw std::runtime_error("GCD only supports I32 and I64 on CPU.");
    }
}

static bool registered = []() {
    Gcd::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::gcd_impl::cpu