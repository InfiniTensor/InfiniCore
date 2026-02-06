#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/atan2.hpp"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <vector>

namespace infinicore::op::atan2_impl::cpu {

template <typename T, typename CompT>
inline T atan2_op(T a, T b) {
    CompT val_a = utils::cast<CompT>(a);
    CompT val_b = utils::cast<CompT>(b);
    return utils::cast<T>(std::atan2(val_a, val_b));
}

template <typename T, typename CompT>
void atan2_kernel(const T *input_ptr, const T *other_ptr, T *output_ptr, size_t numel) {
#pragma omp parallel for
    for (size_t i = 0; i < numel; ++i) {
        output_ptr[i] = atan2_op<T, CompT>(input_ptr[i], other_ptr[i]);
    }
}

template <typename T, typename CompT>
void atan2_strided_kernel(const T *in_data, const T *other_data, T *out_data,
                          const Shape &in_shape, const Strides &in_strides,
                          const Shape &other_shape, const Strides &other_strides,
                          const Shape &out_shape, const Strides &out_strides,
                          size_t numel) {
    int ndim = out_shape.size();
    int in_dim_offset = ndim - in_shape.size();
    int other_dim_offset = ndim - other_shape.size();

#pragma omp parallel for
    for (size_t i = 0; i < numel; ++i) {
        size_t temp_idx = i;
        size_t in_offset = 0;
        size_t other_offset = 0;
        size_t out_offset = 0;

        for (int d = ndim - 1; d >= 0; --d) {
            size_t coord = temp_idx % out_shape[d];
            temp_idx /= out_shape[d];

            out_offset += coord * out_strides[d];

            if (d >= in_dim_offset) {
                int local_d = d - in_dim_offset;
                if (in_shape[local_d] > 1) {
                    in_offset += coord * in_strides[local_d];
                }
            }

            if (d >= other_dim_offset) {
                int local_d = d - other_dim_offset;
                if (other_shape[local_d] > 1) {
                    other_offset += coord * other_strides[local_d];
                }
            }
        }
        out_data[out_offset] = atan2_op<T, CompT>(in_data[in_offset], other_data[other_offset]);
    }
}

void calculate_atan2(Tensor input, Tensor other, Tensor output) {
    auto dtype = input->dtype();
    if (other->dtype() != dtype || output->dtype() != dtype) {
        throw std::runtime_error("Dtype mismatch in atan2 op");
    }

    size_t numel = output->numel();

    bool exact_match = (input->shape() == other->shape()) && (other->shape() == output->shape());
    bool all_contiguous = input->is_contiguous() && other->is_contiguous() && output->is_contiguous();

    if (exact_match && all_contiguous) {
        if (dtype == DataType::F32) {
            atan2_kernel<float, float>(
                reinterpret_cast<float *>(input->data()),
                reinterpret_cast<float *>(other->data()),
                reinterpret_cast<float *>(output->data()), numel);
        } else if (dtype == DataType::F64) {
            atan2_kernel<double, double>(
                reinterpret_cast<double *>(input->data()),
                reinterpret_cast<double *>(other->data()),
                reinterpret_cast<double *>(output->data()), numel);
        } else if (dtype == DataType::F16) {
            atan2_kernel<fp16_t, float>(
                reinterpret_cast<fp16_t *>(input->data()),
                reinterpret_cast<fp16_t *>(other->data()),
                reinterpret_cast<fp16_t *>(output->data()), numel);
        } else if (dtype == DataType::BF16) {
            atan2_kernel<bf16_t, float>(
                reinterpret_cast<bf16_t *>(input->data()),
                reinterpret_cast<bf16_t *>(other->data()),
                reinterpret_cast<bf16_t *>(output->data()), numel);
        } else {
            throw std::runtime_error("Unsupported dtype for atan2 contiguous");
        }
    } else {
        if (dtype == DataType::F32) {
            atan2_strided_kernel<float, float>(
                reinterpret_cast<float *>(input->data()), reinterpret_cast<float *>(other->data()), reinterpret_cast<float *>(output->data()),
                input->shape(), input->strides(), other->shape(), other->strides(), output->shape(), output->strides(), numel);
        } else if (dtype == DataType::F64) {
            atan2_strided_kernel<double, double>(
                reinterpret_cast<double *>(input->data()), reinterpret_cast<double *>(other->data()), reinterpret_cast<double *>(output->data()),
                input->shape(), input->strides(), other->shape(), other->strides(), output->shape(), output->strides(), numel);
        } else if (dtype == DataType::F16) {
            atan2_strided_kernel<fp16_t, float>(
                reinterpret_cast<fp16_t *>(input->data()), reinterpret_cast<fp16_t *>(other->data()), reinterpret_cast<fp16_t *>(output->data()),
                input->shape(), input->strides(), other->shape(), other->strides(), output->shape(), output->strides(), numel);
        } else if (dtype == DataType::BF16) {
            atan2_strided_kernel<bf16_t, float>(
                reinterpret_cast<bf16_t *>(input->data()), reinterpret_cast<bf16_t *>(other->data()), reinterpret_cast<bf16_t *>(output->data()),
                input->shape(), input->strides(), other->shape(), other->strides(), output->shape(), output->strides(), numel);
        } else {
            throw std::runtime_error("Unsupported dtype for atan2 strided");
        }
    }
}

static bool registered = []() {
    Atan2::dispatcher().registerDevice(Device::Type::CPU, &calculate_atan2);
    return true;
}();

} // namespace infinicore::op::atan2_impl::cpu