#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/var.hpp"
#include <cmath>
#include <omp.h>
#include <vector>

namespace infinicore::op::var_impl::cpu {

template <typename T, typename AccT>
void var_global_kernel(const T *input_ptr, T *output_ptr, size_t numel, int correction) {

    AccT sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (size_t i = 0; i < numel; ++i) {
        sum += utils::cast<AccT>(input_ptr[i]);
    }
    AccT mean = sum / static_cast<AccT>(numel);

    AccT sum_sq_diff = 0;
#pragma omp parallel for reduction(+ : sum_sq_diff)
    for (size_t i = 0; i < numel; ++i) {
        AccT val = utils::cast<AccT>(input_ptr[i]);
        AccT diff = val - mean;
        sum_sq_diff += diff * diff;
    }

    AccT divisor = static_cast<AccT>(numel) - static_cast<AccT>(correction);
    if (divisor <= 0) {
        *output_ptr = utils::cast<T>(NAN);
    } else {
        *output_ptr = utils::cast<T>(sum_sq_diff / divisor);
    }
}

template <typename T, typename AccT>
void var_global_strided(const T *input_base, T *output_ptr,
                        const std::vector<size_t> &shape,
                        const std::vector<int64_t> &strides,
                        size_t numel, double correction) {
    int ndim = shape.size();

    auto get_val = [&](size_t linear_idx) -> AccT {
        size_t temp = linear_idx;
        int64_t offset = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            size_t coord = temp % shape[d];
            temp /= shape[d];
            offset += static_cast<int64_t>(coord) * strides[d];
        }
        return utils::cast<AccT>(input_base[offset]);
    };

    AccT sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (size_t i = 0; i < numel; ++i) {
        sum += get_val(i);
    }
    AccT mean = sum / static_cast<AccT>(numel);

    AccT sum_sq_diff = 0;
#pragma omp parallel for reduction(+ : sum_sq_diff)
    for (size_t i = 0; i < numel; ++i) {
        AccT val = get_val(i);
        AccT diff = val - mean;
        sum_sq_diff += diff * diff;
    }

    AccT divisor = static_cast<AccT>(numel) - static_cast<AccT>(correction);
    if (divisor <= 0) {
        *output_ptr = utils::cast<T>(NAN);
    } else {
        *output_ptr = utils::cast<T>(sum_sq_diff / divisor);
    }
}

void calculate_global(Tensor input, Tensor output, int correction) {
    bool is_contiguous = input->is_contiguous();

    if (!is_contiguous) {
        auto strides = input->strides();
        auto shape = input->shape();
        int ndim = input->ndim();
        size_t expected = 1;
        is_contiguous = true;
        for (int i = ndim - 1; i >= 0; --i) {
            if (strides[i] != expected && shape[i] > 1) {
                is_contiguous = false;
                break;
            }
            expected *= shape[i];
        }
    }

    auto dtype = input->dtype();
    if (dtype == DataType::F32) {
        if (is_contiguous) {
            var_global_kernel<float, float>((float *)input->data(), (float *)output->data(), input->numel(), correction);
        } else {
            var_global_strided<float, float>((float *)input->data(), (float *)output->data(), input->shape(), input->strides(), input->numel(), correction);
        }
    } else if (dtype == DataType::F16) {
        if (is_contiguous) {
            var_global_kernel<fp16_t, float>((fp16_t *)input->data(), (fp16_t *)output->data(), input->numel(), correction);
        } else {
            var_global_strided<fp16_t, float>((fp16_t *)input->data(), (fp16_t *)output->data(), input->shape(), input->strides(), input->numel(), correction);
        }
    } else if (dtype == DataType::BF16) {
        if (is_contiguous) {
            var_global_kernel<bf16_t, float>((bf16_t *)input->data(), (bf16_t *)output->data(), input->numel(), correction);
        } else {
            var_global_strided<bf16_t, float>((bf16_t *)input->data(), (bf16_t *)output->data(), input->shape(), input->strides(), input->numel(), correction);
        }
    } else {
        throw std::runtime_error("Unsupported dtype");
    }
}

template <typename T, typename AccT>
void var_reduce_impl(const T *input_base, T *output_base,
                     const std::vector<size_t> &input_shape,
                     const std::vector<int64_t> &input_strides,
                     const std::vector<size_t> &output_shape,
                     int dim, int correction) {

    size_t output_numel = 1;
    for (auto s : output_shape) {
        output_numel *= s;
    }

    size_t dim_size = input_shape[dim];
    int64_t dim_stride = input_strides[dim];
    int ndim = input_shape.size();

    std::vector<int64_t> out_to_in_strides;
    std::vector<size_t> out_dims;
    for (int i = 0; i < ndim; ++i) {
        if (i != dim) {
            out_dims.push_back(input_shape[i]);
            out_to_in_strides.push_back(input_strides[i]);
        }
    }

#pragma omp parallel for
    for (size_t out_idx = 0; out_idx < output_numel; ++out_idx) {
        size_t temp = out_idx;
        int64_t input_offset_base = 0;

        for (int d = (int)out_dims.size() - 1; d >= 0; --d) {
            size_t coord = temp % out_dims[d];
            temp /= out_dims[d];
            input_offset_base += static_cast<int64_t>(coord) * out_to_in_strides[d];
        }

        AccT sum = 0;
        for (size_t k = 0; k < dim_size; ++k) {
            const T *ptr = input_base + (input_offset_base + k * dim_stride);
            sum += utils::cast<AccT>(*ptr);
        }
        AccT mean = sum / static_cast<AccT>(dim_size);

        AccT sum_sq_diff = 0;
        for (size_t k = 0; k < dim_size; ++k) {
            const T *ptr = input_base + (input_offset_base + k * dim_stride);
            AccT val = utils::cast<AccT>(*ptr);
            AccT diff = val - mean;
            sum_sq_diff += diff * diff;
        }

        AccT divisor = static_cast<AccT>(dim_size) - static_cast<AccT>(correction);
        if (divisor <= 0) {
            output_base[out_idx] = utils::cast<T>(NAN);
        } else {
            output_base[out_idx] = utils::cast<T>(sum_sq_diff / divisor);
        }
    }
}

void calculate_reduce(Tensor input, Tensor output, int dim, int correction, bool keepdim) {
    auto ndim = input->ndim();
    if (dim < 0) {
        dim = ndim + dim;
    }

    std::vector<size_t> logical_out_shape;
    for (int i = 0; i < ndim; ++i) {
        if (i != dim) {
            logical_out_shape.push_back(input->shape()[i]);
        }
    }

    if (logical_out_shape.empty()) {
        logical_out_shape.push_back(1);
    }

    auto dtype = input->dtype();
    if (dtype == DataType::F32) {
        var_reduce_impl<float, float>(
            (float *)input->data(), (float *)output->data(),
            input->shape(), input->strides(), logical_out_shape, dim, correction);
    } else if (dtype == DataType::F16) {
        var_reduce_impl<fp16_t, float>(
            (fp16_t *)input->data(), (fp16_t *)output->data(),
            input->shape(), input->strides(), logical_out_shape, dim, correction);
    } else if (dtype == DataType::BF16) {
        var_reduce_impl<bf16_t, float>(
            (bf16_t *)input->data(), (bf16_t *)output->data(),
            input->shape(), input->strides(), logical_out_shape, dim, correction);
    } else {
        throw std::runtime_error("Unsupported dtype");
    }
}

static bool registered_global = []() {
    VarGlobal::dispatcher().registerDevice(Device::Type::CPU, &calculate_global);
    return true;
}();

static bool registered_reduce = []() {
    VarReduce::dispatcher().registerDevice(Device::Type::CPU, &calculate_reduce);
    return true;
}();

} // namespace infinicore::op::var_impl::cpu