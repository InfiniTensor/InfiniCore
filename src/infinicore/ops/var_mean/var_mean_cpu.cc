#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/var_mean.hpp"
#include <cmath>
#include <omp.h>
#include <vector>

namespace infinicore::op::var_mean_impl::cpu {

template <typename T, typename AccT>
void var_mean_global_kernel(const T *input_ptr, T *out_var, T *out_mean, size_t numel, int correction) {
    AccT sum = 0;
#pragma omp parallel for reduction(+ : sum)
    for (size_t i = 0; i < numel; ++i) {
        sum += utils::cast<AccT>(input_ptr[i]);
    }

    AccT mean = sum / numel;
    AccT sum_sq_diff = 0;

#pragma omp parallel for reduction(+ : sum_sq_diff)
    for (size_t i = 0; i < numel; ++i) {
        AccT val = utils::cast<AccT>(input_ptr[i]);
        AccT diff = val - mean;
        sum_sq_diff += diff * diff;
    }

    AccT divisor = (numel > (size_t)correction) ? (AccT)(numel - correction) : 0;
    AccT var = (divisor > 0) ? (sum_sq_diff / divisor) : NAN;

    *out_mean = utils::cast<T>(mean);
    *out_var = utils::cast<T>(var);
}

template <typename T, typename AccT>
void var_mean_global_strided(const T *input_base, T *out_var, T *out_mean,
                             const std::vector<size_t> &shape,
                             const std::vector<int64_t> &strides,
                             size_t numel, int correction) {
    int ndim = shape.size();

    AccT sum = 0;

    std::vector<size_t> indices(ndim, 0);
    for (size_t i = 0; i < numel; ++i) {
        size_t offset = 0;
        for (int d = 0; d < ndim; ++d) {
            offset += indices[d] * strides[d];
        }
        sum += utils::cast<AccT>(input_base[offset]);

        for (int d = ndim - 1; d >= 0; --d) {
            indices[d]++;
            if (indices[d] < shape[d]) {
                break;
            }
            indices[d] = 0;
        }
    }
    AccT mean = sum / numel;

    AccT sum_sq_diff = 0;
    std::fill(indices.begin(), indices.end(), 0);
    for (size_t i = 0; i < numel; ++i) {
        size_t offset = 0;
        for (int d = 0; d < ndim; ++d) {
            offset += indices[d] * strides[d];
        }
        AccT val = utils::cast<AccT>(input_base[offset]);
        AccT diff = val - mean;
        sum_sq_diff += diff * diff;

        for (int d = ndim - 1; d >= 0; --d) {
            indices[d]++;
            if (indices[d] < shape[d]) {
                break;
            }
            indices[d] = 0;
        }
    }

    AccT divisor = (numel > (size_t)correction) ? (AccT)(numel - correction) : 0;
    AccT var = (divisor > 0) ? (sum_sq_diff / divisor) : NAN;

    *out_mean = utils::cast<T>(mean);
    *out_var = utils::cast<T>(var);
}

void calculate_global(Tensor input, Tensor out_var, Tensor out_mean, int correction) {
    bool is_contiguous = true;
    auto strides = input->strides();
    auto shape = input->shape();
    auto ndim = input->ndim();
    size_t expected = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        if (strides[i] != expected && shape[i] > 1) {
            is_contiguous = false;
            break;
        }
        expected *= shape[i];
    }

    auto dtype = input->dtype();
    size_t numel = input->numel();
    auto input_base = input->data();

#define DISPATCH_GLOBAL(T_TYPE, ACC_TYPE)                 \
    if (is_contiguous) {                                  \
        var_mean_global_kernel<T_TYPE, ACC_TYPE>(         \
            reinterpret_cast<T_TYPE *>(input_base),       \
            reinterpret_cast<T_TYPE *>(out_var->data()),  \
            reinterpret_cast<T_TYPE *>(out_mean->data()), \
            numel, correction);                           \
    } else {                                              \
        var_mean_global_strided<T_TYPE, ACC_TYPE>(        \
            reinterpret_cast<T_TYPE *>(input_base),       \
            reinterpret_cast<T_TYPE *>(out_var->data()),  \
            reinterpret_cast<T_TYPE *>(out_mean->data()), \
            shape, strides, numel, correction);           \
    }

    if (dtype == DataType::F32) {
        DISPATCH_GLOBAL(float, float);
    } else if (dtype == DataType::F64) {
        DISPATCH_GLOBAL(double, double);
    } else if (dtype == DataType::F16) {
        DISPATCH_GLOBAL(fp16_t, float);
    } else {
        throw std::runtime_error("Unsupported dtype for CPU var_mean (Global).");
    }
#undef DISPATCH_GLOBAL
}

template <typename T, typename AccT>
void var_mean_reduce_contiguous(const T *input_data, T *out_var, T *out_mean,
                                const std::vector<size_t> &shape, int dim, size_t numel, int correction) {
    size_t dim_size = shape[dim];
    size_t outer_size = 1;
    size_t inner_size = 1;
    for (int i = 0; i < dim; ++i) {
        outer_size *= shape[i];
    }
    for (int i = dim + 1; i < (int)shape.size(); ++i) {
        inner_size *= shape[i];
    }

    if (inner_size == 1) {
#pragma omp parallel for
        for (size_t i = 0; i < outer_size; ++i) {
            const T *row_ptr = input_data + i * dim_size;
            AccT sum = 0;
            for (size_t j = 0; j < dim_size; ++j) {
                sum += utils::cast<AccT>(row_ptr[j]);
            }
            AccT mean = sum / dim_size;
            AccT sum_sq_diff = 0;
            for (size_t j = 0; j < dim_size; ++j) {
                AccT diff = utils::cast<AccT>(row_ptr[j]) - mean;
                sum_sq_diff += diff * diff;
            }
            AccT divisor = (dim_size > (size_t)correction) ? (AccT)(dim_size - correction) : 0;
            out_mean[i] = utils::cast<T>(mean);
            out_var[i] = utils::cast<T>((divisor > 0) ? sum_sq_diff / divisor : NAN);
        }
    } else {
#pragma omp parallel for
        for (size_t o = 0; o < outer_size; ++o) {
            size_t input_base = o * dim_size * inner_size;
            size_t output_base = o * inner_size;
            for (size_t i = 0; i < inner_size; ++i) {
                AccT sum = 0;
                for (size_t d = 0; d < dim_size; ++d) {
                    sum += utils::cast<AccT>(input_data[input_base + i + d * inner_size]);
                }
                AccT mean = sum / dim_size;
                AccT sum_sq_diff = 0;
                for (size_t d = 0; d < dim_size; ++d) {
                    AccT diff = utils::cast<AccT>(input_data[input_base + i + d * inner_size]) - mean;
                    sum_sq_diff += diff * diff;
                }
                AccT divisor = (dim_size > (size_t)correction) ? (AccT)(dim_size - correction) : 0;
                out_mean[output_base + i] = utils::cast<T>(mean);
                out_var[output_base + i] = utils::cast<T>((divisor > 0) ? sum_sq_diff / divisor : NAN);
            }
        }
    }
}

template <typename T, typename AccT>
void var_mean_reduce_strided(const T *input_base, T *output_var, T *output_mean,
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

#pragma omp parallel for
    for (size_t out_idx = 0; out_idx < output_numel; ++out_idx) {
        size_t temp_idx = out_idx;
        size_t input_offset_base = 0;

        for (int i = ndim - 1; i >= 0; --i) {
            if (i == dim) {
                continue;
            }
            size_t coord = temp_idx % input_shape[i];
            temp_idx /= input_shape[i];
            input_offset_base += coord * input_strides[i];
        }

        AccT sum = 0;
        for (size_t d = 0; d < dim_size; ++d) {
            const T *ptr = input_base + (input_offset_base + d * dim_stride);
            sum += utils::cast<AccT>(*ptr);
        }
        AccT mean = sum / dim_size;

        AccT sum_sq_diff = 0;
        for (size_t d = 0; d < dim_size; ++d) {
            const T *ptr = input_base + (input_offset_base + d * dim_stride);
            AccT val = utils::cast<AccT>(*ptr);
            AccT diff = val - mean;
            sum_sq_diff += diff * diff;
        }

        AccT divisor = (dim_size > (size_t)correction) ? (AccT)(dim_size - correction) : 0;
        AccT var = (divisor > 0) ? (sum_sq_diff / divisor) : NAN;

        output_mean[out_idx] = utils::cast<T>(mean);
        output_var[out_idx] = utils::cast<T>(var);
    }
}

void calculate_reduce(Tensor input, Tensor out_var, Tensor out_mean, int dim, int correction, bool keepdim) {
    auto ndim = input->ndim();
    if (dim < 0) {
        dim = ndim + dim;
    }

    bool is_contiguous = true;
    auto strides = input->strides();
    auto shape = input->shape();
    size_t expected = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        if (strides[i] != expected && shape[i] > 1) {
            is_contiguous = false;
            break;
        }
        expected *= shape[i];
    }

#define DISPATCH_REDUCE(T_TYPE, ACC_TYPE)                       \
    if (is_contiguous) {                                        \
        var_mean_reduce_contiguous<T_TYPE, ACC_TYPE>(           \
            reinterpret_cast<T_TYPE *>(input->data()),          \
            reinterpret_cast<T_TYPE *>(out_var->data()),        \
            reinterpret_cast<T_TYPE *>(out_mean->data()),       \
            shape, dim, input->numel(), correction);            \
    } else {                                                    \
        var_mean_reduce_strided<T_TYPE, ACC_TYPE>(              \
            reinterpret_cast<T_TYPE *>(input->data()),          \
            reinterpret_cast<T_TYPE *>(out_var->data()),        \
            reinterpret_cast<T_TYPE *>(out_mean->data()),       \
            shape, strides, out_var->shape(), dim, correction); \
    }

    if (input->dtype() == DataType::F32) {
        DISPATCH_REDUCE(float, float);
    } else if (input->dtype() == DataType::F64) {
        DISPATCH_REDUCE(double, double);
    } else if (input->dtype() == DataType::F16) {
        DISPATCH_REDUCE(fp16_t, float);
    } else {
        throw std::runtime_error("Unsupported dtype for CPU var_mean (Reduce).");
    }
#undef DISPATCH_REDUCE
}

static bool registered_global = []() {
    VarMeanGlobal::dispatcher().registerDevice(Device::Type::CPU, &calculate_global);
    return true;
}();

static bool registered_reduce = []() {
    VarMeanReduce::dispatcher().registerDevice(Device::Type::CPU, &calculate_reduce);
    return true;
}();

} // namespace infinicore::op::var_mean_impl::cpu