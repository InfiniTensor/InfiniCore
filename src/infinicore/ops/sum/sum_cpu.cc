#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/sum.hpp"
#include <omp.h>
#include <vector>

namespace infinicore::op::sum_impl::cpu {

template <typename T, typename AccT>
void sum_global_kernel(const T *input_ptr, T *output_ptr, size_t numel) {
    AccT total_sum = 0;

#pragma omp parallel for reduction(+ : total_sum)
    for (size_t i = 0; i < numel; ++i) {
        total_sum += utils::cast<AccT>(input_ptr[i]);
    }

    *output_ptr = utils::cast<T>(total_sum);
}

void calculate_global(Tensor input, Tensor output) {
    bool is_contiguous = true;
    auto strides = input->strides();
    auto shape = input->shape();
    auto ndim = input->ndim();
    size_t expected_stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        if (strides[i] != expected_stride && shape[i] > 1) {
            is_contiguous = false;
            break;
        }
        expected_stride *= shape[i];
    }

    auto dtype = input->dtype();
    size_t numel = input->numel();
    auto output_base = output->data();
    auto input_base = input->data();

    if (is_contiguous) {
        if (dtype == DataType::F32) {
            sum_global_kernel<float, float>(
                reinterpret_cast<float *>(input_base),
                reinterpret_cast<float *>(output_base), numel);
        } else if (dtype == DataType::F64) {
            sum_global_kernel<double, double>(
                reinterpret_cast<double *>(input_base),
                reinterpret_cast<double *>(output_base), numel);
        } else if (dtype == DataType::F16) {
            sum_global_kernel<fp16_t, float>(
                reinterpret_cast<fp16_t *>(input_base),
                reinterpret_cast<fp16_t *>(output_base), numel);
        } else {
            throw std::runtime_error("Unsupported dtype.");
        }
    } else {
        if (dtype == DataType::F16) {
            float total_sum = 0;
            std::vector<size_t> indices(ndim, 0);
            auto *ptr_base = reinterpret_cast<fp16_t *>(input_base);

            for (size_t i = 0; i < numel; ++i) {
                size_t offset = 0;
                for (int d = 0; d < ndim; ++d) {
                    offset += indices[d] * strides[d];
                }
                total_sum += utils::cast<float>(ptr_base[offset]);

                for (int d = ndim - 1; d >= 0; --d) {
                    indices[d]++;
                    if (indices[d] < shape[d]) {
                        break;
                    }
                    indices[d] = 0;
                }
            }
            *reinterpret_cast<fp16_t *>(output_base) = utils::cast<fp16_t>(total_sum);
        } else {
            float total_sum = 0;
            std::vector<size_t> indices(ndim, 0);

            for (size_t i = 0; i < numel; ++i) {
                size_t offset = 0;
                for (int d = 0; d < ndim; ++d) {
                    offset += indices[d] * strides[d];
                }

                if (dtype == DataType::F32) {
                    total_sum += reinterpret_cast<float *>(input_base)[offset];
                } else if (dtype == DataType::F64) {
                    total_sum += reinterpret_cast<double *>(input_base)[offset];
                }

                for (int d = ndim - 1; d >= 0; --d) {
                    indices[d]++;
                    if (indices[d] < shape[d]) {
                        break;
                    }
                    indices[d] = 0;
                }
            }
            if (dtype == DataType::F32) {
                *reinterpret_cast<float *>(output_base) = total_sum;
            } else if (dtype == DataType::F64) {
                *reinterpret_cast<double *>(output_base) = total_sum;
            }
        }
    }
}

template <typename T, typename AccT>
void sum_reduce_contiguous(const T *input_data, T *output_data,
                           const std::vector<size_t> &shape,
                           int dim, size_t numel) {
    int ndim = shape.size();

    size_t dim_size = shape[dim];
    size_t outer_size = 1;
    size_t inner_size = 1;

    for (int i = 0; i < dim; ++i) {
        outer_size *= shape[i];
    }
    for (int i = dim + 1; i < ndim; ++i) {
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
            output_data[i] = utils::cast<T>(sum);
        }
        return;
    }

    size_t output_numel = outer_size * inner_size;

#pragma omp parallel for
    for (size_t o = 0; o < outer_size; ++o) {
        size_t input_base_offset = o * dim_size * inner_size;
        size_t output_base_offset = o * inner_size;

        for (size_t i = 0; i < inner_size; ++i) {
            AccT sum = 0;
            size_t col_offset = input_base_offset + i;

            for (size_t d = 0; d < dim_size; ++d) {
                sum += utils::cast<AccT>(input_data[col_offset + d * inner_size]);
            }
            output_data[output_base_offset + i] = utils::cast<T>(sum);
        }
    }
}

template <typename T, typename AccT>
void sum_reduce_strided(const T *input_base, T *output_base,
                        const std::vector<size_t> &input_shape,
                        const std::vector<int64_t> &input_strides,
                        const std::vector<size_t> &output_shape,
                        int dim) {

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
        output_base[out_idx] = utils::cast<T>(sum);
    }
}

void calculate_reduce(Tensor input, Tensor output, int dim, bool keepdim) {
    auto ndim = input->ndim();
    if (dim < 0) {
        dim = ndim + dim;
    }

    auto dtype = input->dtype();

    bool is_contiguous = true;
    auto strides = input->strides();
    auto shape = input->shape();
    size_t expected_stride = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        if (strides[i] != expected_stride) {
            is_contiguous = false;
            if (shape[i] > 1) {
                break;
            }
        }
        expected_stride *= shape[i];
    }

    if (dtype == DataType::F32) {
        if (is_contiguous) {
            sum_reduce_contiguous<float, float>(
                reinterpret_cast<float *>(input->data()),
                reinterpret_cast<float *>(output->data()),
                shape, dim, input->numel());
        } else {
            sum_reduce_strided<float, float>(
                reinterpret_cast<float *>(input->data()),
                reinterpret_cast<float *>(output->data()),
                shape, strides, output->shape(), dim);
        }
    } else if (dtype == DataType::F64) {
        if (is_contiguous) {
            sum_reduce_contiguous<double, double>(
                reinterpret_cast<double *>(input->data()),
                reinterpret_cast<double *>(output->data()),
                shape, dim, input->numel());
        } else {
            sum_reduce_strided<double, double>(
                reinterpret_cast<double *>(input->data()),
                reinterpret_cast<double *>(output->data()),
                shape, strides, output->shape(), dim);
        }
    } else if (dtype == DataType::F16) {
        if (is_contiguous) {
            sum_reduce_contiguous<fp16_t, float>(
                reinterpret_cast<fp16_t *>(input->data()),
                reinterpret_cast<fp16_t *>(output->data()),
                shape, dim, input->numel());
        } else {
            sum_reduce_strided<fp16_t, float>(
                reinterpret_cast<fp16_t *>(input->data()),
                reinterpret_cast<fp16_t *>(output->data()),
                shape, strides, output->shape(), dim);
        }
    } else {
        throw std::runtime_error("Unsupported data type for sum reduce.");
    }
}

static bool registered_global = []() {
    SumGlobal::dispatcher().registerDevice(Device::Type::CPU, &calculate_global);
    return true;
}();

static bool registered_reduce = []() {
    SumReduce::dispatcher().registerDevice(Device::Type::CPU, &calculate_reduce);
    return true;
}();

} // namespace infinicore::op::sum_impl::cpu