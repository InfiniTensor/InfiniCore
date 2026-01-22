#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/topk.hpp"
#include <algorithm>
#include <cmath>
#include <omp.h>
#include <vector>

namespace infinicore::op::topk_impl::cpu {

template <typename T>
struct Element {
    T value;
    int64_t index;
};

template <typename T, typename ValT = T>
void topk_kernel(const T *input_base, T *values_base, int64_t *indices_base,
                 const std::vector<size_t> &input_shape,
                 const std::vector<int64_t> &input_strides,
                 const std::vector<size_t> &output_shape,
                 const std::vector<int64_t> &values_strides,
                 const std::vector<int64_t> &indices_strides,
                 int k, int dim, bool largest, bool sorted) {

    size_t ndim = input_shape.size();
    size_t dim_size = input_shape[dim];
    int64_t dim_stride = input_strides[dim];

    size_t num_rows = 1;
    for (size_t i = 0; i < ndim; ++i) {
        if (i != dim) {
            num_rows *= input_shape[i];
        }
    }

#pragma omp parallel for
    for (size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
        size_t t = row_idx;
        size_t current_inp_offset = 0;
        size_t current_val_offset = 0;
        size_t current_idx_offset = 0;

        for (int i = ndim - 1; i >= 0; --i) {
            if (i == dim) {
                continue;
            }

            size_t size = input_shape[i];
            size_t coord = t % size;
            t /= size;

            current_inp_offset += coord * input_strides[i];
            current_val_offset += coord * values_strides[i];
            current_idx_offset += coord * indices_strides[i];
        }

        std::vector<Element<ValT>> row_data;
        row_data.reserve(dim_size);
        for (size_t i = 0; i < dim_size; ++i) {
            ValT val = utils::cast<ValT>(input_base[current_inp_offset + i * dim_stride]);
            row_data.push_back({val, static_cast<int64_t>(i)});
        }

        auto cmp = [largest](const Element<ValT> &a, const Element<ValT> &b) {
            bool isnan_a = std::isnan(a.value);
            bool isnan_b = std::isnan(b.value);

            if (isnan_a || isnan_b) {
                if (isnan_a && isnan_b) {
                    return a.index < b.index;
                }

                return largest ? isnan_a : !isnan_a;
            }

            if (a.value != b.value) {
                return largest ? (a.value > b.value) : (a.value < b.value);
            }

            return a.index < b.index;
        };

        if (k < dim_size) {

            std::partial_sort(row_data.begin(), row_data.begin() + k, row_data.end(), cmp);
        } else {

            std::sort(row_data.begin(), row_data.end(), cmp);
        }

        int64_t out_val_dim_stride = values_strides[dim];
        int64_t out_idx_dim_stride = indices_strides[dim];

        for (int i = 0; i < k; ++i) {
            values_base[current_val_offset + i * out_val_dim_stride] = utils::cast<T>(row_data[i].value);
            indices_base[current_idx_offset + i * out_idx_dim_stride] = row_data[i].index;
        }
    }
}

void calculate(Tensor input, Tensor values, Tensor indices, int k, int dim, bool largest, bool sorted) {
    auto input_shape = input->shape();
    auto input_strides = input->strides();
    auto values_strides = values->strides();
    auto indices_strides = indices->strides();
    auto output_shape = values->shape();
    auto dtype = input->dtype();

    if (dtype == DataType::F32) {
        topk_kernel<float, float>(
            reinterpret_cast<float *>(input->data()),
            reinterpret_cast<float *>(values->data()),
            reinterpret_cast<int64_t *>(indices->data()),
            input_shape, input_strides, output_shape, values_strides, indices_strides,
            k, dim, largest, sorted);
    } else if (dtype == DataType::F64) {
        topk_kernel<double, double>(
            reinterpret_cast<double *>(input->data()),
            reinterpret_cast<double *>(values->data()),
            reinterpret_cast<int64_t *>(indices->data()),
            input_shape, input_strides, output_shape, values_strides, indices_strides,
            k, dim, largest, sorted);
    } else if (dtype == DataType::F16) {
        topk_kernel<fp16_t, float>(
            reinterpret_cast<fp16_t *>(input->data()),
            reinterpret_cast<fp16_t *>(values->data()),
            reinterpret_cast<int64_t *>(indices->data()),
            input_shape, input_strides, output_shape, values_strides, indices_strides,
            k, dim, largest, sorted);
    } else if (dtype == DataType::BF16) {
        topk_kernel<bf16_t, float>(
            reinterpret_cast<bf16_t *>(input->data()),
            reinterpret_cast<bf16_t *>(values->data()),
            reinterpret_cast<int64_t *>(indices->data()),
            input_shape, input_strides, output_shape, values_strides, indices_strides,
            k, dim, largest, sorted);
    } else {
        throw std::runtime_error("Unsupported data type for topk.");
    }
}

static bool registered = []() {
    TopK::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::topk_impl::cpu