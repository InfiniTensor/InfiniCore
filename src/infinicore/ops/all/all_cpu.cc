#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/all.hpp"
#include <cmath>
#include <cstdio>
#include <iostream>
#include <omp.h>
#include <vector>

namespace infinicore::op::all_impl::cpu {

template <typename T>
inline bool is_false_val(T val) {
    if constexpr (std::is_same_v<T, fp16_t> || std::is_same_v<T, bf16_t>) {
        return utils::cast<float>(val) == 0.0f;
    } else {
        return !static_cast<bool>(val);
    }
}

template <typename T>
void all_global_kernel(const T *input_base, uint8_t *output_ptr,
                       const std::vector<size_t> &shape,
                       const std::vector<int64_t> &strides,
                       size_t numel) {
    size_t ndim = shape.size();
    int global_result = 1;

#pragma omp parallel for reduction(min : global_result)
    for (size_t i = 0; i < numel; ++i) {
        if (global_result == 0) {
            continue;
        }

        size_t temp = i;
        int64_t current_offset = 0;
        for (int d = ndim - 1; d >= 0; --d) {
            size_t dim_sz = shape[d];
            size_t coord = temp % dim_sz;
            temp /= dim_sz;
            current_offset += coord * strides[d];
        }

        if (is_false_val(input_base[current_offset])) {
            global_result = 0;
        }
    }
    *output_ptr = static_cast<uint8_t>(global_result);
}

template <typename T>
void all_global_kernel_contiguous_fast(const T *input_base, uint8_t *output_ptr, size_t numel) {
    int global_result = 1;
#pragma omp parallel for reduction(min : global_result)
    for (size_t i = 0; i < numel; ++i) {
        if (global_result == 0) {
            continue;
        }
        if (is_false_val(input_base[i])) {
            global_result = 0;
        }
    }
    *output_ptr = static_cast<uint8_t>(global_result);
}

void calculate_global(Tensor input, Tensor output) {
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

    size_t numel = input->numel();
    auto input_base = input->data();
    auto output_base = reinterpret_cast<uint8_t *>(output->data());
    auto dtype = input->dtype();

    auto dispatch = [&](auto dummy) {
        using T = decltype(dummy);
        if (is_contiguous) {
            all_global_kernel_contiguous_fast<T>(
                reinterpret_cast<const T *>(input_base), output_base, numel);
        } else {
            all_global_kernel<T>(
                reinterpret_cast<const T *>(input_base), output_base, input->shape(), input->strides(), numel);
        }
    };

    if (dtype == DataType::F32) {
        dispatch(float{});
    } else if (dtype == DataType::F64) {
        dispatch(double{});
    } else if (dtype == DataType::F16) {
        dispatch(fp16_t{});
    } else if (dtype == DataType::BF16) {
        dispatch(bf16_t{});
    } else if (dtype == DataType::BOOL || dtype == DataType::U8) {
        dispatch(uint8_t{});
    } else if (dtype == DataType::I32) {
        dispatch(int32_t{});
    } else if (dtype == DataType::I64) {
        dispatch(int64_t{});
    } else {
        throw std::runtime_error("Unsupported dtype for CPU all (Global).");
    }
}

template <typename T>
void all_reduce_kernel(const T *input_base, uint8_t *output_base,
                       const std::vector<size_t> &input_shape,
                       const std::vector<int64_t> &input_strides,
                       const std::vector<int64_t> &output_strides,
                       int dim,
                       bool keepdim) {

    size_t ndim = input_shape.size();
    size_t dim_size = input_shape[dim];
    int64_t dim_stride = input_strides[dim];

    std::vector<size_t> logical_out_shape;
    std::vector<int64_t> out_to_in_strides;
    std::vector<int64_t> out_to_out_strides;

    size_t output_numel = 1;
    for (size_t i = 0; i < ndim; ++i) {
        if (static_cast<int>(i) != dim) {
            logical_out_shape.push_back(input_shape[i]);
            output_numel *= input_shape[i];
            out_to_in_strides.push_back(input_strides[i]);
            if (keepdim) {
                out_to_out_strides.push_back(output_strides[i]);
            }
        }
    }
    if (!keepdim) {
        out_to_out_strides = output_strides;
    }

    std::vector<uint8_t> temp_results(output_numel);

#pragma omp parallel for
    for (size_t i = 0; i < output_numel; ++i) {
        size_t temp = i;
        int64_t input_offset_base = 0;

        for (int d = static_cast<int>(logical_out_shape.size()) - 1; d >= 0; --d) {
            size_t size = logical_out_shape[d];
            size_t coord = temp % size;
            temp /= size;
            input_offset_base += coord * out_to_in_strides[d];
        }

        int row_result = 1;
        for (size_t j = 0; j < dim_size; ++j) {
            int64_t offset = input_offset_base + j * dim_stride;
            if (is_false_val(input_base[offset])) {
                row_result = 0;
                break;
            }
        }
        temp_results[i] = static_cast<uint8_t>(row_result);
    }

    for (size_t i = 0; i < output_numel; ++i) {
        size_t temp = i;
        int64_t output_offset = 0;

        for (int d = static_cast<int>(logical_out_shape.size()) - 1; d >= 0; --d) {
            size_t size = logical_out_shape[d];
            size_t coord = temp % size;
            temp /= size;
            output_offset += coord * out_to_out_strides[d];
        }

        output_base[output_offset] = temp_results[i];
    }
}

void calculate_reduce(Tensor input, Tensor output, int dim, bool keepdim) {
    auto ndim = input->ndim();
    if (dim < 0) {
        dim = ndim + dim;
    }

    auto input_shape = input->shape();
    auto input_strides = input->strides();
    auto output_strides = output->strides();

    auto input_base = input->data();
    auto output_base = reinterpret_cast<uint8_t *>(output->data());
    auto dtype = input->dtype();

    auto dispatch = [&](auto dummy) {
        using T = decltype(dummy);

        all_reduce_kernel<T>(
            reinterpret_cast<const T *>(input_base),
            output_base,
            input_shape,
            input_strides,
            output_strides,
            dim,
            keepdim);
    };

    if (dtype == DataType::F32) {
        dispatch(float{});
    } else if (dtype == DataType::F64) {
        dispatch(double{});
    } else if (dtype == DataType::F16) {
        dispatch(fp16_t{});
    } else if (dtype == DataType::BF16) {
        dispatch(bf16_t{});
    } else if (dtype == DataType::BOOL || dtype == DataType::U8) {
        dispatch(uint8_t{});
    } else if (dtype == DataType::I32) {
        dispatch(int32_t{});
    } else if (dtype == DataType::I64) {
        dispatch(int64_t{});
    } else {
        throw std::runtime_error("Unsupported dtype for CPU all (Reduce).");
    }
}

static bool registered_global = []() {
    AllGlobal::dispatcher().registerDevice(Device::Type::CPU, &calculate_global);
    return true;
}();

static bool registered_reduce = []() {
    AllReduce::dispatcher().registerDevice(Device::Type::CPU, &calculate_reduce);
    return true;
}();

} // namespace infinicore::op::all_impl::cpu