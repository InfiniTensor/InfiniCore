#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/bucketize.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <omp.h>
#include <vector>

namespace infinicore::op::bucketize_impl::cpu {

template <typename T>
void bucketize_contiguous_kernel(const T *in_ptr, const T *bound_ptr, int64_t *out_ptr,
                                 size_t numel, size_t bound_len, bool right) {
    const T *bound_end = bound_ptr + bound_len;

#pragma omp parallel for
    for (size_t i = 0; i < numel; ++i) {
        T val = in_ptr[i];
        const T *result_ptr;

        if (right) {
            result_ptr = std::upper_bound(bound_ptr, bound_end, val);
        } else {
            result_ptr = std::lower_bound(bound_ptr, bound_end, val);
        }

        out_ptr[i] = static_cast<int64_t>(result_ptr - bound_ptr);
    }
}

template <typename T>
void bucketize_strided_kernel(const T *in_ptr, const T *bound_ptr, int64_t *out_ptr,
                              const Shape &in_shape, const Strides &in_strides,
                              const Shape &out_shape, const Strides &out_strides,
                              size_t numel, size_t bound_len, bool right) {
    int ndim = out_shape.size();
    const T *bound_end = bound_ptr + bound_len;

#pragma omp parallel for
    for (size_t i = 0; i < numel; ++i) {
        size_t temp_idx = i;
        size_t in_offset = 0;
        size_t out_offset = 0;

        for (int d = ndim - 1; d >= 0; --d) {
            size_t coord = temp_idx % out_shape[d];
            temp_idx /= out_shape[d];

            out_offset += coord * out_strides[d];
            in_offset += coord * in_strides[d];
        }

        T val = in_ptr[in_offset];
        const T *result_ptr;

        if (right) {
            result_ptr = std::upper_bound(bound_ptr, bound_end, val);
        } else {
            result_ptr = std::lower_bound(bound_ptr, bound_end, val);
        }

        out_ptr[out_offset] = static_cast<int64_t>(result_ptr - bound_ptr);
    }
}

void calculate_bucketize(Tensor input, Tensor boundaries, Tensor output, bool right) {
    if (output->dtype() != DataType::I64) {
        throw std::runtime_error("Bucketize output must be int64");
    }

    Tensor boundaries_contig = boundaries;
    if (!boundaries->is_contiguous()) {
        boundaries_contig = boundaries->contiguous();
    }

    size_t bound_len = boundaries_contig->numel();
    size_t numel = input->numel();
    auto dtype = input->dtype();

    std::vector<float> sorted_boundaries(bound_len);
    const float *raw_bound_ptr = reinterpret_cast<const float *>(boundaries_contig->data());

    std::memcpy(sorted_boundaries.data(), raw_bound_ptr, bound_len * sizeof(float));

    std::sort(sorted_boundaries.begin(), sorted_boundaries.end());

    const float *bound_ptr = sorted_boundaries.data();

    bool in_out_contiguous = input->is_contiguous() && output->is_contiguous();

    if (in_out_contiguous) {
        int64_t *out_ptr = reinterpret_cast<int64_t *>(output->data());
        if (dtype == DataType::F32) {
            bucketize_contiguous_kernel<float>(
                (float *)input->data(),
                bound_ptr,
                out_ptr, numel, bound_len, right);
        } else if (dtype == DataType::F16) {
            throw std::runtime_error("F16 bucketize cpu not implemented yet");
        } else {
            throw std::runtime_error("Unsupported input dtype");
        }
    } else {
        int64_t *out_ptr = reinterpret_cast<int64_t *>(output->data());
        if (dtype == DataType::F32) {
            bucketize_strided_kernel<float>(
                (float *)input->data(),
                bound_ptr,
                out_ptr,
                input->shape(), input->strides(), output->shape(), output->strides(),
                numel, bound_len, right);
        } else {
            throw std::runtime_error("Unsupported input dtype");
        }
    }
}

static bool registered = []() {
    Bucketize::dispatcher().registerDevice(Device::Type::CPU, &calculate_bucketize);
    return true;
}();

} // namespace infinicore::op::bucketize_impl::cpu