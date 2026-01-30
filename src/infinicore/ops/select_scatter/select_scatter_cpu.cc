#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/select_scatter.hpp"
#include <cstring>
#include <omp.h>
#include <vector>

namespace infinicore::op::select_scatter_impl::cpu {

template <typename T>
void copy_kernel(T *dst_ptr, const std::vector<size_t> &dst_shape, const std::vector<int64_t> &dst_strides,
                 const T *src_ptr, const std::vector<size_t> &src_shape, const std::vector<int64_t> &src_strides) {

    size_t numel = 1;
    for (auto s : dst_shape) {
        numel *= s;
    }
    int ndim = dst_shape.size();

    std::vector<int64_t> effective_src_strides = src_strides;
    for (int i = 0; i < ndim; ++i) {
        if (src_shape[i] == 1 && dst_shape[i] > 1) {
            effective_src_strides[i] = 0;
        }
    }

#pragma omp parallel for
    for (size_t i = 0; i < numel; ++i) {
        size_t temp_idx = i;
        size_t dst_offset = 0;
        size_t src_offset = 0;

        for (int d = ndim - 1; d >= 0; --d) {
            size_t coord = temp_idx % dst_shape[d];
            temp_idx /= dst_shape[d];

            dst_offset += coord * dst_strides[d];
            src_offset += coord * effective_src_strides[d];
        }

        dst_ptr[dst_offset] = utils::cast<T>(src_ptr[src_offset]);
    }
}

void calculate(Tensor input, Tensor src, int64_t dim, int64_t index, Tensor output) {
    auto ndim = input->ndim();
    if (dim < 0) {
        dim += ndim;
    }
    if (index < 0) {
        index += input->shape()[dim];
    }

    size_t total_numel = input->numel();
    auto dtype = input->dtype();

    if (input->is_contiguous() && output->is_contiguous() && input->dtype() == output->dtype()) {
        memcpy(output->data(), input->data(), total_numel * input->element_size());
    } else {

        if (dtype == DataType::F32) {
            copy_kernel<float>(
                reinterpret_cast<float *>(output->data()), output->shape(), output->strides(),
                reinterpret_cast<float *>(input->data()), input->shape(), input->strides());
        } else if (dtype == DataType::F16) {
            copy_kernel<fp16_t>(
                reinterpret_cast<fp16_t *>(output->data()), output->shape(), output->strides(),
                reinterpret_cast<fp16_t *>(input->data()), input->shape(), input->strides());
        } else if (dtype == DataType::BF16) {
            copy_kernel<bf16_t>(
                reinterpret_cast<bf16_t *>(output->data()), output->shape(), output->strides(),
                reinterpret_cast<bf16_t *>(input->data()), input->shape(), input->strides());
        }
    }

    std::vector<size_t> slice_shape = input->shape();
    slice_shape[dim] = 1;

    std::vector<int64_t> slice_strides = output->strides();

    size_t slice_offset_bytes = index * slice_strides[dim] * output->element_size();

    void *slice_data_ptr = reinterpret_cast<char *>(output->data()) + slice_offset_bytes;

    std::vector<size_t> virtual_src_shape = src->shape();
    std::vector<int64_t> virtual_src_strides = src->strides();

    if (virtual_src_shape.size() == ndim - 1) {
        virtual_src_shape.insert(virtual_src_shape.begin() + dim, 1);
        virtual_src_strides.insert(virtual_src_strides.begin() + dim, 0);
    }

    if (dtype == DataType::F32) {
        copy_kernel<float>(
            reinterpret_cast<float *>(slice_data_ptr), slice_shape, slice_strides,
            reinterpret_cast<float *>(src->data()), virtual_src_shape, virtual_src_strides);
    } else if (dtype == DataType::F16) {
        copy_kernel<fp16_t>(
            reinterpret_cast<fp16_t *>(slice_data_ptr), slice_shape, slice_strides,
            reinterpret_cast<fp16_t *>(src->data()), virtual_src_shape, virtual_src_strides);
    } else if (dtype == DataType::BF16) {
        copy_kernel<bf16_t>(
            reinterpret_cast<bf16_t *>(slice_data_ptr), slice_shape, slice_strides,
            reinterpret_cast<bf16_t *>(src->data()), virtual_src_shape, virtual_src_strides);
    } else {
        throw std::runtime_error("Unsupported dtype for select_scatter");
    }
}

static bool registered = []() {
    SelectScatter::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::select_scatter_impl::cpu