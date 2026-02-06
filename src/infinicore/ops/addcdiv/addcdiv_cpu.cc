#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/addcdiv.hpp"
#include <cmath>
#include <omp.h>
#include <vector>

namespace infinicore::op::addcdiv_impl::cpu {

template <typename T>
inline T addcdiv_op(T in, T t1, T t2, float value) {
    // out = input + value * (t1 / t2)
    float val_in = utils::cast<float>(in);
    float val_t1 = utils::cast<float>(t1);
    float val_t2 = utils::cast<float>(t2);

    float res = val_in + value * (val_t1 / val_t2);
    return utils::cast<T>(res);
}

template <typename T>
void addcdiv_kernel(const T *in_ptr, const T *t1_ptr, const T *t2_ptr, T *out_ptr, float value, size_t numel) {
#pragma omp parallel for
    for (size_t i = 0; i < numel; ++i) {
        out_ptr[i] = addcdiv_op<T>(in_ptr[i], t1_ptr[i], t2_ptr[i], value);
    }
}

template <typename T>
void addcdiv_strided_kernel(const T *in_ptr, const T *t1_ptr, const T *t2_ptr, T *out_ptr, float value,
                            const Shape &in_shape, const Strides &in_strides,
                            const Shape &t1_shape, const Strides &t1_strides,
                            const Shape &t2_shape, const Strides &t2_strides,
                            const Shape &out_shape, const Strides &out_strides,
                            size_t numel) {
    int ndim = out_shape.size();
    int in_dim_offset = ndim - in_shape.size();
    int t1_dim_offset = ndim - t1_shape.size();
    int t2_dim_offset = ndim - t2_shape.size();

#pragma omp parallel for
    for (size_t i = 0; i < numel; ++i) {
        size_t temp_idx = i;
        size_t in_offset = 0;
        size_t t1_offset = 0;
        size_t t2_offset = 0;
        size_t out_offset = 0;

        for (int d = ndim - 1; d >= 0; --d) {
            size_t coord = temp_idx % out_shape[d];
            temp_idx /= out_shape[d];

            out_offset += coord * out_strides[d];

            if (d >= in_dim_offset && in_shape[d - in_dim_offset] > 1) {
                in_offset += coord * in_strides[d - in_dim_offset];
            }

            if (d >= t1_dim_offset && t1_shape[d - t1_dim_offset] > 1) {
                t1_offset += coord * t1_strides[d - t1_dim_offset];
            }

            if (d >= t2_dim_offset && t2_shape[d - t2_dim_offset] > 1) {
                t2_offset += coord * t2_strides[d - t2_dim_offset];
            }
        }

        out_ptr[out_offset] = addcdiv_op<T>(in_ptr[in_offset], t1_ptr[t1_offset], t2_ptr[t2_offset], value);
    }
}

void calculate_addcdiv(Tensor input, Tensor t1, Tensor t2, Tensor output, float value) {
    auto dtype = input->dtype();
    if (t1->dtype() != dtype || t2->dtype() != dtype || output->dtype() != dtype) {
        throw std::runtime_error("Dtype mismatch in addcdiv op");
    }

    size_t numel = output->numel();

    bool exact_match = (input->shape() == t1->shape()) && (t1->shape() == t2->shape()) && (t2->shape() == output->shape());
    bool all_contiguous = input->is_contiguous() && t1->is_contiguous() && t2->is_contiguous() && output->is_contiguous();

    if (exact_match && all_contiguous) {
        if (dtype == DataType::F32) {
            addcdiv_kernel<float>((float *)input->data(), (float *)t1->data(), (float *)t2->data(), (float *)output->data(), value, numel);
        } else if (dtype == DataType::F16) {
            addcdiv_kernel<fp16_t>((fp16_t *)input->data(), (fp16_t *)t1->data(), (fp16_t *)t2->data(), (fp16_t *)output->data(), value, numel);
        } else {
            throw std::runtime_error("Unsupported dtype for addcdiv contiguous");
        }
    } else {
        if (dtype == DataType::F32) {
            addcdiv_strided_kernel<float>(
                (float *)input->data(), (float *)t1->data(), (float *)t2->data(), (float *)output->data(), value,
                input->shape(), input->strides(), t1->shape(), t1->strides(), t2->shape(), t2->strides(), output->shape(), output->strides(), numel);
        } else if (dtype == DataType::F16) {
            addcdiv_strided_kernel<fp16_t>(
                (fp16_t *)input->data(), (fp16_t *)t1->data(), (fp16_t *)t2->data(), (fp16_t *)output->data(), value,
                input->shape(), input->strides(), t1->shape(), t1->strides(), t2->shape(), t2->strides(), output->shape(), output->strides(), numel);
        } else {
            throw std::runtime_error("Unsupported dtype for addcdiv strided");
        }
    }
}

static bool registered = []() {
    Addcdiv::dispatcher().registerDevice(Device::Type::CPU, &calculate_addcdiv);
    return true;
}();

} // namespace infinicore::op::addcdiv_impl::cpu