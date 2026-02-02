#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/glu.hpp"
#include <cmath>
#include <omp.h>

namespace infinicore::op::glu_impl::cpu {

template <typename T>
inline T sigmoid(T x) {
    float val = utils::cast<float>(x);
    float res = 1.0f / (1.0f + std::exp(-val));
    return utils::cast<T>(res);
}

template <typename T>
void calculate_glu_cpu(Tensor input, Tensor output, int dim) {
    auto in_shape = input->shape();
    auto in_strides = input->strides();
    auto out_shape = output->shape();
    auto out_strides = output->strides();
    int ndim = input->ndim();

    if (dim < 0) {
        dim += ndim;
    }

    size_t M = 1;
    for (int i = 0; i < dim; ++i) {
        M *= in_shape[i];
    }
    size_t split_size = out_shape[dim];
    size_t N = 1;
    for (int i = dim + 1; i < ndim; ++i) {
        N *= in_shape[i];
    }

    const T *in_ptr = reinterpret_cast<const T *>(input->data());
    T *out_ptr = reinterpret_cast<T *>(output->data());

#pragma omp parallel for collapse(2)
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            for (size_t k = 0; k < split_size; ++k) {

                size_t offset_base = i * in_strides[dim] * (split_size * 2) + j * in_strides[ndim - 1];
                auto get_offset = [&](size_t d_idx, size_t k_val) {
                    size_t off = 0;
                    size_t remaining = i;
                    for (int d = dim - 1; d >= 0; --d) {
                        off += (remaining % in_shape[d]) * in_strides[d];
                        remaining /= in_shape[d];
                    }
                    off += (k_val)*in_strides[dim];
                    remaining = j;
                    for (int d = ndim - 1; d > dim; --d) {
                        off += (remaining % in_shape[d]) * in_strides[d];
                        remaining /= in_shape[d];
                    }
                    return off;
                };

                size_t off_a = get_offset(dim, k);
                size_t off_b = get_offset(dim, k + split_size);

                size_t off_out = 0;
                size_t rem = i;
                for (int d = dim - 1; d >= 0; --d) {
                    off_out += (rem % out_shape[d]) * out_strides[d];
                    rem /= out_shape[d];
                }
                off_out += k * out_strides[dim];
                rem = j;
                for (int d = ndim - 1; d > dim; --d) {
                    off_out += (rem % out_shape[d]) * out_strides[d];
                    rem /= out_shape[d];
                }

                T a = in_ptr[off_a];
                T b = in_ptr[off_b];

                float val_a = utils::cast<float>(a);
                float sig_b = utils::cast<float>(sigmoid(b));
                out_ptr[off_out] = utils::cast<T>(val_a * sig_b);
            }
        }
    }
}

void calculate(Tensor input, Tensor output, int dim) {
    auto dtype = input->dtype();
    if (dtype == DataType::F32) {
        calculate_glu_cpu<float>(input, output, dim);
    } else if (dtype == DataType::F16) {
        calculate_glu_cpu<fp16_t>(input, output, dim);
    } else if (dtype == DataType::BF16) {
        calculate_glu_cpu<bf16_t>(input, output, dim);
    } else {
        throw std::runtime_error("GLU unsupported dtype: " + toString(dtype));
    }
}

static bool registered = []() {
    Glu::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::glu_impl::cpu