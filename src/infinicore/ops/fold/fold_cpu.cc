#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/fold.hpp"
#include "infinicore/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace infinicore::op::fold_impl::cpu {

void calculate(Tensor output, Tensor input, std::tuple<size_t, size_t> output_size, std::tuple<size_t, size_t> kernel_size, std::tuple<size_t, size_t> dilation, std::tuple<size_t, size_t> padding, std::tuple<size_t, size_t> stride) {
    // input: [N, C * K_h * K_w, L], output: [N, C, H_out, W_out]
    // L = floor((H_out + 2*pad_h - dil_h*(K_h-1) - 1)/stride_h + 1) *
    //     floor((W_out + 2*pad_w - dil_w*(K_w-1) - 1)/stride_w + 1)

    auto in_shape = input->shape();
    auto in_strides = input->strides();
    auto out_shape = output->shape();
    auto out_strides = output->strides();
    auto dtype = input->dtype();

    const size_t N = in_shape[0];
    const size_t Ckk = in_shape[1]; // C * K_h * K_w
    const size_t L = in_shape[2];   // number of sliding positions

    const size_t C = out_shape[1];
    const size_t H_out = out_shape[2];
    const size_t W_out = out_shape[3];

    const size_t strideN_in = in_strides[0];
    const size_t strideC_in = in_strides[1];
    const size_t strideL_in = in_strides[2];

    const size_t strideN_out = out_strides[0];
    const size_t strideC_out = out_strides[1];
    const size_t strideH_out = out_strides[2];
    const size_t strideW_out = out_strides[3];

    const size_t K_h = std::get<0>(kernel_size);
    const size_t K_w = std::get<1>(kernel_size);
    const size_t S_h = std::get<0>(stride);
    const size_t S_w = std::get<1>(stride);
    const size_t D_h = std::get<0>(dilation);
    const size_t D_w = std::get<1>(dilation);
    const size_t P_h = std::get<0>(padding);
    const size_t P_w = std::get<1>(padding);

    // Basic sanity check
    if (C * K_h * K_w != Ckk) {
        throw std::runtime_error("Input channel dimension is not divisible by kernel size product");
    }

    auto in_base = input->data();
    auto out_base = output->data();
    const auto elem_size = input->element_size();

    // Compute L_h and L_w (number of sliding positions per dimension)
    const size_t L_h = (H_out + 2 * P_h >= D_h * (K_h - 1) + 1)
                           ? (static_cast<size_t>(std::floor((static_cast<double>(H_out) + 2.0 * P_h - static_cast<double>(D_h) * (K_h - 1) - 1) / S_h)) + 1)
                           : 0;
    const size_t L_w = (W_out + 2 * P_w >= D_w * (K_w - 1) + 1)
                           ? (static_cast<size_t>(std::floor((static_cast<double>(W_out) + 2.0 * P_w - static_cast<double>(D_w) * (K_w - 1) - 1) / S_w)) + 1)
                           : 0;
    if (L != L_h * L_w) {
        throw std::runtime_error("Input L does not match computed sliding window count");
    }

    // Zero-initialize output (accumulate)
    std::memset(out_base, 0, out_shape[0] * out_shape[1] * out_shape[2] * out_shape[3] * elem_size);

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t oh = 0; oh < L_h; ++oh) {
                for (size_t ow = 0; ow < L_w; ++ow) {
                    const size_t l_idx = oh * L_w + ow;

                    for (size_t kh = 0; kh < K_h; ++kh) {
                        const long y = static_cast<long>(oh) * static_cast<long>(S_h) - static_cast<long>(P_h) + static_cast<long>(kh) * static_cast<long>(D_h);
                        if (y < 0 || y >= static_cast<long>(H_out)) continue;

                        for (size_t kw = 0; kw < K_w; ++kw) {
                            const long x = static_cast<long>(ow) * static_cast<long>(S_w) - static_cast<long>(P_w) + static_cast<long>(kw) * static_cast<long>(D_w);
                            if (x < 0 || x >= static_cast<long>(W_out)) continue;

                            const size_t ckk = c * (K_h * K_w) + kh * K_w + kw;

                            const size_t in_offset = n * strideN_in + ckk * strideC_in + l_idx * strideL_in;
                            const size_t out_offset = n * strideN_out + c * strideC_out + static_cast<size_t>(y) * strideH_out + static_cast<size_t>(x) * strideW_out;

                            if (dtype == DataType::F32) {
                                auto *in_ptr = reinterpret_cast<float *>(in_base + in_offset * elem_size);
                                auto *out_ptr = reinterpret_cast<float *>(out_base + out_offset * elem_size);
                                *out_ptr += *in_ptr;
                            } else if (dtype == DataType::F64) {
                                auto *in_ptr = reinterpret_cast<double *>(in_base + in_offset * elem_size);
                                auto *out_ptr = reinterpret_cast<double *>(out_base + out_offset * elem_size);
                                *out_ptr += *in_ptr;
                            } else if (dtype == DataType::F16) {
                                auto *in_ptr = reinterpret_cast<fp16_t *>(in_base + in_offset * elem_size);
                                auto *out_ptr = reinterpret_cast<fp16_t *>(out_base + out_offset * elem_size);
                                float acc = utils::cast<float>(*out_ptr);
                                acc += utils::cast<float>(*in_ptr);
                                *out_ptr = utils::cast<fp16_t>(acc);
                            } else {
                                throw std::runtime_error("Unsupported dtype for fold CPU");
                            }
                        }
                    }
                }
            }
        }
    }
}

static bool registered = []() {
    Fold::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::fold_impl::cpu
