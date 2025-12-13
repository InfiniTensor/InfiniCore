#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/lp_pool3d.hpp"
#include "infinicore/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace infinicore::op::lp_pool3d_impl::cpu {

void calculate(Tensor output, Tensor input, float norm_type, const std::tuple<size_t, size_t, size_t> kernel_size, const std::tuple<size_t, size_t, size_t> stride, bool ceil_mode) {
    // input: [N, C, D_in, H_in, W_in], output: [N, C, D_out, H_out, W_out]
    auto input_shapes = input->shape();
    auto input_strides = input->strides();
    auto output_shapes = output->shape();
    auto dtype = input->dtype();

    const size_t N = input_shapes[0];
    const size_t C = input_shapes[1];
    const size_t D_in = input_shapes[2];
    const size_t H_in = input_shapes[3];
    const size_t W_in = input_shapes[4];

    const size_t D_out = output_shapes[2];
    const size_t H_out = output_shapes[3];
    const size_t W_out = output_shapes[4];

    const size_t stride_N = input_strides[0];
    const size_t stride_C = input_strides[1];
    const size_t stride_D = input_strides[2];
    const size_t stride_H = input_strides[3];
    const size_t stride_W = input_strides[4];

    const size_t kernel_d = std::get<0>(kernel_size);
    const size_t kernel_h = std::get<1>(kernel_size);
    const size_t kernel_w = std::get<2>(kernel_size);

    const size_t stride_d = std::get<0>(stride);
    const size_t stride_h = std::get<1>(stride);
    const size_t stride_w = std::get<2>(stride);

    auto input_base = input->data();
    auto output_base = output->data();
    const auto element_size = input->element_size();

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t od = 0; od < D_out; ++od) {
                const size_t d_start = od * stride_d;
                const size_t d_end = std::min(d_start + kernel_d, D_in);

                for (size_t oh = 0; oh < H_out; ++oh) {
                    const size_t h_start = oh * stride_h;
                    const size_t h_end = std::min(h_start + kernel_h, H_in);

                    for (size_t ow = 0; ow < W_out; ++ow) {
                        const size_t w_start = ow * stride_w;
                        const size_t w_end = std::min(w_start + kernel_w, W_in);

                        double sum_power = 0.0;

                        // 累加有效元素
                        for (size_t id = d_start; id < d_end; ++id) {
                            for (size_t ih = h_start; ih < h_end; ++ih) {
                                for (size_t iw = w_start; iw < w_end; ++iw) {
                                    const size_t offset = n * stride_N + c * stride_C + id * stride_D + ih * stride_H + iw * stride_W;

                                    double val = 0.0;
                                    if (dtype == DataType::F32) {
                                        auto *ptr = reinterpret_cast<float *>(input_base + offset * element_size);
                                        val = static_cast<double>(*ptr);
                                    } else if (dtype == DataType::F64) {
                                        auto *ptr = reinterpret_cast<double *>(input_base + offset * element_size);
                                        val = static_cast<double>(*ptr);
                                    } else if (dtype == DataType::F16) {
                                        auto *ptr = reinterpret_cast<fp16_t *>(input_base + offset * element_size);
                                        val = static_cast<double>(utils::cast<float>(*ptr));
                                    }

                                    sum_power += std::pow(std::abs(val), norm_type);
                                }
                            }
                        }

                        // ceil_mode: 放大到完整窗口
                        const size_t valid_d = d_end - d_start;
                        const size_t valid_h = h_end - h_start;
                        const size_t valid_w = w_end - w_start;
                        const size_t valid_cnt = valid_d * valid_h * valid_w;
                        const size_t full_cnt = kernel_d * kernel_h * kernel_w;

                        double scale = 1.0;
                        if (ceil_mode && valid_cnt < full_cnt) {
                            scale = static_cast<double>(full_cnt) / static_cast<double>(valid_cnt);
                        }

                        const double norm = std::pow(sum_power * scale, 1.0 / norm_type);

                        const size_t out_offset = n * (C * D_out * H_out * W_out) + c * (D_out * H_out * W_out) + od * (H_out * W_out) + oh * W_out + ow;
                        if (dtype == DataType::F32) {
                            auto *ptr = reinterpret_cast<float *>(output_base + out_offset * element_size);
                            *ptr = static_cast<float>(norm);
                        } else if (dtype == DataType::F64) {
                            auto *ptr = reinterpret_cast<double *>(output_base + out_offset * element_size);
                            *ptr = norm;
                        } else if (dtype == DataType::F16) {
                            auto *ptr = reinterpret_cast<fp16_t *>(output_base + out_offset * element_size);
                            *ptr = utils::cast<fp16_t>(static_cast<float>(norm));
                        }
                    }
                }
            }
        }
    }
}

static bool registered = []() {
    Lp_Pool3d::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::lp_pool3d_impl::cpu
