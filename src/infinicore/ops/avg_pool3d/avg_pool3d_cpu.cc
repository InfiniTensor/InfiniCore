#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/avg_pool3d.hpp"
#include "infinicore/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace infinicore::op::avg_pool3d_impl::cpu {

void calculate(Tensor output, Tensor input, std::tuple<size_t, size_t, size_t> kernel_size, std::tuple<size_t, size_t, size_t> stride, std::tuple<size_t, size_t, size_t> padding, bool ceil_mode) {
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

    // 使用 padding 参数（count_include_pad=True）

    auto input_base = input->data();
    auto output_base = output->data();
    const auto element_size = input->element_size();

    // 无调试输出

    const double kernel_vol = static_cast<double>(kernel_d * kernel_h * kernel_w);

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t od = 0; od < D_out; ++od) {
                long long d_start_raw = (long long)od * (long long)stride_d - (long long)std::get<0>(padding);
                long long d_end_raw = d_start_raw + (long long)kernel_d;
                const size_t d_begin = (size_t)std::max<long long>(0, d_start_raw);
                const size_t d_end = (size_t)std::min<long long>(d_end_raw, (long long)D_in);

                for (size_t oh = 0; oh < H_out; ++oh) {
                    long long h_start_raw = (long long)oh * (long long)stride_h - (long long)std::get<1>(padding);
                    long long h_end_raw = h_start_raw + (long long)kernel_h;
                    const size_t h_begin = (size_t)std::max<long long>(0, h_start_raw);
                    const size_t h_end = (size_t)std::min<long long>(h_end_raw, (long long)H_in);

                    for (size_t ow = 0; ow < W_out; ++ow) {
                        long long w_start_raw = (long long)ow * (long long)stride_w - (long long)std::get<2>(padding);
                        long long w_end_raw = w_start_raw + (long long)kernel_w;
                        const size_t w_begin = (size_t)std::max<long long>(0, w_start_raw);
                        const size_t w_end = (size_t)std::min<long long>(w_end_raw, (long long)W_in);

                        double sum = 0.0;

                        // 累加有效元素（padding 视为 0）
                        for (size_t id = d_begin; id < d_end; ++id) {
                            for (size_t ih = h_begin; ih < h_end; ++ih) {
                                for (size_t iw = w_begin; iw < w_end; ++iw) {
                                    const size_t offset = n * stride_N + c * stride_C + id * stride_D + ih * stride_H + iw * stride_W;

                                    if (dtype == DataType::F32) {
                                        auto *ptr = reinterpret_cast<float *>(input_base + offset * element_size);
                                        sum += static_cast<double>(*ptr);
                                    } else if (dtype == DataType::F64) {
                                        auto *ptr = reinterpret_cast<double *>(input_base + offset * element_size);
                                        sum += *ptr;
                                    } else if (dtype == DataType::F16) {
                                        auto *ptr = reinterpret_cast<fp16_t *>(input_base + offset * element_size);
                                        sum += static_cast<double>(utils::cast<float>(*ptr));
                                    } else {
                                        throw std::runtime_error("Unsupported data type for avg_pool3d operation.");
                                    }
                                }
                            }
                        }

                        const double avg = sum / kernel_vol; // count_include_pad=True

                        const size_t out_offset = n * (C * D_out * H_out * W_out) + c * (D_out * H_out * W_out) + od * (H_out * W_out) + oh * W_out + ow;
                        if (dtype == DataType::F32) {
                            auto *ptr = reinterpret_cast<float *>(output_base + out_offset * element_size);
                            *ptr = static_cast<float>(avg);
                        } else if (dtype == DataType::F64) {
                            auto *ptr = reinterpret_cast<double *>(output_base + out_offset * element_size);
                            *ptr = avg;
                        } else if (dtype == DataType::F16) {
                            auto *ptr = reinterpret_cast<fp16_t *>(output_base + out_offset * element_size);
                            *ptr = utils::cast<fp16_t>(static_cast<float>(avg));
                        }
                    }
                }
            }
        }
    }
}

static bool registered = []() {
    AvgPool3d::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::avg_pool3d_impl::cpu
