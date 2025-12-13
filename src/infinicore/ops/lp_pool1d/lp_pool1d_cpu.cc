#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/lp_pool1d.hpp"
#include "infinicore/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace infinicore::op::lp_pool1d_impl::cpu {

void calculate(Tensor output, Tensor input, float norm_type, size_t kernel_size, size_t stride, bool ceil_mode) {
    // input: [N, C, L_in], output: [N, C, L_out]
    auto input_shapes = input->shape();
    auto input_strides = input->strides();
    auto output_shapes = output->shape();
    auto dtype = input->dtype();

    auto N = input_shapes[0];
    auto C = input_shapes[1];
    auto L_in = input_shapes[2];
    auto L_out = output_shapes[2];

    auto stride_N = input_strides[0];
    auto stride_C = input_strides[1];
    auto stride_L = input_strides[2];

    auto input_base = input->data();
    auto output_base = output->data();
    auto element_size = input->element_size();

    // 遍历所有样本、通道、输出位置
    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C; ++c) {
            for (size_t out_l = 0; out_l < L_out; ++out_l) {
                // 计算窗口的起始位置
                size_t window_start = out_l * stride;
                size_t window_end = std::min(window_start + kernel_size, L_in);

                // 计算 Lp 范数
                double sum_power = 0.0;

                // 标准处理：处理有效元素
                for (size_t i = window_start; i < window_end; ++i) {
                    // 计算元素在内存中的偏移
                    size_t offset = n * stride_N + c * stride_C + i * stride_L;

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

                    // 累加 |val|^norm_type
                    sum_power += std::pow(std::abs(val), norm_type);
                }

                // 处理 replicate padding（仅当 ceil_mode=True 且窗口不完整时）
                if (ceil_mode && window_end < window_start + kernel_size) {
                    // 窗口不完整，需要用 replicate padding
                    // 获取最后一个有效元素
                    size_t last_valid_idx = window_end - 1;
                    size_t offset = n * stride_N + c * stride_C + last_valid_idx * stride_L;

                    double last_val = 0.0;
                    if (dtype == DataType::F32) {
                        auto *ptr = reinterpret_cast<float *>(input_base + offset * element_size);
                        last_val = static_cast<double>(*ptr);
                    } else if (dtype == DataType::F64) {
                        auto *ptr = reinterpret_cast<double *>(input_base + offset * element_size);
                        last_val = static_cast<double>(*ptr);
                    } else if (dtype == DataType::F16) {
                        auto *ptr = reinterpret_cast<fp16_t *>(input_base + offset * element_size);
                        last_val = static_cast<double>(utils::cast<float>(*ptr));
                    }

                    // 重复最后一个有效元素来补全窗口
                    size_t padding_count = (window_start + kernel_size) - window_end;
                    sum_power += padding_count * std::pow(std::abs(last_val), norm_type);
                }

                // 计算 Lp 范数结果：(sum_power)^(1/norm_type)
                double result = std::pow(sum_power, 1.0 / norm_type);

                // 写入输出（output 一定是连续的）
                size_t out_offset = n * (C * L_out) + c * L_out + out_l;

                if (dtype == DataType::F32) {
                    auto *ptr = reinterpret_cast<float *>(output_base + out_offset * element_size);
                    *ptr = static_cast<float>(result);
                } else if (dtype == DataType::F64) {
                    auto *ptr = reinterpret_cast<double *>(output_base + out_offset * element_size);
                    *ptr = result;
                } else if (dtype == DataType::F16) {
                    auto *ptr = reinterpret_cast<fp16_t *>(output_base + out_offset * element_size);
                    *ptr = utils::cast<fp16_t>(static_cast<float>(result));
                }
            }
        }
    }
}

static bool registered = []() {
    Lp_Pool1d::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::lp_pool1d_impl::cpu
