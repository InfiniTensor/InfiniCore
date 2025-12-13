#include "../../../utils.h"
#include "infinicore/device.hpp"
#include "infinicore/ops/logsumexp.hpp"
#include "infinicore/tensor.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace infinicore::op::logsumexp_impl::cpu {

void calculate(Tensor input, int dim, bool keepdim, Tensor output) {
    auto input_shapes = input->shape();
    auto input_strides = input->strides();
    auto output_shapes = output->shape();
    auto output_strides = output->strides();
    auto ndim = input->ndim();
    auto dtype = input->dtype();
    auto dtype_size = input->element_size();

    // 规范化 dim 到 [0, ndim)
    if (dim < 0) {
        dim = ndim + dim;
    }

    auto input_base = input->data();
    auto output_base = output->data();

    // 获取约化维度的大小
    size_t reduce_size = input_shapes[dim];
    size_t output_numel = output->numel();

// 对每个输出元素，计算沿着 dim 的 logsumexp
#pragma omp parallel for collapse(1)
    for (size_t output_idx = 0; output_idx < output_numel; ++output_idx) {
        // 根据输出索引计算多维坐标
        std::vector<size_t> output_indices(output_shapes.size());
        size_t temp_idx = output_idx;
        for (int i = static_cast<int>(output_shapes.size()) - 1; i >= 0; --i) {
            output_indices[i] = temp_idx % output_shapes[i];
            temp_idx /= output_shapes[i];
        }

        // 根据输出坐标映射到输入坐标，计算起始位置
        // 对于 keepdim=True: 输出形状 = 输入形状，但减少维度为 1
        // 对于 keepdim=False: 输出形状 < 输入形状，缺少减少的维度
        std::vector<size_t> input_indices(ndim);
        if (keepdim) {
            // 直接对应：输出维度对应输入维度
            for (int i = 0; i < ndim; ++i) {
                if (i == dim) {
                    input_indices[i] = 0; // 减少维度设为 0（我们稍后会遍历）
                } else {
                    input_indices[i] = output_indices[i];
                }
            }
        } else {
            // 跳过减少的维度：输出缺少一个维度
            int output_dim = 0;
            for (int i = 0; i < ndim; ++i) {
                if (i == dim) {
                    input_indices[i] = 0; // 减少维度设为 0
                } else {
                    input_indices[i] = output_indices[output_dim];
                    output_dim++;
                }
            }
        }

        // 计算在输入中的起始偏移
        size_t offset = 0;
        for (int i = 0; i < ndim; ++i) {
            if (i != dim) { // 跳过约化维度
                offset += input_indices[i] * input_strides[i];
            }
        }

        // 计算沿着 reduce 维度的 logsumexp
        if (dtype == DataType::F32) {
            float max_val = -std::numeric_limits<float>::infinity();

            // 第一遍：找最大值
            for (size_t reduce_idx = 0; reduce_idx < reduce_size; ++reduce_idx) {
                size_t current_offset = offset + reduce_idx * input_strides[dim];
                float *input_ptr = reinterpret_cast<float *>(input_base + current_offset * dtype_size);
                max_val = std::max(max_val, *input_ptr);
            }

            // 第二遍：计算 sum(exp(x - max))
            float sum_exp = 0.0f;
            for (size_t reduce_idx = 0; reduce_idx < reduce_size; ++reduce_idx) {
                size_t current_offset = offset + reduce_idx * input_strides[dim];
                float *input_ptr = reinterpret_cast<float *>(input_base + current_offset * dtype_size);
                sum_exp += std::exp(*input_ptr - max_val);
            }

            // 结果：log(sum(exp(x))) = max + log(sum(exp(x - max)))
            float result = max_val + std::log(sum_exp);

            float *output_ptr = reinterpret_cast<float *>(output_base + output_idx * dtype_size);
            *output_ptr = result;

        } else if (dtype == DataType::F16) {
            float max_val = -std::numeric_limits<float>::infinity();

            // 第一遍：找最大值（转换为 F32）
            for (size_t reduce_idx = 0; reduce_idx < reduce_size; ++reduce_idx) {
                size_t current_offset = offset + reduce_idx * input_strides[dim];
                auto *input_ptr = reinterpret_cast<fp16_t *>(input_base + current_offset * dtype_size);
                float val_f32 = utils::cast<float>(*input_ptr);
                max_val = std::max(max_val, val_f32);
            }

            // 第二遍：计算 sum(exp(x - max))
            float sum_exp = 0.0f;
            for (size_t reduce_idx = 0; reduce_idx < reduce_size; ++reduce_idx) {
                size_t current_offset = offset + reduce_idx * input_strides[dim];
                auto *input_ptr = reinterpret_cast<fp16_t *>(input_base + current_offset * dtype_size);
                float val_f32 = utils::cast<float>(*input_ptr);
                sum_exp += std::exp(val_f32 - max_val);
            }

            // 结果：log(sum(exp(x))) = max + log(sum(exp(x - max)))
            float result = max_val + std::log(sum_exp);

            auto *output_ptr = reinterpret_cast<fp16_t *>(output_base + output_idx * dtype_size);
            *output_ptr = utils::cast<fp16_t>(result);

        } else {
            throw std::runtime_error("Unsupported data type for logsumexp operation.");
        }
    }
}

static bool registered = []() {
    LogSumExp::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::logsumexp_impl::cpu
