#include "../../../utils.h"

#include "infinicore/device.hpp"
#include "infinicore/ops/mish.hpp"
#include "infinicore/tensor.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace infinicore::op::mish_impl::cpu {

void calculate(Tensor output, Tensor input, bool inplace) {
    auto input_shapes = input->shape();
    auto output_shapes = output->shape();
    auto input_strides = input->strides();
    auto output_strides = output->strides();
    auto dtype = input->dtype();
    auto dtype_size = input->element_size();
    auto in_base = input->data();
    auto out_base = output->data();

    size_t out_numel = output->numel();

    // 支持 F16 / BF16 / F32 / F64
    if (!(dtype == DataType::F16 || dtype == DataType::BF16 || dtype == DataType::F32 || dtype == DataType::F64)) {
        throw std::runtime_error("Mish CPU supports only F16/BF16/F32/F64.");
    }

    auto softplus = [](double x) {
        // 稳定的 softplus：log1p(exp(-|x|)) + max(x, 0)
        double ax = std::abs(x);
        return std::log1p(std::exp(-ax)) + std::max(x, 0.0);
    };

#pragma omp parallel for
    for (size_t o_idx = 0; o_idx < out_numel; ++o_idx) {
        // 计算输出张量的多维索引
        std::vector<size_t> o_indices(output_shapes.size());
        size_t tmp = o_idx;
        for (int i = static_cast<int>(output_shapes.size()) - 1; i >= 0; --i) {
            o_indices[i] = tmp % output_shapes[i];
            tmp /= output_shapes[i];
        }

        // 计算输入偏移（处理广播）
        size_t in_offset = 0;
        int in_dim_offset = static_cast<int>(output_shapes.size()) - static_cast<int>(input_shapes.size());
        for (int i = 0; i < static_cast<int>(output_shapes.size()); ++i) {
            if (i >= in_dim_offset) {
                int in_i = i - in_dim_offset;
                size_t idx = (input_shapes[in_i] > 1) ? o_indices[i] : 0;
                in_offset += idx * input_strides[in_i];
            }
        }

        // 输出偏移（支持非连续）
        size_t out_offset = 0;
        for (int i = 0; i < static_cast<int>(output_shapes.size()); ++i) {
            out_offset += o_indices[i] * output_strides[i];
        }

        // 读取输入值为 double（根据 dtype 做转换）
        auto in_ptr = in_base + in_offset * dtype_size;
        double x;
        if (dtype == DataType::F32) {
            x = static_cast<double>(*reinterpret_cast<float *>(in_ptr));
        } else if (dtype == DataType::F64) {
            x = *reinterpret_cast<double *>(in_ptr);
        } else if (dtype == DataType::F16) {
            auto *hptr = reinterpret_cast<fp16_t *>(in_ptr);
            float xf = utils::cast<float>(*hptr);
            x = static_cast<double>(xf);
        } else { // BF16
            auto *bptr = reinterpret_cast<bf16_t *>(in_ptr);
            float xf = utils::cast<float>(*bptr);
            x = static_cast<double>(xf);
        }

        // mish: x * tanh(softplus(x))
        double sp = softplus(x);
        double y = x * std::tanh(sp);

        // 写回输出，保持 dtype（使用 utils::cast 简化 F16/BF16）
        auto out_ptr = out_base + out_offset * dtype_size;
        if (dtype == DataType::F32) {
            *reinterpret_cast<float *>(out_ptr) = static_cast<float>(y);
        } else if (dtype == DataType::F64) {
            *reinterpret_cast<double *>(out_ptr) = y;
        } else if (dtype == DataType::F16) {
            auto *hptr = reinterpret_cast<fp16_t *>(out_ptr);
            *hptr = utils::cast<fp16_t>(static_cast<float>(y));
        } else { // BF16
            auto *bptr = reinterpret_cast<bf16_t *>(out_ptr);
            *bptr = utils::cast<bf16_t>(static_cast<float>(y));
        }
    }
}

static bool registered = []() {
    Mish::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::mish_impl::cpu