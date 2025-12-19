#include "../../../utils.h"

#include "infinicore/device.hpp"
#include "infinicore/ops/index_select.hpp"
#include "infinicore/tensor.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace infinicore::op::index_select_impl::cpu {

void calculate(Tensor output, Tensor input, int dim, Tensor index) {
    auto input_shapes = input->shape();
    auto index_shapes = index->shape();
    auto output_shapes = output->shape();
    auto input_strides = input->strides();
    auto index_strides = index->strides();
    auto output_strides = output->strides();
    auto dtype = input->dtype();
    auto dtype_size = input->element_size();

    auto input_base = input->data();
    auto index_base = index->data();
    auto output_base = output->data();

    size_t output_numel = output->numel();
    auto ndim = input->ndim();

    // 规范化 dim 到 [0, ndim)
    if (dim < 0) {
        dim = ndim + dim;
    }

    // 获取索引张量的数据类型
    auto index_dtype = index->dtype();
    size_t index_numel = index->numel();

    // 并行遍历输出张量的每个元素
#pragma omp parallel for
    for (size_t output_idx = 0; output_idx < output_numel; ++output_idx) {
        // 计算输出张量的多维索引
        std::vector<size_t> output_indices(ndim);
        size_t temp_idx = output_idx;
        for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
            output_indices[i] = temp_idx % output_shapes[i];
            temp_idx /= output_shapes[i];
        }

        // 构造输入张量的多维索引
        // 对于 dim 维度，需要从 index 张量中读取实际索引值
        std::vector<size_t> input_indices(ndim);
        for (int i = 0; i < static_cast<int>(ndim); ++i) {
            if (i == dim) {
                // 从 index 张量读取索引值
                size_t index_offset = output_indices[i];
                int64_t selected_index = 0;

                if (index_dtype == DataType::I32) {
                    selected_index = static_cast<int64_t>(*reinterpret_cast<int32_t *>(index_base + index_offset * sizeof(int32_t)));
                } else if (index_dtype == DataType::I64) {
                    selected_index = *reinterpret_cast<int64_t *>(index_base + index_offset * sizeof(int64_t));
                } else if (index_dtype == DataType::I8) {
                    selected_index = static_cast<int64_t>(*reinterpret_cast<int8_t *>(index_base + index_offset * sizeof(int8_t)));
                } else if (index_dtype == DataType::I16) {
                    selected_index = static_cast<int64_t>(*reinterpret_cast<int16_t *>(index_base + index_offset * sizeof(int16_t)));
                } else {
                    throw std::runtime_error("Unsupported index data type for index_select operation.");
                }

                // 处理负索引
                if (selected_index < 0) {
                    selected_index += input_shapes[dim];
                }

                input_indices[i] = static_cast<size_t>(selected_index);
            } else {
                input_indices[i] = output_indices[i];
            }
        }

        // 计算输入张量的偏移
        size_t input_offset = 0;
        for (int i = 0; i < static_cast<int>(ndim); ++i) {
            input_offset += input_indices[i] * input_strides[i];
        }

        // 计算输出张量的偏移
        size_t output_offset = 0;
        for (int i = 0; i < static_cast<int>(ndim); ++i) {
            output_offset += output_indices[i] * output_strides[i];
        }

        // 根据数据类型复制数据
        if (dtype == DataType::F32) {
            auto *input_ptr = reinterpret_cast<float *>(input_base + input_offset * dtype_size);
            auto *output_ptr = reinterpret_cast<float *>(output_base + output_offset * dtype_size);
            *output_ptr = *input_ptr;
        } else if (dtype == DataType::F16) {
            auto *input_ptr = reinterpret_cast<fp16_t *>(input_base + input_offset * dtype_size);
            auto *output_ptr = reinterpret_cast<fp16_t *>(output_base + output_offset * dtype_size);
            *output_ptr = *input_ptr;
        } else if (dtype == DataType::I32) {
            auto *input_ptr = reinterpret_cast<int32_t *>(input_base + input_offset * dtype_size);
            auto *output_ptr = reinterpret_cast<int32_t *>(output_base + output_offset * dtype_size);
            *output_ptr = *input_ptr;
        } else if (dtype == DataType::I64) {
            auto *input_ptr = reinterpret_cast<int64_t *>(input_base + input_offset * dtype_size);
            auto *output_ptr = reinterpret_cast<int64_t *>(output_base + output_offset * dtype_size);
            *output_ptr = *input_ptr;
        } else if (dtype == DataType::I8) {
            auto *input_ptr = reinterpret_cast<int8_t *>(input_base + input_offset * dtype_size);
            auto *output_ptr = reinterpret_cast<int8_t *>(output_base + output_offset * dtype_size);
            *output_ptr = *input_ptr;
        } else if (dtype == DataType::I16) {
            auto *input_ptr = reinterpret_cast<int16_t *>(input_base + input_offset * dtype_size);
            auto *output_ptr = reinterpret_cast<int16_t *>(output_base + output_offset * dtype_size);
            *output_ptr = *input_ptr;
        } else if (dtype == DataType::U8) {
            auto *input_ptr = reinterpret_cast<uint8_t *>(input_base + input_offset * dtype_size);
            auto *output_ptr = reinterpret_cast<uint8_t *>(output_base + output_offset * dtype_size);
            *output_ptr = *input_ptr;
        } else if (dtype == DataType::U16) {
            auto *input_ptr = reinterpret_cast<uint16_t *>(input_base + input_offset * dtype_size);
            auto *output_ptr = reinterpret_cast<uint16_t *>(output_base + output_offset * dtype_size);
            *output_ptr = *input_ptr;
        } else if (dtype == DataType::U32) {
            auto *input_ptr = reinterpret_cast<uint32_t *>(input_base + input_offset * dtype_size);
            auto *output_ptr = reinterpret_cast<uint32_t *>(output_base + output_offset * dtype_size);
            *output_ptr = *input_ptr;
        } else if (dtype == DataType::U64) {
            auto *input_ptr = reinterpret_cast<uint64_t *>(input_base + input_offset * dtype_size);
            auto *output_ptr = reinterpret_cast<uint64_t *>(output_base + output_offset * dtype_size);
            *output_ptr = *input_ptr;
        } else if (dtype == DataType::BF16) {
            auto *input_ptr = reinterpret_cast<bf16_t *>(input_base + input_offset * dtype_size);
            auto *output_ptr = reinterpret_cast<bf16_t *>(output_base + output_offset * dtype_size);
            *output_ptr = *input_ptr;
        } else if (dtype == DataType::BOOL) {
            auto *input_ptr = reinterpret_cast<bool *>(input_base + input_offset * dtype_size);
            auto *output_ptr = reinterpret_cast<bool *>(output_base + output_offset * dtype_size);
            *output_ptr = *input_ptr;
        } else {
            throw std::runtime_error("Unsupported data type for index_select operation.");
        }
    }
}

static bool registered = []() {
    IndexSelect::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::index_select_impl::cpu