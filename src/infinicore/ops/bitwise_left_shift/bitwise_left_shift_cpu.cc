#include "../../../utils.h"

#include "infinicore/device.hpp"
#include "infinicore/ops/bitwise_left_shift.hpp"
#include "infinicore/tensor.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace infinicore::op::bitwise_left_shift_impl::cpu {

void calculate(Tensor c, Tensor a, Tensor b) {
    auto a_shapes = a->shape();
    auto b_shapes = b->shape();
    auto c_shapes = c->shape();
    auto a_strides = a->strides();
    auto b_strides = b->strides();
    auto c_strides = c->strides();
    auto dtype = a->dtype();
    auto dtype_size = a->element_size();

    auto a_base = a->data();
    auto b_base = b->data();
    auto c_base = c->data();

    size_t c_numel = c->numel();

    // 处理广播：逐元素操作
#pragma omp parallel for
    for (size_t c_idx = 0; c_idx < c_numel; ++c_idx) {
        // 计算输出张量的多维索引
        std::vector<size_t> c_indices(c_shapes.size());
        size_t temp_idx = c_idx;
        for (int i = static_cast<int>(c_shapes.size()) - 1; i >= 0; --i) {
            c_indices[i] = temp_idx % c_shapes[i];
            temp_idx /= c_shapes[i];
        }

        // 计算输入张量 a 和 b 的偏移（考虑广播）
        size_t a_offset = 0;
        size_t b_offset = 0;

        // 处理维度差异（广播）
        int a_dim_offset = c_shapes.size() - a_shapes.size();
        int b_dim_offset = c_shapes.size() - b_shapes.size();

        for (int i = 0; i < static_cast<int>(c_shapes.size()); ++i) {
            // 计算 a 的偏移
            if (i >= a_dim_offset) {
                int a_idx = i - a_dim_offset;
                if (a_shapes[a_idx] > 1) {
                    a_offset += c_indices[i] * a_strides[a_idx];
                }
            }

            // 计算 b 的偏移
            if (i >= b_dim_offset) {
                int b_idx = i - b_dim_offset;
                if (b_shapes[b_idx] > 1) {
                    b_offset += c_indices[i] * b_strides[b_idx];
                }
            }
        }

        // 获取位移量的数据类型和大小
        auto b_dtype = b->dtype();
        auto b_dtype_size = b->element_size();

        // 读取位移量（转换为 int）
        int shift_amount = 0;
        if (b_dtype == DataType::I8) {
            shift_amount = static_cast<int>(*reinterpret_cast<int8_t *>(b_base + b_offset * b_dtype_size));
        } else if (b_dtype == DataType::I16) {
            shift_amount = static_cast<int>(*reinterpret_cast<int16_t *>(b_base + b_offset * b_dtype_size));
        } else if (b_dtype == DataType::I32) {
            shift_amount = static_cast<int>(*reinterpret_cast<int32_t *>(b_base + b_offset * b_dtype_size));
        } else if (b_dtype == DataType::I64) {
            shift_amount = static_cast<int>(*reinterpret_cast<int64_t *>(b_base + b_offset * b_dtype_size));
        } else if (b_dtype == DataType::U8) {
            shift_amount = static_cast<int>(*reinterpret_cast<uint8_t *>(b_base + b_offset * b_dtype_size));
        } else if (b_dtype == DataType::U16) {
            shift_amount = static_cast<int>(*reinterpret_cast<uint16_t *>(b_base + b_offset * b_dtype_size));
        } else if (b_dtype == DataType::U32) {
            shift_amount = static_cast<int>(*reinterpret_cast<uint32_t *>(b_base + b_offset * b_dtype_size));
        } else if (b_dtype == DataType::U64) {
            shift_amount = static_cast<int>(*reinterpret_cast<uint64_t *>(b_base + b_offset * b_dtype_size));
        } else {
            throw std::runtime_error("Unsupported shift amount data type for bitwise_left_shift operation.");
        }

        // 计算 c 的偏移（考虑非连续内存布局）
        size_t c_offset = 0;
        for (int i = 0; i < static_cast<int>(c_shapes.size()); ++i) {
            c_offset += c_indices[i] * c_strides[i];
        }

        // 根据数据类型执行按位左移
        if (dtype == DataType::I8) {
            auto *a_ptr = reinterpret_cast<int8_t *>(a_base + a_offset * dtype_size);
            auto *c_ptr = reinterpret_cast<int8_t *>(c_base + c_offset * dtype_size);
            *c_ptr = (shift_amount >= 0 && shift_amount < 8) ? (*a_ptr << shift_amount) : 0;
        } else if (dtype == DataType::I16) {
            auto *a_ptr = reinterpret_cast<int16_t *>(a_base + a_offset * dtype_size);
            auto *c_ptr = reinterpret_cast<int16_t *>(c_base + c_offset * dtype_size);
            *c_ptr = (shift_amount >= 0 && shift_amount < 16) ? (*a_ptr << shift_amount) : 0;
        } else if (dtype == DataType::I32) {
            auto *a_ptr = reinterpret_cast<int32_t *>(a_base + a_offset * dtype_size);
            auto *c_ptr = reinterpret_cast<int32_t *>(c_base + c_offset * dtype_size);
            *c_ptr = (shift_amount >= 0 && shift_amount < 32) ? (*a_ptr << shift_amount) : 0;
        } else if (dtype == DataType::I64) {
            auto *a_ptr = reinterpret_cast<int64_t *>(a_base + a_offset * dtype_size);
            auto *c_ptr = reinterpret_cast<int64_t *>(c_base + c_offset * dtype_size);
            *c_ptr = (shift_amount >= 0 && shift_amount < 64) ? (*a_ptr << shift_amount) : 0;
        } else if (dtype == DataType::U8) {
            auto *a_ptr = reinterpret_cast<uint8_t *>(a_base + a_offset * dtype_size);
            auto *c_ptr = reinterpret_cast<uint8_t *>(c_base + c_offset * dtype_size);
            *c_ptr = (shift_amount >= 0 && shift_amount < 8) ? (*a_ptr << shift_amount) : 0;
        } else if (dtype == DataType::U16) {
            auto *a_ptr = reinterpret_cast<uint16_t *>(a_base + a_offset * dtype_size);
            auto *c_ptr = reinterpret_cast<uint16_t *>(c_base + c_offset * dtype_size);
            *c_ptr = (shift_amount >= 0 && shift_amount < 16) ? (*a_ptr << shift_amount) : 0;
        } else if (dtype == DataType::U32) {
            auto *a_ptr = reinterpret_cast<uint32_t *>(a_base + a_offset * dtype_size);
            auto *c_ptr = reinterpret_cast<uint32_t *>(c_base + c_offset * dtype_size);
            *c_ptr = (shift_amount >= 0 && shift_amount < 32) ? (*a_ptr << shift_amount) : 0;
        } else if (dtype == DataType::U64) {
            auto *a_ptr = reinterpret_cast<uint64_t *>(a_base + a_offset * dtype_size);
            auto *c_ptr = reinterpret_cast<uint64_t *>(c_base + c_offset * dtype_size);
            *c_ptr = (shift_amount >= 0 && shift_amount < 64) ? (*a_ptr << shift_amount) : 0;
        } else {
            throw std::runtime_error("Unsupported data type for bitwise_left_shift operation.");
        }
    }
}

static bool registered = []() {
    BitwiseLeftShift::dispatcher().registerDevice(Device::Type::CPU, &calculate);
    return true;
}();

} // namespace infinicore::op::bitwise_left_shift_impl::cpu
