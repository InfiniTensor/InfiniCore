#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <tuple>

namespace infinicore::op {
class Fold {
public:
    using schema = void (*)(Tensor, Tensor, std::tuple<size_t, size_t>, std::tuple<size_t, size_t>, std::tuple<size_t, size_t>, std::tuple<size_t, size_t>, std::tuple<size_t, size_t>);
    // Pytorch 文档目前说明了只支持 (N, C, H, W) 和 (C, H, W) 格式的输入输出
    static void execute(Tensor output, Tensor input, std::tuple<size_t, size_t> output_size, std::tuple<size_t, size_t> kernel_size, std::tuple<size_t, size_t> dilation, std::tuple<size_t, size_t> padding, std::tuple<size_t, size_t> stride);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor fold(Tensor input, std::tuple<size_t, size_t> output_size, std::tuple<size_t, size_t> kernel_size, std::tuple<size_t, size_t> dilation, std::tuple<size_t, size_t> padding, std::tuple<size_t, size_t> stride);
void fold_(Tensor output, Tensor input, std::tuple<size_t, size_t> output_size, std::tuple<size_t, size_t> kernel_size, std::tuple<size_t, size_t> dilation, std::tuple<size_t, size_t> padding, std::tuple<size_t, size_t> stride);
} // namespace infinicore::op
