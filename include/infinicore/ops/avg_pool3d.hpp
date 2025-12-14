#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <tuple>

namespace infinicore::op {
class AvgPool3d {
public:
    using schema = void (*)(Tensor, Tensor, std::tuple<size_t, size_t, size_t>, std::tuple<size_t, size_t, size_t>, std::tuple<size_t, size_t, size_t>, bool);
    static void execute(Tensor output, Tensor input, std::tuple<size_t, size_t, size_t> kernel_size, std::tuple<size_t, size_t, size_t> stride, std::tuple<size_t, size_t, size_t> padding, bool ceil_mode);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor avg_pool3d(Tensor input, std::tuple<size_t, size_t, size_t> kernel_size, std::tuple<size_t, size_t, size_t> stride, std::tuple<size_t, size_t, size_t> padding, bool ceil_mode);
void avg_pool3d_(Tensor output, Tensor input, std::tuple<size_t, size_t, size_t> kernel_size, std::tuple<size_t, size_t, size_t> stride, std::tuple<size_t, size_t, size_t> padding, bool ceil_mode);
} // namespace infinicore::op
