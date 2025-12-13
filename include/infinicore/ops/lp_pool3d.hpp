#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <tuple>

namespace infinicore::op {
class Lp_Pool3d {
public:
    using schema = void (*)(Tensor, Tensor, float, const std::tuple<size_t, size_t, size_t>, const std::tuple<size_t, size_t, size_t>, bool);
    static void execute(Tensor output, Tensor input, float norm_type, const std::tuple<size_t, size_t, size_t> kernel_size, const std::tuple<size_t, size_t, size_t> stride, bool ceil_mode);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor lp_pool3d(Tensor input, float norm_type, const std::tuple<size_t, size_t, size_t> kernel_size, const std::tuple<size_t, size_t, size_t> stride, bool ceil_mode);
void lp_pool3d_(Tensor output, Tensor input, float norm_type, const std::tuple<size_t, size_t, size_t> kernel_size, const std::tuple<size_t, size_t, size_t> stride, bool ceil_mode);
} // namespace infinicore::op
