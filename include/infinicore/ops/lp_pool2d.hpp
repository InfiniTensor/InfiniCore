#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <tuple>

namespace infinicore::op {
using tuple_size_2d = const std::tuple<size_t, size_t>;
class Lp_Pool2d {
public:
    using schema = void (*)(Tensor, Tensor, float, tuple_size_2d, tuple_size_2d, bool);
    static void execute(Tensor output, Tensor input, float norm_type, tuple_size_2d kernel_size, tuple_size_2d stride, bool ceil_mode);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor lp_pool2d(Tensor input, float norm_type, tuple_size_2d kernel_size, tuple_size_2d stride, bool ceil_mode);
void lp_pool2d_(Tensor output, Tensor input, float norm_type, tuple_size_2d kernel_size, tuple_size_2d stride, bool ceil_mode);
} // namespace infinicore::op
