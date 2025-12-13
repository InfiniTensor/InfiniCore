#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <tuple>

namespace infinicore::op {
using tuple_size_3d = const std::tuple<size_t, size_t, size_t>;
class Lp_Pool3d {
public:
    using schema = void (*)(Tensor, Tensor, float, tuple_size_3d, tuple_size_3d, bool);
    static void execute(Tensor output, Tensor input, float norm_type, tuple_size_3d kernel_size, tuple_size_3d stride, bool ceil_mode);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor lp_pool3d(Tensor input, float norm_type, tuple_size_3d kernel_size, tuple_size_3d stride, bool ceil_mode);
void lp_pool3d_(Tensor output, Tensor input, float norm_type, tuple_size_3d kernel_size, tuple_size_3d stride, bool ceil_mode);
} // namespace infinicore::op
