#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Lp_Pool1d {
public:
    using schema = void (*)(Tensor, Tensor, float, size_t, size_t, bool);
    static void execute(Tensor output, Tensor input, float norm_type, size_t kernel_size, size_t stride, bool ceil_mode);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor lp_pool1d(Tensor input, float norm_type, size_t kernel_size, size_t stride, bool ceil_mode);
void lp_pool1d_(Tensor output, Tensor input, float norm_type, size_t kernel_size, size_t stride, bool ceil_mode);
} // namespace infinicore::op
