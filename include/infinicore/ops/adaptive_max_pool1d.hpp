#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class AdaptiveMaxPool1D {
public:
    using schema = void (*)(Tensor, Tensor, size_t);
    static void execute(Tensor y, Tensor x, size_t output_size);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor adaptive_max_pool1d(Tensor x, size_t output_size);
void adaptive_max_pool1d_(Tensor y, Tensor x, size_t output_size);
} // namespace infinicore::op