#pragma once
#include "../device.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {
class PerChannelQuantI8 {
public:
    using schema = void (*)(Tensor, Tensor, Tensor);
    static void execute(Tensor x, Tensor x_packed, Tensor x_scale);
    static common::OpDispatcher<schema> &dispatcher();
};

// Tensor scaled_mm_i8(Tensor a_p, Tensor a_s, Tensor b_p, Tensor b_s, Tensor bias);
void per_channel_quant_i8_(Tensor x, Tensor x_packed, Tensor x_scale);
} // namespace infinicore::op
