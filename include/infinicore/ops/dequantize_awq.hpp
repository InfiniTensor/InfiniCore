#pragma once
#include "../device.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {
class DequantizeAWQ {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor);
    static void execute(Tensor x, Tensor x_packed, Tensor x_scale, Tensor x_zeros);
    static common::OpDispatcher<schema> &dispatcher();
};

void dequantize_awq_(Tensor x, Tensor x_packed, Tensor x_scale, Tensor x_zeros);
} // namespace infinicore::op
