#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {
class ScaledMMI8 {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, Tensor, Tensor, std::optional<Tensor>);
    static void execute(Tensor c, Tensor a_p, Tensor a_s, Tensor b_p, Tensor b_s, std::optional<Tensor> bias);
    static common::OpDispatcher<schema> &dispatcher();
};

void scaled_mm_i8_(Tensor c, Tensor a_p, Tensor a_s, Tensor b_p, Tensor b_s, std::optional<Tensor> bias);
} // namespace infinicore::op
