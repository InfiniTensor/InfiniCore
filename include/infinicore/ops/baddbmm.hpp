#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

Tensor baddbmm(Tensor input, Tensor batch1, Tensor batch2, 
                 std::optional<Tensor> beta = std::nullopt, 
                 std::optional<Tensor> alpha = std::nullopt);
void baddbmm_(Tensor out, Tensor input, Tensor batch1, Tensor batch2, 
                 std::optional<Tensor> beta = std::nullopt, 
                 std::optional<Tensor> alpha = std::nullopt);
} // namespace infinicore::op