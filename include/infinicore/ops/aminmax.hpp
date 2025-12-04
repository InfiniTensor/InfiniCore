#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <optional>
#include <utility>

namespace infinicore::op {
class Aminmax {
public:
    using schema = void (*)(Tensor, Tensor, Tensor, std::optional<int64_t>, bool);
    static void execute(Tensor min_output, Tensor max_output, Tensor input,
                        std::optional<int64_t> dim, bool keepdim);
    static common::OpDispatcher<schema> &dispatcher();
};

// 返回 (min_tensor, max_tensor) 的 pair
std::pair<Tensor, Tensor> aminmax(Tensor input,
                                  std::optional<int64_t> dim = std::nullopt,
                                  bool keepdim = false);

void aminmax_(Tensor min_output, Tensor max_output, Tensor input,
              std::optional<int64_t> dim = std::nullopt,
              bool keepdim = false);
} // namespace infinicore::op
