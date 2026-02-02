#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <optional>

namespace infinicore::op {

class NLLLoss {
public:
    using schema = void (*)(Tensor, Tensor, std::optional<Tensor>, Tensor, int64_t);
    static void execute(Tensor input, Tensor target, std::optional<Tensor> weight, Tensor output, int64_t ignore_index);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor nll_loss(Tensor input, Tensor target, std::optional<Tensor> weight, int64_t ignore_index);
void nll_loss_(Tensor input, Tensor target, std::optional<Tensor> weight, Tensor output, int64_t ignore_index);

} // namespace infinicore::op