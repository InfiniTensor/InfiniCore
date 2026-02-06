#pragma once

#include "../device.hpp"
#include "common/op.hpp"
#include <optional>
#include <string>

namespace infinicore::op {

class BinaryCrossEntropy {
public:
    using schema = void (*)(Tensor, Tensor, std::optional<Tensor>, Tensor, std::string);
    static void execute(Tensor input, Tensor target, std::optional<Tensor> weight, Tensor output, std::string reduction);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor binary_cross_entropy(Tensor input, Tensor target, std::optional<Tensor> weight, std::string reduction);
void binary_cross_entropy_(Tensor input, Tensor target, std::optional<Tensor> weight, Tensor output, std::string reduction);

} // namespace infinicore::op