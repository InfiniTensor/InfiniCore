#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Axpy {
public:
    using schema = void (*)(Tensor, Tensor, Tensor);
    static void execute(Tensor alpha, Tensor x, Tensor y);
    static common::OpDispatcher<schema> &dispatcher();
};

void axpy_(Tensor alpha, Tensor x, Tensor y);

} // namespace infinicore::op
