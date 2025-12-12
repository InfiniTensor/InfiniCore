#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Erf {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor erf(Tensor input);
void erf_(Tensor output, Tensor input);
} // namespace infinicore::op

