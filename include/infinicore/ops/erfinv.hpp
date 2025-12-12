#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {
class Erfinv {
public:
    using schema = void (*)(Tensor, Tensor);
    static void execute(Tensor output, Tensor input);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor erfinv(Tensor input);
void erfinv_(Tensor output, Tensor input);
} // namespace infinicore::op

