#pragma once

#include "../device.hpp"
#include "common/op.hpp"

namespace infinicore::op {

class Vdot {
public:
    using schema = void (*)(Tensor out, Tensor a, Tensor b);
    static void execute(Tensor out, Tensor a, Tensor b);
    static common::OpDispatcher<schema> &dispatcher();
};

Tensor vdot(Tensor a, Tensor b);
void vdot_(Tensor out, Tensor a, Tensor b);

} // namespace infinicore::op


